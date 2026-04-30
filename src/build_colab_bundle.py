"""Build a configurable Colab bundle containing structures and CSV metadata."""

from __future__ import annotations

import argparse
import csv
import subprocess
import shutil
from pathlib import Path

from build_dataset_csv import (
    build_structure_rows,
    validate_rows,
    validate_rows_match_structure_dir,
    write_rows,
)
from project_paths import CATALYTIC_ONLY_SUMMARY_CSV


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

BASE_DATA_DIR = PROJECT_ROOT / ".data" / "train_and_test_sets_structures_non_overlapped_pinmymetal"
TRAIN_DIR = BASE_DATA_DIR / "train"
TEST_DIR = BASE_DATA_DIR / "test"
OUTPUT_DIR = PROJECT_ROOT / ".data" / "Colab_Bundles"
SUMMARY_CSV_BASENAME = CATALYTIC_ONLY_SUMMARY_CSV.name
SITE_SUMMARY_COLUMN_ALIASES = {
    "pdbid": ("pdbid", "structure"),
    "metal residue number": ("metal residue number", "chain_resi"),
    "EC number": ("EC number", "ecnumber"),
    "metal residue type": ("metal residue type", "metaltype"),
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a compressed Colab bundle for train/test structures and CSVs.")
    parser.add_argument("--dataset-root", type=Path, default=BASE_DATA_DIR)
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--test-dir", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--train-summary-csv", type=Path, default=None)
    parser.add_argument("--test-summary-csv", type=Path, default=None)
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument("--csv-output-dir", type=Path, default=None)
    parser.add_argument("--output-bundle", type=Path, default=None)
    parser.set_defaults(allow_multi_metal_structures=True)
    parser.add_argument(
        "--allow-multi-metal-structures",
        action="store_true",
        help=(
            "Allow structure-level CSV rows to contain semicolon-joined metal labels. "
            "This is the default for Colab bundles because training uses the included site-level summary CSVs."
        ),
    )
    parser.add_argument(
        "--strict-single-metal-structures",
        action="store_false",
        dest="allow_multi_metal_structures",
        help="Fail if a generated structure-level CSV row would contain more than one metal label.",
    )
    parser.add_argument("--ec-label-depth", type=int, default=1)
    parser.add_argument("--skip-bundle", action="store_true")
    return parser.parse_args()


def validate_inputs(paths: list[Path]) -> None:
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Bundle input path not found: {path}")
    if shutil.which("zstd") is None:
        raise RuntimeError(
            "zstd is not installed or not found in PATH.\n"
            "Install it on Ubuntu with:\n"
            "sudo apt update && sudo apt install zstd"
        )


def resolve_split_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    dataset_root = args.dataset_root
    train_dir = args.train_dir if args.train_dir is not None else dataset_root / "train"
    test_dir = args.test_dir if args.test_dir is not None else dataset_root / "test"
    return train_dir, test_dir


def _default_summary_csv_for_structure_dir(structure_dir: Path, *, split_name: str) -> Path:
    local_summary_csv = structure_dir / SUMMARY_CSV_BASENAME
    if local_summary_csv.exists():
        return local_summary_csv
    raise FileNotFoundError(
        f"Expected MAHOMES-style site-level {split_name} summary CSV at {local_summary_csv}. "
        f"Pass --{split_name}-summary-csv explicitly if it lives elsewhere."
    )


def validate_site_level_summary_csv(summary_csv: Path, *, split_name: str) -> None:
    if not summary_csv.exists():
        raise FileNotFoundError(f"{split_name.capitalize()} site-level summary CSV not found: {summary_csv}")
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = {field.strip().lower() for field in (reader.fieldnames or []) if field}
    missing_columns = [
        canonical_name
        for canonical_name, aliases in SITE_SUMMARY_COLUMN_ALIASES.items()
        if not fieldnames.intersection(alias.lower() for alias in aliases)
    ]
    if missing_columns:
        raise ValueError(
            f"{split_name.capitalize()} summary CSV {summary_csv} is not a site-level MAHOMES summary. "
            f"Missing columns compatible with: {', '.join(missing_columns)}."
        )


def resolve_summary_csv_paths(
    args: argparse.Namespace,
    *,
    train_dir: Path,
    test_dir: Path,
) -> tuple[Path | None, Path | None]:
    train_summary_csv = args.train_summary_csv
    test_summary_csv = args.test_summary_csv

    if args.summary_csv is not None:
        if train_summary_csv is None:
            train_summary_csv = args.summary_csv
        if test_summary_csv is None:
            test_summary_csv = args.summary_csv

    if train_summary_csv is None and args.train_csv is None:
        train_summary_csv = _default_summary_csv_for_structure_dir(train_dir, split_name="train")
    if test_summary_csv is None and args.test_csv is None:
        test_summary_csv = _default_summary_csv_for_structure_dir(test_dir, split_name="test")
    return train_summary_csv, test_summary_csv


def default_csv_output_dir(dataset_root: Path) -> Path:
    return OUTPUT_DIR / dataset_root.name


def default_output_bundle(dataset_root: Path) -> Path:
    return OUTPUT_DIR / f"{dataset_root.name}_colab_bundle_structures.tar.zst"


def ensure_project_relative(path: Path) -> str:
    resolved_path = path.resolve()
    try:
        return str(resolved_path.relative_to(PROJECT_ROOT))
    except ValueError as exc:
        raise ValueError(
            f"Bundle input path must live under the project root {PROJECT_ROOT}: {resolved_path}"
        ) from exc


def path_is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def append_unique_path(paths: list[Path], path: Path) -> None:
    resolved = path.resolve()
    if all(existing.resolve() != resolved for existing in paths):
        paths.append(path)


def generate_structure_csv(
    *,
    structure_dir: Path,
    summary_csv: Path,
    output_csv: Path,
    allow_multi_metal_structures: bool,
    ec_label_depth: int,
) -> tuple[Path, int]:
    rows = build_structure_rows(
        structure_dir=structure_dir,
        summary_csv=summary_csv,
        allow_multi_metal_structures=allow_multi_metal_structures,
        ec_label_depth=ec_label_depth,
    )
    validate_rows(rows)
    validate_rows_match_structure_dir(structure_dir=structure_dir, rows=rows)
    write_rows(output_csv, rows)
    multi_metal_row_count = sum(1 for row in rows if ";" in row["metal_type"])
    return output_csv, multi_metal_row_count


def prepare_csv_artifacts(
    args: argparse.Namespace,
    *,
    dataset_root: Path,
    train_dir: Path,
    test_dir: Path,
    train_summary_csv: Path | None,
    test_summary_csv: Path | None,
) -> tuple[Path | None, Path | None, int, int]:
    csv_output_dir = args.csv_output_dir or default_csv_output_dir(dataset_root)
    csv_output_dir.mkdir(parents=True, exist_ok=True)

    train_csv = args.train_csv
    train_multi_metal_rows = 0
    if train_csv is None:
        if train_summary_csv is None:
            raise ValueError("A train summary CSV is required to generate the train CSV artifact.")
        train_csv = csv_output_dir / f"{dataset_root.name}_train.csv"
        train_csv, train_multi_metal_rows = generate_structure_csv(
            structure_dir=train_dir,
            summary_csv=train_summary_csv,
            output_csv=train_csv,
            allow_multi_metal_structures=args.allow_multi_metal_structures,
            ec_label_depth=args.ec_label_depth,
        )

    test_csv = args.test_csv
    test_multi_metal_rows = 0
    if test_csv is None:
        if test_summary_csv is None:
            raise ValueError("A test summary CSV is required to generate the test CSV artifact.")
        test_csv = csv_output_dir / f"{dataset_root.name}_test.csv"
        test_csv, test_multi_metal_rows = generate_structure_csv(
            structure_dir=test_dir,
            summary_csv=test_summary_csv,
            output_csv=test_csv,
            allow_multi_metal_structures=args.allow_multi_metal_structures,
            ec_label_depth=args.ec_label_depth,
        )
    return train_csv, test_csv, train_multi_metal_rows, test_multi_metal_rows


def format_multi_metal_note(row_count: int) -> str:
    if row_count <= 0:
        return ""
    return f" ({row_count} multi-metal structure row(s) use semicolon-joined labels)"


def build_bundle(selected_paths: list[Path], *, output_bundle: Path) -> Path:
    validate_inputs(selected_paths)

    output_bundle.parent.mkdir(parents=True, exist_ok=True)
    relative_paths = [ensure_project_relative(path) for path in selected_paths]
    cmd = [
        "tar",
        "--use-compress-program=zstd -T0 -19",
        "-cf",
        str(output_bundle),
        "-C",
        str(PROJECT_ROOT),
        *relative_paths,
    ]
    subprocess.run(cmd, check=True)
    return output_bundle


def main() -> None:
    args = parse_args()
    train_dir, test_dir = resolve_split_dirs(args)
    train_summary_csv, test_summary_csv = resolve_summary_csv_paths(args, train_dir=train_dir, test_dir=test_dir)
    dataset_root = args.dataset_root
    required_paths = [dataset_root, train_dir, test_dir]
    if train_summary_csv is not None:
        required_paths.append(train_summary_csv)
    if test_summary_csv is not None:
        required_paths.append(test_summary_csv)
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required bundle input path not found: {path}")
    if train_summary_csv is None or test_summary_csv is None:
        raise ValueError("Train and test site-level summary CSVs are required for a runnable Colab bundle.")
    validate_site_level_summary_csv(train_summary_csv, split_name="train")
    validate_site_level_summary_csv(test_summary_csv, split_name="test")

    train_csv, test_csv, train_multi_metal_rows, test_multi_metal_rows = prepare_csv_artifacts(
        args,
        dataset_root=dataset_root,
        train_dir=train_dir,
        test_dir=test_dir,
        train_summary_csv=train_summary_csv,
        test_summary_csv=test_summary_csv,
    )

    selected_paths = [train_dir, test_dir]
    if not path_is_inside(train_summary_csv, train_dir):
        append_unique_path(selected_paths, train_summary_csv)
    if not path_is_inside(test_summary_csv, test_dir):
        append_unique_path(selected_paths, test_summary_csv)
    if train_csv is not None:
        append_unique_path(selected_paths, train_csv)
    if test_csv is not None:
        append_unique_path(selected_paths, test_csv)

    if args.skip_bundle:
        print(f"Prepared train directory: {train_dir}")
        print(f"Prepared test directory: {test_dir}")
        print(f"Verified train site-level summary CSV: {train_summary_csv}")
        print(f"Verified test site-level summary CSV: {test_summary_csv}")
        if train_csv is not None:
            print(f"Prepared train CSV: {train_csv}{format_multi_metal_note(train_multi_metal_rows)}")
        if test_csv is not None:
            print(f"Prepared test CSV: {test_csv}{format_multi_metal_note(test_multi_metal_rows)}")
        return

    output_bundle = args.output_bundle or default_output_bundle(dataset_root)
    output_bundle = build_bundle(selected_paths, output_bundle=output_bundle)
    print(f"Created bundle: {output_bundle}")
    print(f"Verified train site-level summary CSV: {train_summary_csv}")
    print(f"Verified test site-level summary CSV: {test_summary_csv}")
    if train_csv is not None:
        print(f"Included train CSV: {train_csv}{format_multi_metal_note(train_multi_metal_rows)}")
    if test_csv is not None:
        print(f"Included test CSV: {test_csv}{format_multi_metal_note(test_multi_metal_rows)}")


if __name__ == "__main__":
    main()
