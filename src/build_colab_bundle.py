"""Build a configurable Colab bundle containing structures and CSV metadata."""

from __future__ import annotations

import argparse
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

BASE_DATA_DIR = PROJECT_ROOT / ".data" / "train_and_test_sets_structures_exact_pinmymetal"
TRAIN_DIR = BASE_DATA_DIR / "train"
TEST_DIR = BASE_DATA_DIR / "test"
OUTPUT_DIR = PROJECT_ROOT / ".data" / "Colab_Bundles"
SUMMARY_CSV_BASENAME = CATALYTIC_ONLY_SUMMARY_CSV.name

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
    parser.add_argument("--allow-multi-metal-structures", action="store_true")
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


def _default_summary_csv_for_structure_dir(structure_dir: Path) -> Path:
    local_summary_csv = structure_dir / SUMMARY_CSV_BASENAME
    if local_summary_csv.exists():
        return local_summary_csv
    return CATALYTIC_ONLY_SUMMARY_CSV


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
        train_summary_csv = _default_summary_csv_for_structure_dir(train_dir)
    if test_summary_csv is None and args.test_csv is None:
        test_summary_csv = _default_summary_csv_for_structure_dir(test_dir)
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


def generate_structure_csv(
    *,
    structure_dir: Path,
    summary_csv: Path,
    output_csv: Path,
    allow_multi_metal_structures: bool,
    ec_label_depth: int,
) -> Path:
    rows = build_structure_rows(
        structure_dir=structure_dir,
        summary_csv=summary_csv,
        allow_multi_metal_structures=allow_multi_metal_structures,
        ec_label_depth=ec_label_depth,
    )
    validate_rows(rows)
    validate_rows_match_structure_dir(structure_dir=structure_dir, rows=rows)
    write_rows(output_csv, rows)
    return output_csv


def prepare_csv_artifacts(
    args: argparse.Namespace,
    *,
    dataset_root: Path,
    train_dir: Path,
    test_dir: Path,
    train_summary_csv: Path | None,
    test_summary_csv: Path | None,
) -> tuple[Path | None, Path | None]:
    csv_output_dir = args.csv_output_dir or default_csv_output_dir(dataset_root)
    csv_output_dir.mkdir(parents=True, exist_ok=True)

    train_csv = args.train_csv
    if train_csv is None:
        if train_summary_csv is None:
            raise ValueError("A train summary CSV is required to generate the train CSV artifact.")
        train_csv = csv_output_dir / f"{dataset_root.name}_train.csv"
        generate_structure_csv(
            structure_dir=train_dir,
            summary_csv=train_summary_csv,
            output_csv=train_csv,
            allow_multi_metal_structures=args.allow_multi_metal_structures,
            ec_label_depth=args.ec_label_depth,
        )

    test_csv = args.test_csv
    if test_csv is None:
        if test_summary_csv is None:
            raise ValueError("A test summary CSV is required to generate the test CSV artifact.")
        test_csv = csv_output_dir / f"{dataset_root.name}_test.csv"
        generate_structure_csv(
            structure_dir=test_dir,
            summary_csv=test_summary_csv,
            output_csv=test_csv,
            allow_multi_metal_structures=args.allow_multi_metal_structures,
            ec_label_depth=args.ec_label_depth,
        )
    return train_csv, test_csv


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

    train_csv, test_csv = prepare_csv_artifacts(
        args,
        dataset_root=dataset_root,
        train_dir=train_dir,
        test_dir=test_dir,
        train_summary_csv=train_summary_csv,
        test_summary_csv=test_summary_csv,
    )

    selected_paths = [train_dir, test_dir]
    if train_csv is not None:
        selected_paths.append(train_csv)
    if test_csv is not None:
        selected_paths.append(test_csv)

    if args.skip_bundle:
        print(f"Prepared train directory: {train_dir}")
        print(f"Prepared test directory: {test_dir}")
        if train_csv is not None:
            print(f"Prepared train CSV: {train_csv}")
        if test_csv is not None:
            print(f"Prepared test CSV: {test_csv}")
        return

    output_bundle = args.output_bundle or default_output_bundle(dataset_root)
    output_bundle = build_bundle(selected_paths, output_bundle=output_bundle)
    print(f"Created bundle: {output_bundle}")
    if train_csv is not None:
        print(f"Included train CSV: {train_csv}")
    if test_csv is not None:
        print(f"Included test CSV: {test_csv}")


if __name__ == "__main__":
    main()
