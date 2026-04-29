"""Build a configurable Colab bundle containing structures and CSV metadata."""

from __future__ import annotations

import argparse
import subprocess
import shutil
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

BASE_DATA_DIR = PROJECT_ROOT / ".data" / "train_and_test_sets_structures_exact_pinmymetal"
TRAIN_DIR = BASE_DATA_DIR / "train"
TEST_DIR = BASE_DATA_DIR / "test"
OUTPUT_DIR = PROJECT_ROOT / ".data" / "Colab_Bundles"
OUTPUT_BUNDLE = OUTPUT_DIR / "pinmymetal_colab_bundle_structures.tar.zst"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a compressed Colab bundle for train/test structures and CSVs.")
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--test-dir", type=Path, default=TEST_DIR)
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument("--output-bundle", type=Path, default=OUTPUT_BUNDLE)
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


def build_bundle(args: argparse.Namespace) -> Path:
    selected_paths = [args.train_dir, args.test_dir]
    if args.train_csv is not None:
        selected_paths.append(args.train_csv)
    if args.test_csv is not None:
        selected_paths.append(args.test_csv)
    validate_inputs(selected_paths)

    output_bundle = args.output_bundle
    output_bundle.parent.mkdir(parents=True, exist_ok=True)
    relative_paths = [str(path.resolve().relative_to(PROJECT_ROOT)) for path in selected_paths]
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
    output_bundle = build_bundle(args)
    print(f"Created bundle: {output_bundle}")


if __name__ == "__main__":
    main()
