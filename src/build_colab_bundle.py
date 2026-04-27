"""
Create a Colab bundle that includes:
- train set structures + CSV files
- test set structures + CSV files

The bundle is compressed as Zstandard:
    pinmymetal_colab_bundle_structures.tar.zst
"""

from pathlib import Path
import subprocess
import shutil


# -----------------------------
# Project paths
# -----------------------------

# This file is in:
#   DeepMzyme/src/build_colab_bundle.py
#
# Therefore:
#   SCRIPT_DIR   = DeepMzyme/src
#   PROJECT_ROOT = DeepMzyme

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

BASE_DATA_DIR = PROJECT_ROOT / ".data" / "train_and_test_sets_structures"

TRAIN_DIR = BASE_DATA_DIR / "train"
TEST_DIR = BASE_DATA_DIR / "test"

OUTPUT_DIR = PROJECT_ROOT / ".data" / "Colab_Bundles"
OUTPUT_BUNDLE = OUTPUT_DIR / "pinmymetal_colab_bundle_structures.tar.zst"


# -----------------------------
# Safety checks
# -----------------------------

if not BASE_DATA_DIR.exists():
    raise FileNotFoundError(
        f"Base data directory not found:\n{BASE_DATA_DIR}\n\n"
        f"Expected folder structure:\n"
        f"{PROJECT_ROOT}/.data/train_and_test_sets_structures/\n"
        f"├── train/\n"
        f"└── test/"
    )

if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Train directory not found:\n{TRAIN_DIR}"
    )

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Test directory not found:\n{TEST_DIR}"
    )

if shutil.which("zstd") is None:
    raise RuntimeError(
        "zstd is not installed or not found in PATH.\n"
        "Install it on Ubuntu with:\n"
        "sudo apt update && sudo apt install zstd"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Create tar.zst bundle
# -----------------------------
# Archive layout will be:
#   train/...
#   test/...

cmd = [
    "tar",
    "--use-compress-program=zstd -T0 -19",
    "-cf",
    str(OUTPUT_BUNDLE),
    "-C",
    str(BASE_DATA_DIR),
    "train",
    "test",
]

print("Project root:")
print(PROJECT_ROOT)

print("\nBase data directory:")
print(BASE_DATA_DIR)

print("\nRunning:")
print(" ".join(cmd))

subprocess.run(cmd, check=True)

print(f"\nCreated bundle:\n{OUTPUT_BUNDLE}")