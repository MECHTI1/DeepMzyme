#!/usr/bin/env bash
# ==============================================================================
# Master End-to-End Reproducibility Script for Exact Zenodo PinMyMetal Benchmark
# ==============================================================================
# This script enables complete 1-command deterministic replication from scratch:
# 1. Checks/downloads the reconstructed 99.89% Zenodo dataset from Hugging Face.
# 2. Validates the SHA-256 cryptographic checksum.
# 3. Extracts the dataset archive into the target data directory.
# 4. Runs the automated pre-flight verification suite (ion-level contract & PDBs).
# 5. Executes the 5-fold cross-validation campaign and held-out test evaluation.
# 6. Outputs the final markdown comparison table against PinMyMetal and Metal3D.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default paths
# Probe known data roots in order. The external-disk candidates are only used when
# the disk is actually mounted -- an unmounted mount point is an empty directory on the
# root filesystem, and writing ~1.1 GB of dataset into it would silently fill /.
DEFAULT_DATA_ROOT=""
DEFAULT_RUNS_DIR=""
if [ -d "/content/DeepMzyme_Data/DeepMzyme_Data" ]; then
    DEFAULT_DATA_ROOT="/content/DeepMzyme_Data/DeepMzyme_Data"
    DEFAULT_RUNS_DIR="/content/runs/runs_zenodo_pmm_exact"
else
    for candidate in /media/mechti/Data1 /media/mechti/Data; do
        if [ -d "${candidate}/DeepMzyme_Data" ] && mountpoint -q "${candidate}" 2>/dev/null; then
            DEFAULT_DATA_ROOT="${candidate}/DeepMzyme_Data"
            DEFAULT_RUNS_DIR="${candidate}/DeepMzyme_Data/runs_zenodo_pmm_exact"
            break
        fi
    done
fi
if [ -z "${DEFAULT_DATA_ROOT}" ]; then
    DEFAULT_DATA_ROOT="${REPO_ROOT}/DeepMzyme_Data"
    DEFAULT_RUNS_DIR="${REPO_ROOT}/runs/runs_zenodo_pmm_exact"
fi

DATA_ROOT="${DATA_ROOT:-$DEFAULT_DATA_ROOT}"
RUNS_DIR="${RUNS_DIR:-$DEFAULT_RUNS_DIR}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
FOLDS="${FOLDS:-0 1 2 3 4}"
MODELS="${MODELS:-benchmark_enhanced_only_gvp benchmark_only_esm benchmark_enhanced_gvp_esmc}"

HF_DATASET_URL="https://huggingface.co/datasets/GMBioinformatics/DeepMzyme/resolve/main/train_and_test_sets_structures_zenodo_pmm_exact.tar.gz"
EXPECTED_SHA256="24903f120462aacb90b43c4af97f4f08d61f1847d12636606fcc66e6351db296"

echo "============================================================================"
echo "DEEPMZYME: ZENODO PINMYMETAL REPRODUCIBILITY PIPELINE"
echo "Repository Root: ${REPO_ROOT}"
echo "Data Directory:  ${DATA_ROOT}"
echo "Runs Directory:  ${RUNS_DIR}"
echo "Compute Device:  ${DEVICE}"
echo "Models:          ${MODELS}"
echo "Folds:           ${FOLDS}"
echo "Epochs:          ${EPOCHS}"
echo "============================================================================"

mkdir -p "${DATA_ROOT}" "${RUNS_DIR}"

TARGET_DATASET_DIR="${DATA_ROOT}/train_and_test_sets_structures_zenodo_pmm_exact"
ARCHIVE_PATH="${DATA_ROOT}/train_and_test_sets_structures_zenodo_pmm_exact.tar.gz"

# Step 1: Ensure Dataset is Downloaded and Extracted
if [ ! -d "${TARGET_DATASET_DIR}/train/structures" ] || [ ! -f "${TARGET_DATASET_DIR}/train/site_manifest.csv" ]; then
    echo "[STEP 1/4] Dataset not found in ${TARGET_DATASET_DIR}. Fetching from Hugging Face..."
    if [ ! -f "${ARCHIVE_PATH}" ]; then
        echo "[DOWNLOAD] Downloading archive from ${HF_DATASET_URL}..."
        curl -L -C - --retry 5 -o "${ARCHIVE_PATH}" "${HF_DATASET_URL}"
    fi

    echo "[CHECKSUM] Verifying SHA-256 checksum..."
    ACTUAL_SHA256=$(sha256sum "${ARCHIVE_PATH}" | awk '{print $1}')
    if [ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]; then
        echo "[ERROR] Checksum mismatch!"
        echo "  Expected: ${EXPECTED_SHA256}"
        echo "  Got:      ${ACTUAL_SHA256}"
        exit 1
    fi
    echo "[CHECKSUM] Checksum verified: ${ACTUAL_SHA256}"

    echo "[EXTRACT] Extracting archive..."
    tar -xzf "${ARCHIVE_PATH}" -C "${DATA_ROOT}"
    echo "[EXTRACT] Extraction complete."
else
    echo "[STEP 1/4] Verified dataset exists at ${TARGET_DATASET_DIR}."
fi

# Step 2: Run Automated Verification Suite
echo "============================================================================"
echo "[STEP 2/4] Running automated pre-flight dataset verification..."
python3 "${SCRIPT_DIR}/verify_zenodo_pmm_ion_dataset.py" --data-dir "${TARGET_DATASET_DIR}"
echo "[STEP 2/4] Verification suite PASSED with 0 errors."

# Step 3: Run Benchmark Cross-Validation Campaign
echo "============================================================================"
echo "[STEP 3/4] Launching benchmark training runner..."
# shellcheck disable=SC2086
python3 -u "${SCRIPT_DIR}/run_zenodo_pmm_exact_5fold_cv.py" \
    --models ${MODELS} \
    --folds ${FOLDS} \
    --n-folds 5 \
    --data-root "${DATA_ROOT}" \
    --runs-dir "${RUNS_DIR}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}"

# Step 4: Display Results Table
echo "============================================================================"
echo "[STEP 4/4] Pipeline Complete! Final Benchmark Table:"
TABLE_PATH="${RUNS_DIR}/zenodo_pmm_exact_5fold_comparison_table.md"
if [ -f "${TABLE_PATH}" ]; then
    cat "${TABLE_PATH}"
else
    echo "Results saved in ${RUNS_DIR}."
fi
echo "============================================================================"
