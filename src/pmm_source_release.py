"""Pinned PinMyMetal training-source release identity (dependency-free).

Shared by the training-only cohort audit and the PinMyMetal comparator, which runs
in its own environment pinned to the released recipe's library versions.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PMM_SOURCE_DEFAULT = REPO_ROOT / "prepare_training_and_test_set" / "pinmymetal_files" / "classmodel_train_set"
PMM_SOURCE_SHA256 = "4748babd2b6ac0706cd9ed4bcfd4855c8d2c5535f01813fb2e10b68b31a24c0f"
PMM_SOURCE_RELEASE = {
    "zenodo_record": "14830978",
    "github_repository": "hhz-lab/PinMyMetal",
    "github_commit": "59ef46795920322c798db4e5ec500b04451f7904",
    "training_script": "data_model/train_chedhclassmodel.py",
    "environment_file": "metal_prediction/environment.yml",
    "source_file": "classmodel_train_set",
}
# Versions pinned in the released environment.yml (imbalanced-learn is unpinned there).
PMM_RELEASED_ENVIRONMENT = {
    "python": "3.11.5",
    "scikit-learn": "1.3.0",
    "numpy": "1.23.5",
    "pandas": "2.1.4",
    "joblib": "1.2.0",
    "imbalanced-learn": "unpinned",
}
# Columns the released script drops before fitting.
PMM_NON_FEATURE_COLUMNS = ("metalid", "pdbid", "residueid_ion", "label_metal", "source", "ched_count")


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
