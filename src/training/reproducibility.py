from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch

RUNTIME_METADATA_SCHEMA_VERSION = 1
PACKAGE_DISTRIBUTIONS = {
    "torch": "torch",
    "torch_geometric": "torch-geometric",
    "biopython": "biopython",
    "biotite": "biotite",
    "esm": "esm",
    "gemmi": "gemmi",
    "numpy": "numpy",
    "optuna": "optuna",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "propka": "propka",
    "pytest": "pytest",
    "scikit_learn": "scikit-learn",
}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def installed_package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for output_name, distribution_name in PACKAGE_DISTRIBUTIONS.items():
        try:
            versions[output_name] = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            versions[output_name] = None
    return versions


def _nvidia_driver_version() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    versions = sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})
    return ",".join(versions) or None


def runtime_environment_payload(argv: Sequence[str] | None = None) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    gpu_devices: list[dict[str, Any]] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpu_devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": int(properties.total_memory),
                    "compute_capability": f"{properties.major}.{properties.minor}",
                }
            )
    invocation = [sys.executable, *(list(sys.argv) if argv is None else list(argv))]
    return {
        "schema_version": RUNTIME_METADATA_SCHEMA_VERSION,
        "captured_at_utc": utc_timestamp(),
        "invocation": invocation,
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "packages": installed_package_versions(),
        "accelerator": {
            "cuda_available": cuda_available,
            "torch_cuda_version": torch.version.cuda,
            "cuda_arch_list": torch.cuda.get_arch_list() if cuda_available else [],
            "nvidia_driver_version": _nvidia_driver_version() if cuda_available else None,
            "devices": gpu_devices,
        },
    }


def source_control_payload(repo_root: Path) -> dict[str, Any]:
    def git_output(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception:
            return None
        value = result.stdout.strip()
        return value or None

    commit = git_output("rev-parse", "HEAD")
    branch = git_output("branch", "--show-current")
    status = git_output("status", "--porcelain")
    return {
        "repository_root": str(repo_root.resolve()),
        "commit": commit,
        "branch": branch,
        "dirty": bool(status) if commit is not None else None,
        "status_entry_count": (len(status.splitlines()) if status else 0) if commit is not None else None,
    }


def _file_artifact(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        payload.update({"size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return payload


def _split_manifest_for_structure_dir(structure_dir: Path | None) -> dict[str, Any] | None:
    if structure_dir is None:
        return None
    return _file_artifact(Path(structure_dir).resolve().parent / "split_metadata.json")


def source_artifacts_payload(config: Any) -> dict[str, Any]:
    bundle_id = getattr(config, "dataset_bundle_id", None)
    bundle_sha256 = getattr(config, "dataset_bundle_sha256", None)
    return {
        "dataset_bundle": {
            "identifier": bundle_id,
            "sha256": bundle_sha256,
            "declared_by_user": bool(bundle_id or bundle_sha256),
        },
        "evaluation_protocol": {
            "identifier": getattr(config, "evaluation_protocol_id", None),
            "held_out_overlap_policy": getattr(config, "held_out_overlap_policy", None),
        },
        "training_source": {
            "structure_dir": str(config.structure_dir),
            "summary_csv": _file_artifact(Path(config.summary_csv)),
            "split_manifest": _split_manifest_for_structure_dir(Path(config.structure_dir)),
        },
        "held_out_test_source": {
            "structure_dir": str(config.test_structure_dir) if config.test_structure_dir is not None else None,
            "summary_csv": _file_artifact(Path(config.test_summary_csv)) if config.test_summary_csv is not None else None,
            "split_manifest": _split_manifest_for_structure_dir(
                Path(config.test_structure_dir) if config.test_structure_dir is not None else None
            ),
            "evaluation_requested": bool(config.run_test_eval),
        },
    }
