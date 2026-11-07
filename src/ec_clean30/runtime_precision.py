"""Explicit child precision policy and a wrapper for the unchanged private ESM-1b farm."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

POLICY_PREFIX = "DEEPMZYME_RUNTIME_PRECISION "
PRIVATE_FARM_SHA256 = "73c25c5396f89a603a2ca97cf2adf82bb6b220bccf4d9f45801d1bdcf60bede4"


def execution_environment(base: Mapping[str, str], device: str) -> dict[str, str]:
    if device not in {"cpu", "cuda"}:
        raise RuntimeError("Execution device must be cpu or cuda.")
    env = dict(base)
    if device == "cuda" and "CUDA_VISIBLE_DEVICES" in env and not env["CUDA_VISIBLE_DEVICES"].strip():
        raise RuntimeError("CUDA execution cannot override explicitly disabled GPU visibility.")
    env["NVIDIA_TF32_OVERRIDE"] = "0"
    return env


def configure_and_record(device: str) -> dict[str, Any]:
    """Run inside the guarded child before loading its model or producer."""
    env = execution_environment(os.environ, device)
    os.environ["NVIDIA_TF32_OVERRIDE"] = env["NVIDIA_TF32_OVERRIDE"]
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    count = torch.cuda.device_count() if device == "cuda" else 0
    if device == "cuda" and (not torch.cuda.is_available() or count != 1):
        raise RuntimeError("CUDA execution requires exactly one available GPU; CPU fallback is forbidden.")
    record = {"device": device, "nvidia_tf32_override": os.environ["NVIDIA_TF32_OVERRIDE"],
              "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
              "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
              "torch_version": str(torch.__version__), "cuda_device_count": count,
              "cuda_device_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
              "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    validate_policy(record, device=device)
    print(POLICY_PREFIX + json.dumps(record, sort_keys=True), flush=True)
    return record


def bootstrap_code(device: str) -> str:
    """Load this authenticated standard-library helper after the read guard is installed."""
    if device not in {"cpu", "cuda"}:
        raise RuntimeError("Execution device must be cpu or cuda.")
    return ("import importlib.util\n"
            f"precision_spec = importlib.util.spec_from_file_location('deepmzyme_runtime_precision', {str(Path(__file__).resolve())!r})\n"
            "precision_module = importlib.util.module_from_spec(precision_spec)\n"
            "precision_spec.loader.exec_module(precision_module)\n"
            f"precision_module.configure_and_record({device!r})\n")


def validate_policy(record: Mapping[str, Any], *, device: str) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise RuntimeError("Child precision-policy record must be an object.")
    valid = (record.get("device") == device
             and record.get("nvidia_tf32_override") == "0"
             and record.get("cuda_matmul_allow_tf32") is False and record.get("cudnn_allow_tf32") is False
             and isinstance(record.get("torch_version"), str) and bool(record["torch_version"])
             and record.get("helper_sha256") == hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if device == "cuda":
        valid = valid and type(record.get("cuda_device_count")) is int and record["cuda_device_count"] == 1
        valid = valid and isinstance(record.get("cuda_device_name"), str) and bool(record["cuda_device_name"])
    elif device == "cpu":
        valid = valid and type(record.get("cuda_device_count")) is int and record["cuda_device_count"] == 0
        valid = valid and record.get("cuda_device_name") is None
    else:
        valid = False
    if not valid:
        raise RuntimeError("Child precision policy does not prove the requested device and TF32-off state.")
    return dict(record)


def read_policy_log(path: Path, *, device: str) -> dict[str, Any]:
    records = [line[len(POLICY_PREFIX):] for line in Path(path).read_text().splitlines()
               if line.startswith(POLICY_PREFIX)]
    if len(records) != 1:
        raise RuntimeError("Child log must contain exactly one precision-policy record.")
    try:
        record = json.loads(records[0])
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Child precision-policy record is invalid JSON.") from exc
    return validate_policy(record, device=device)


def run_private_farm(harness: Path, arguments: Sequence[str]) -> int:
    """Change only child launch policy; preserve the pinned private producer bytes and recipe."""
    harness = Path(harness)
    if not harness.is_file() or harness.is_symlink() or hashlib.sha256(harness.read_bytes()).hexdigest() != PRIVATE_FARM_SHA256:
        raise RuntimeError("Private ESM-1b farm differs from its frozen source pin.")
    args = list(arguments)
    if args.count("--device") != 1 or args.index("--device") + 1 >= len(args):
        raise RuntimeError("The farm wrapper requires one explicit device.")
    device = args[args.index("--device") + 1]
    execution_environment(os.environ, device)
    spec = importlib.util.spec_from_file_location("deepmzyme_pinned_esm1b_farm", harness)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    marker = "import torch.hub"
    if module.EXTRACT_LAUNCH.count(marker) != 1:
        raise RuntimeError("Private farm child launch no longer matches the pinned adapter seam.")
    module.EXTRACT_LAUNCH = module.EXTRACT_LAUNCH.replace(marker, bootstrap_code(device) + marker)
    original_environment = module.extraction_env

    def extraction_environment(runner, roots, actual_device, threads):
        if actual_device != device:
            raise RuntimeError("Private farm device changed after launch preparation.")
        return execution_environment(original_environment(runner, roots, actual_device, threads), actual_device)

    module.extraction_env = extraction_environment
    return module.main(args)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("esm1b-farm",))
    parser.add_argument("--harness", type=Path, required=True)
    args, remaining = parser.parse_known_args(argv)
    if remaining[:1] == ["--"]:
        remaining = remaining[1:]
    return run_private_farm(args.harness, remaining)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"Runtime precision refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
