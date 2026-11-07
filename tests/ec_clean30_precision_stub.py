"""Synthetic stand-in for ``ec_clean30.runtime_precision`` (written by the execution lane) in science tests.

The stub applies the same acceptance rule as ``runtime_precision.validate_policy`` (device, TF32 off, device count,
helper hash), on a fixed synthetic helper hash. ``SUBPROCESS_PRELUDE`` installs it in a child interpreter.
"""
from __future__ import annotations

import json
import sys
import types
from typing import Any

POLICY_PREFIX = "DEEPMZYME_RUNTIME_PRECISION "
STUB_HELPER_SHA256 = "5" * 64


def stub_policy(device: str = "cuda", **changes: Any) -> dict[str, Any]:
    record = {"device": device, "nvidia_tf32_override": "0", "cuda_matmul_allow_tf32": False,
              "cudnn_allow_tf32": False, "torch_version": "2.x", "cuda_device_count": 1 if device == "cuda" else 0,
              "cuda_device_name": "synthetic GPU" if device == "cuda" else None, "helper_sha256": STUB_HELPER_SHA256}
    record.update(changes)
    return record


def validate_policy(record: Any, *, device: str) -> dict[str, Any]:
    valid = (isinstance(record, dict) and record.get("device") == device
             and record.get("nvidia_tf32_override") == "0" and record.get("cuda_matmul_allow_tf32") is False
             and record.get("cudnn_allow_tf32") is False and record.get("helper_sha256") == STUB_HELPER_SHA256
             and record.get("cuda_device_count") == (1 if device == "cuda" else 0))
    if not valid:
        raise RuntimeError("Child precision policy does not prove the requested device and TF32-off state.")
    return dict(record)


def read_policy_log(path: Any, *, device: str) -> dict[str, Any]:
    from pathlib import Path

    records = [line[len(POLICY_PREFIX):] for line in Path(path).read_text().splitlines()
               if line.startswith(POLICY_PREFIX)]
    if len(records) != 1:
        raise RuntimeError("Child log must contain exactly one precision-policy record.")
    return validate_policy(json.loads(records[0]), device=device)


def execution_environment(base: Any, device: str) -> dict[str, str]:
    if device not in {"cpu", "cuda"}:
        raise RuntimeError("Execution device must be cpu or cuda.")
    env = dict(base)
    env["NVIDIA_TF32_OVERRIDE"] = "0"
    return env


def bootstrap_code(device: str) -> str:
    """A child prelude that records the synthetic policy (the comparator builds its child prelude at import)."""
    if device not in {"cpu", "cuda"}:
        raise RuntimeError("Execution device must be cpu or cuda.")
    return f"print({POLICY_PREFIX!r} + {json.dumps(json.dumps(stub_policy(device), sort_keys=True))}, flush=True)\n"


def module() -> types.ModuleType:
    stub = types.ModuleType("ec_clean30.runtime_precision")
    stub.POLICY_PREFIX = POLICY_PREFIX
    stub.validate_policy = validate_policy
    stub.read_policy_log = read_policy_log
    stub.execution_environment = execution_environment
    stub.bootstrap_code = bootstrap_code
    return stub


def install(monkeypatch) -> types.ModuleType:
    stub = module()
    monkeypatch.setitem(sys.modules, "ec_clean30.runtime_precision", stub)
    return stub


def policy_log_text(device: str = "cuda", **changes: Any) -> str:
    return "fit started\n" + POLICY_PREFIX + json.dumps(stub_policy(device, **changes), sort_keys=True) + "\nfit done\n"


def fit_policy_entry(device: str = "cuda", **changes: Any) -> dict[str, Any]:
    """A fit's precision-policy entry as ``clean_sanity.read_fit_policy`` returns it (synthetic log hash)."""
    import hashlib

    return {"device": device, "log_sha256": hashlib.sha256(policy_log_text(device, **changes).encode()).hexdigest(),
            "record": stub_policy(device, **changes)}


SUBPROCESS_PRELUDE = (
    "import sys, runpy; sys.path.insert(0, {tests!r}); import ec_clean30_precision_stub as s; "
    "sys.modules['ec_clean30.runtime_precision'] = s.module(); "
)
