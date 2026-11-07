"""Fit workers request independent CPU replay and validate its recorded policy."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from ec_clean30 import phase3, phase4


@pytest.fixture(params=["timing", "s1"])
def replay_worker(request, tmp_path, monkeypatch):
    module = phase3 if request.param == "timing" else phase4
    attempt = tmp_path / "attempt"
    runs = attempt / "fit"
    run = runs / "unit"
    run.mkdir(parents=True)
    (run / "runtime_profile.json").write_text("{}")
    runner = runs / "runner.json"
    runner.write_text("{}")
    source = {"execution_manifest_sha256": "a" * 64}
    view = SimpleNamespace(forbidden_read_roots=[str(tmp_path / "held_out")], manifest_sha256="b" * 64)
    command = SimpleNamespace(runs_dir=str(runs), unit_id="unit", environment=lambda: {"UNCHANGED": "yes"})
    count = 3 if module is phase3 else 5
    monkeypatch.setattr(module, "_runtime", lambda *args, **kwargs: (source, view, *([None] * (count - 2))))
    monkeypatch.setattr(module, "_build_timing_command" if module is phase3 else "build_s1_command",
                        lambda *args: command)
    monkeypatch.setattr(module.commands, "run_unit", lambda *args, **kwargs: {"status": "completed"})
    monkeypatch.setattr(module.commands, "verify_unit", lambda *args, **kwargs: {"passed": True})
    monkeypatch.setattr(module.commands, "verify_units", lambda *args, **kwargs: {"status": "passed"})
    monkeypatch.setattr(module.commands, "runner_receipt_path", lambda *args: runner)
    if module is phase3:
        monkeypatch.setattr(module, "_prediction_table", lambda path: {"ids": ["fixture"]})
    else:
        monkeypatch.setattr(module, "_attempt", lambda value: attempt)
    spec = {"python": sys.executable, "cap_prefix": ["fixture-cap"], "memory_limit_bytes": 2 * 1024 ** 3,
            "device": "cuda", "recipe": "F2", "unit_id": "unit", "g3_sha256": "c" * 64}
    return module, spec, run, view


@pytest.mark.parametrize("fit_device,expected_atol", [("cuda", "1e-5"), ("cpu", "1e-6")])
def test_workers_replay_on_cpu_with_g1_device_tolerance(replay_worker, monkeypatch, fit_device, expected_atol):
    module, spec, run, view = replay_worker
    spec["device"] = fit_device
    observed = {}

    def replay(argv, **kwargs):
        observed.update(argv=argv, **kwargs)
        path = run / "ec_prediction_replay/ec_prediction_replay_receipt.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"device": "cpu", "logit_atol": float(expected_atol), "replay_status": "match"}))

    monkeypatch.setattr(module.subprocess, "run", replay)
    result = (module.timing_worker if module is phase3 else module.s1_worker)(spec)
    argv = observed["argv"]
    assert argv[:2] == ["fixture-cap", sys.executable]
    assert argv[argv.index("--device") + 1] == "cpu"
    assert argv[argv.index("--logit-atol") + 1] == expected_atol
    assert argv[argv.index("--forbidden-read-root") + 1] == view.forbidden_read_roots[0]
    assert observed["env"]["UNCHANGED"] == "yes" and observed["check"] is True
    assert spec["device"] == fit_device and result["status"] == "passed"


@pytest.mark.parametrize("field,value", [("device", "cuda"), ("logit_atol", 1e-4), ("replay_status", "mismatch")])
def test_workers_refuse_replay_receipt_with_wrong_policy_or_result(replay_worker, monkeypatch, field, value):
    module, spec, run, _ = replay_worker

    def replay(argv, **kwargs):
        record = {"device": "cpu", "logit_atol": 1e-5, "replay_status": "match", field: value}
        path = run / "ec_prediction_replay/ec_prediction_replay_receipt.json"
        path.parent.mkdir()
        path.write_text(json.dumps(record))

    monkeypatch.setattr(module.subprocess, "run", replay)
    with pytest.raises(phase3.Phase3Error, match="replay policy or result differs"):
        (module.timing_worker if module is phase3 else module.s1_worker)(spec)
