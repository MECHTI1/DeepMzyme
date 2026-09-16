"""Per-fit profiling without allocating a CUDA device or running training."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import train_serial_metal_profile as wrapper


def fake_torch():
    reset = []
    cuda = SimpleNamespace(is_available=lambda: True, reset_peak_memory_stats=reset.append,
                           max_memory_allocated=lambda device: 123, max_memory_reserved=lambda device: 456)
    return SimpleNamespace(cuda=cuda), reset


def test_profiles_one_call_without_changing_arguments_or_return_value(tmp_path):
    torch, reset = fake_torch()
    argv = ["--runs-dir", str(tmp_path), "--run-name=fit", "--device", "cuda:0", "--unknown-training-option", "value"]
    calls = []
    def train():
        calls.append(1)
        directory = tmp_path / "fit"
        directory.mkdir()
        (directory / "prepare_status.json").write_text('{"status":"ready"}')
        return "original-return"
    assert wrapper.profile_training(train, argv, torch_module=torch) == "original-return"
    row = json.loads((tmp_path / "fit/performance_profile.json").read_text())
    assert calls == [1] and reset == ["cuda:0"]
    assert row["status"] == "completed" and row["cuda_peak_allocated_bytes"] == 123
    assert row["cuda_peak_reserved_bytes"] == 456
    assert row["setup_seconds"] >= 0 and row["training_seconds"] >= 0
    assert row["setup_seconds"] + row["training_seconds"] == pytest.approx(row["elapsed_seconds"])


@pytest.mark.parametrize("error", [RuntimeError("synthetic failure"), SystemExit(7)])
def test_partial_profile_preserves_original_exception_and_exit(tmp_path, error):
    torch, _ = fake_torch()
    def train():
        raise error
    with pytest.raises(type(error)) as caught:
        wrapper.profile_training(train, ["--runs-dir", str(tmp_path), "--run-name", "failed", "--device", "cuda"], torch_module=torch)
    assert caught.value is error
    row = json.loads((tmp_path / "failed/performance_profile.json").read_text())
    assert row["status"] == "failed" and row["exception_type"] == type(error).__name__
    assert row["prepare_seconds"] is None and row["cuda_peak_reserved_bytes"] == 456
    if isinstance(error, SystemExit):
        assert row["exit_code"] == 7


def test_cpu_run_does_not_touch_cuda_counters(tmp_path):
    torch, reset = fake_torch()
    wrapper.profile_training(lambda: None, ["--runs-dir", str(tmp_path), "--run-name", "cpu", "--device=cpu"], torch_module=torch)
    assert reset == []
    row = json.loads((tmp_path / "cpu/performance_profile.json").read_text())
    assert row["cuda_peak_allocated_bytes"] is None


def test_training_module_import_occurs_inside_profiled_callable(monkeypatch):
    events = []
    class TrainingModule:
        @property
        def main(self):
            events.append("train_import")
            return lambda: events.append("training")
    monkeypatch.setitem(sys.modules, "train", TrainingModule())
    def profile(entrypoint):
        events.append("profile_started")
        return entrypoint()
    monkeypatch.setattr(wrapper, "profile_training", profile)
    wrapper.main()
    assert events == ["profile_started", "train_import", "training"]


@pytest.mark.parametrize("complete_profile", [True, False])
def test_outside_wrapper_overhead_is_fixed_setup_when_training_timing_is_known(tmp_path, complete_profile):
    from serial_metal_campaign import profile, workflow
    run = dict(id="smoke", arm="gvp_four", epochs=1, parameters={})
    directory = tmp_path / "runs/smoke"
    measured = dict(setup_seconds=80., total_seconds=100., elapsed_seconds=100.)
    if complete_profile:
        measured["training_seconds"] = 20.
    profile.base.save(directory / "performance_profile.json", measured)
    profile.base.save(directory / "prepare_status.json", {"status": "ready"})
    profile.base.save(tmp_path / "sessions.json", [{"session_id": "one", "hardware": {"gpu": "fixture"}, "stopped_epoch": None}])
    profile.base.save(tmp_path / "queue.json", {"runs": [run]})
    profile.base.save(tmp_path / "attempts.json", [{"status": "completed", "run_id": "smoke", "session_id": "one",
        "run_dir": str(directory), "started_epoch": 1., "elapsed_seconds": 130.}])
    row = workflow._timings(tmp_path)[0]
    assert row["setup_seconds"] == (110. if complete_profile else 80.)
    assert row["epoch_seconds"] == (20. if complete_profile else 50.)
