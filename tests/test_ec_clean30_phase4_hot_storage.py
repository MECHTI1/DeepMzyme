"""CLEAN termination must restore hot bytes before the ordinary persistence state advances."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ec_clean30 import hot_storage, phase4


@pytest.fixture
def execution(tmp_path, monkeypatch):
    root = tmp_path / "canonical"
    request = {"provider": "runpod", "execution_root": str(root), "unit_id": "ecc30_clean_fixture",
               "session_id": "session", "g3_sha256": "b" * 64, "hot_work_root": str(tmp_path / "hot"),
               "forecast_seconds": 10, "unit_timeout_seconds": 10, "python": sys.executable,
               "cap_prefix": ["cap"], "distance_cap_prefix": ["map-cap"],
               "clean_cap_profile": {"memory_bytes": 100, "distance_map_memory_bytes": 200}}
    source = {"execution_manifest_sha256": "a" * 64}
    calls = []

    class Execution:
        state = {}
        failure = False

        def __enter__(self):
            calls.append("enter")
            return self

        def __exit__(self, *args):
            return False

        def admit(self, name, seconds):
            calls.append("admit")

        def run_subprocess(self, argv, **kwargs):
            marker = phase4._attempt(request) / hot_storage.MARKER
            hot = hot_storage.worker_hot_attempt(marker, **phase4._hot_identity(request, source))
            (hot / "checkpoint.pt").write_bytes(b"partial or completed checkpoint")
            calls.append("child_terminated")
            if self.failure:
                raise subprocess.TimeoutExpired(argv, 10)
            export = hot / "validation.npz"
            export.write_bytes(b"synthetic export")
            phase4._write(hot / "unit_result.json", {
                "status": "passed", "kind": "clean", "unit_id": request["unit_id"],
                "export": str(phase4._attempt(request) / "validation.npz"), "export_sha256": phase4._sha(export)})
            return SimpleNamespace(returncode=0)

        def record_result(self, *args):
            assert json.loads((phase4._attempt(request) / hot_storage.MARKER).read_text())["status"] == "restored"
            calls.append("record")

        def persist(self, paths):
            assert phase4._attempt(request) / "checkpoint.pt" in paths
            assert calls.index("child_terminated") < calls.index("record")
            calls.append("persist")
            return {"status": "awaiting_host_ack"}

    engine = Execution()
    monkeypatch.setattr(phase4, "_runtime", lambda *a, **kw: (source, None, None, {}, {}))
    monkeypatch.setattr(phase4, "_unit", lambda *a: {})
    monkeypatch.setattr(phase4, "_execution", lambda *a: engine)
    monkeypatch.setattr(phase4, "_check_clean_hot_storage", lambda *a: None)
    return request, source, engine, calls


@pytest.mark.parametrize("failed", [False, True])
def test_terminated_hot_attempt_is_restored_before_result_and_persistence(execution, failed):
    request, source, engine, calls = execution
    engine.failure = failed
    if failed:
        with pytest.raises(phase4.Phase4Error, match="TimeoutExpired"):
            phase4.run_unit(request, kind="clean")
    else:
        assert phase4.run_unit(request, kind="clean")["status"] == "awaiting_host_ack"
    assert calls == ["enter", "admit", "child_terminated", "record", "persist"]
    marker = phase4._attempt(request) / hot_storage.MARKER
    assert json.loads(marker.read_text())["status"] == "restored"
    assert (phase4._attempt(request) / "checkpoint.pt").read_bytes() == b"partial or completed checkpoint"
    assert Path(json.loads(marker.read_text())["identity"]["hot_attempt"]).is_dir()


def test_pending_hot_bytes_block_automatic_execution_recovery(execution):
    request, source, _, calls = execution
    hot_storage.prepare_hot_attempt(request["hot_work_root"], phase4._attempt(request),
                                    **phase4._hot_identity(request, source))
    with pytest.raises(hot_storage.HotStorageError, match="Pending hot data"):
        phase4.run_unit(request, kind="clean")
    assert calls == []


def test_pending_hot_guard_resolves_a_legacy_execution_alias(execution, tmp_path):
    request, source, _, calls = execution
    hot_storage.prepare_hot_attempt(request["hot_work_root"], phase4._attempt(request),
                                    **phase4._hot_identity(request, source))
    alias = tmp_path / "execution_alias"
    alias.symlink_to(request["execution_root"], target_is_directory=True)
    request["execution_root"] = str(alias)
    with pytest.raises(hot_storage.HotStorageError, match="Pending hot data"):
        phase4.run_unit(request, kind="clean")
    assert calls == []


def test_hot_cleanup_occurs_only_after_ack_and_result_verification(execution, monkeypatch):
    request, _, engine, calls = execution
    phase4.run_unit(request, kind="clean")
    attempt = phase4._attempt(request)
    marker = json.loads((attempt / hot_storage.MARKER).read_text())
    hot = Path(marker["identity"]["hot_attempt"])
    phase4._write(attempt.parent / "complete.json", json.loads((attempt / "unit_result.json").read_text()))
    monkeypatch.setattr(phase4, "_verify_result", lambda *a: calls.append("verified"))
    engine.state = {"pending_unit": {"run_name": request["unit_id"], "persisted": False}}
    phase4.run_unit(request, kind="clean")
    assert hot.exists()
    engine.state["pending_unit"]["persisted"] = True
    phase4.run_unit(request, kind="clean")
    assert calls[-1] == "verified" and not hot.exists()
    assert (attempt / "checkpoint.pt").is_file()


@pytest.mark.parametrize("problem", ["authority", "overlap", "duplicate_capacity"])
def test_hot_placement_and_duplicate_capacity_are_required(tmp_path, monkeypatch, problem):
    hot = tmp_path / "hot"
    request = {"hot_work_root": str(hot), "working_bytes": 1000, "execution_root": str(tmp_path / "outputs"),
               "python": sys.executable, "clean_python": sys.executable,
               **{key: str(tmp_path / key) for key in ("clean_clone", "clean_inputs", "base_farm", "masked_farm")}}
    authority = {"hot_work_root": str(hot)}
    view = SimpleNamespace(root=tmp_path / "view")
    monkeypatch.setattr(phase4, "_guard", lambda *a: None)
    monkeypatch.setattr(phase4, "check_hot_storage", lambda *a, **kw: None)
    if problem == "authority":
        authority.clear()
    elif problem == "overlap":
        request["base_farm"] = str(hot / "input")
    else:
        monkeypatch.setattr(phase4.shutil, "disk_usage", lambda path: SimpleNamespace(free=phase4.RESERVE_BYTES + 1500))
    with pytest.raises(phase4.Phase4Error):
        phase4._check_clean_hot_storage(request, view, authority)
