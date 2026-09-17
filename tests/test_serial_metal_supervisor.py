"""Same-endpoint orchestration failures using an in-memory provider adapter."""
import hashlib
import io
import os
from pathlib import Path
import shutil
import sys
import tarfile
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serial_metal_campaign import control, remote_queue, runtime, supervisor


class FakeAdapter:
    def __init__(self, output, count=3):
        self.output, self.count = output, count
        self.next_index = 0
        self.pending = []
        self.launches = []
        self.persisted = []
        self.stops = []
        self.reconnections = 0
        self.setup_calls = 0
        self.fail_launch_reply = False
        self.fail_persistence = False
        self.stop_after_persist = False

    def verify_setup(self):
        self.setup_calls += 1

    def observe(self):
        return {"active": None, "running": [], "pending_persistence": list(self.pending)}

    def persist_attempt(self, ident):
        if self.fail_persistence:
            raise ValueError("Corrupt independent persistence bytes")
        self.persisted.append(ident)
        self.pending.remove(ident)
        if self.stop_after_persist:
            control.record(self.output, "pause", request_id="user-stop", reason="Stop this campaign")
        return {"attempt_status": "completed"}

    def persist_state(self, *, prepare_next):
        assert not self.pending, "No next fit before every terminal result is durable"
        if not prepare_next or self.next_index == self.count:
            return {"next": None, "rollover_required": False}
        return {"next": {"intent_id": str(self.next_index), "attempt_id": str(self.next_index)}}

    def launch(self, info):
        if info["intent_id"] not in self.launches:
            self.launches.append(info["intent_id"])
            self.next_index += 1
        if self.fail_launch_reply:
            self.fail_launch_reply = False
            raise TimeoutError("Lost launch acknowledgement")
        return {"status": "submitted", "attempt_id": info["attempt_id"]}

    def await_attempt(self, info):
        self.pending.append(info["attempt_id"])

    def reconnect(self):
        self.reconnections += 1

    def stop(self, reason):
        self.stops.append(reason)
        return {"provider_verified_stopped": True, "session_id": "same-owned-endpoint"}


def test_three_fits_run_on_one_prepared_endpoint_with_zero_pending_acknowledgements(tmp_path):
    adapter = FakeAdapter(tmp_path)
    result = supervisor.run(tmp_path, adapter, sleep=lambda _: None)
    assert result["completed_this_segment"] == 3
    assert adapter.setup_calls == 1
    assert adapter.launches == adapter.persisted == ["0", "1", "2"]
    assert adapter.stops == ["queue_complete"]
    assert adapter.reconnections == 0


def test_lost_launch_reply_reconciles_same_intent_without_duplicate_fit(tmp_path):
    adapter = FakeAdapter(tmp_path, count=1)
    adapter.fail_launch_reply = True
    supervisor.run(tmp_path, adapter, sleep=lambda _: None)
    assert adapter.launches == adapter.persisted == ["0"]
    assert adapter.reconnections == 1
    assert len(adapter.stops) == 1


def test_corrupt_persistence_stops_queue_before_next_launch(tmp_path):
    adapter = FakeAdapter(tmp_path)
    adapter.fail_persistence = True
    with pytest.raises(ValueError, match="Corrupt"):
        supervisor.run(tmp_path, adapter, sleep=lambda _: None)
    assert adapter.launches == ["0"]
    assert adapter.stops == ["controller_fault"]


def test_new_user_pause_stops_owned_endpoint_and_prevents_next_fit(tmp_path):
    adapter = FakeAdapter(tmp_path)
    adapter.stop_after_persist = True
    result = supervisor.run(tmp_path, adapter, sleep=lambda _: None)
    assert result["reason"] == "user_paused"
    assert adapter.launches == adapter.persisted == ["0"]
    assert adapter.stops == ["user_paused"]


def test_exhausted_transport_recovery_never_allocates_replacement(tmp_path):
    adapter = FakeAdapter(tmp_path)
    def disconnected():
        raise ConnectionError("Endpoint response unavailable")
    adapter.observe = disconnected
    with pytest.raises(ConnectionError):
        supervisor.run(tmp_path, adapter, sleep=lambda _: None, transport_retries=2)
    assert adapter.reconnections == 2
    assert adapter.launches == []
    assert adapter.stops == ["controller_fault"]


def test_duplicate_host_controller_cannot_stop_the_active_owners_vm(tmp_path):
    adapter = FakeAdapter(tmp_path)
    with control.controller_lease(tmp_path):
        with pytest.raises(RuntimeError, match="controller"):
            supervisor.run(tmp_path, adapter, sleep=lambda _: None)
    assert adapter.setup_calls == 0
    assert adapter.stops == []


@pytest.mark.parametrize("failure", [None, "extra", "duplicate", "symlink", "corrupt", "existing_corrupt"])
def test_independent_readback_rejects_archive_corruption_and_is_idempotent(tmp_path, failure):
    data = b"completed checkpoint fixture"
    inventory = {"checkpoint.pt": {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}}
    archive = tmp_path / "attempt.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo("checkpoint.pt")
        member.size = len(data)
        if failure == "symlink":
            member.type = tarfile.SYMTYPE
            member.linkname = "../outside"
        stream.addfile(member, io.BytesIO(b"x" * len(data) if failure == "corrupt" else data))
        if failure in {"extra", "duplicate"}:
            member.name = "extra" if failure == "extra" else member.name
            stream.addfile(member, io.BytesIO(data))
    destination = tmp_path / "readback"
    if failure == "existing_corrupt":
        destination.mkdir()
        (destination / "checkpoint.pt").write_bytes(b"corrupt")
    if failure:
        with pytest.raises(ValueError):
            remote_queue.verify_archive(archive, inventory, destination)
    else:
        remote_queue.verify_archive(archive, inventory, destination)
        remote_queue.verify_archive(archive, inventory, destination)
        assert (destination / "checkpoint.pt").read_bytes() == data


def test_ambiguous_dispatch_blocks_second_submission(tmp_path, monkeypatch):
    info = {"intent_id": "intent", "attempt_id": "attempt", "run_id": "run"}
    runtime.atomic_json(tmp_path / "queue_transport/intent_dispatch.json", {
        "status": "dispatching", **info, **runtime._identity()})
    monkeypatch.setattr(remote_queue.workflow, "verify_host_guard", lambda *_: {})
    monkeypatch.setattr(remote_queue.workflow, "require_persistent_state", lambda *_: None)
    monkeypatch.setattr(remote_queue.runtime, "_pending_intents", lambda *_: [info])
    def forbidden(*_args, **_kwargs):
        pytest.fail("An ambiguous dispatch must never launch another subprocess")
    monkeypatch.setattr(remote_queue.subprocess, "Popen", forbidden)
    with pytest.raises(RuntimeError, match="Ambiguous prior launch"):
        remote_queue.launch(tmp_path, info, tmp_path / "guard.json")


def test_lost_reply_after_attempt_registration_reuses_existing_attempt(tmp_path, monkeypatch):
    runtime.atomic_json(tmp_path / "attempts.json", [])
    snapshot = remote_queue.workflow.export_state(tmp_path)
    readback = tmp_path / "independent_readback"
    shutil.copytree(snapshot["source_root"], readback)
    remote_queue.workflow.verify_state_transfer(tmp_path, {
        "state_sha256": snapshot["state_sha256"], "readback_root": str(readback),
        "destination_uri": "fixture://durable"})
    runtime.atomic_json(tmp_path / "attempts.json", [{"attempt_id": "a", "run_id": "r", "launch_intent_id": "i"}])
    with pytest.raises(ValueError, match="Export current campaign state"):
        remote_queue.workflow.require_persistent_state(tmp_path)
    monkeypatch.setattr(remote_queue.workflow, "verify_host_guard", lambda *_: {})
    with remote_queue.workflow.controller_lock(tmp_path):
        result = remote_queue.launch(tmp_path, {"intent_id": "i", "attempt_id": "a", "run_id": "r"}, tmp_path / "guard.json")
    assert result == {"status": "reconciled_existing_attempt", "attempt_id": "a"}
    with pytest.raises(ValueError, match="differs from the registered attempt"):
        remote_queue.launch(tmp_path, {"intent_id": "i", "attempt_id": "other", "run_id": "r"}, tmp_path / "guard.json")


def test_dispatch_releases_controller_lock_before_fast_child_start(tmp_path, monkeypatch):
    info = {"intent_id": "intent", "attempt_id": "attempt", "run_id": "run"}
    runtime.atomic_json(tmp_path / "campaign_manifest.json", {
        "root": str(tmp_path), "runs": [{"command": ["/frozen/python312", "train.py"]}]})
    monkeypatch.setattr(remote_queue.workflow, "verify_host_guard", lambda *_: {})
    monkeypatch.setattr(remote_queue.workflow, "require_persistent_state", lambda *_: None)
    monkeypatch.setattr(remote_queue.runtime, "_pending_intents", lambda *_: [info])
    starts = []
    def immediate_child(command, **kwargs):
        with remote_queue.workflow.controller_lock(tmp_path):
            starts.append(command)
        return SimpleNamespace(pid=os.getpid())
    monkeypatch.setattr(remote_queue.subprocess, "Popen", immediate_child)
    result = remote_queue.launch(tmp_path, info, tmp_path / "guard.json")
    assert result["status"] == "submitted"
    assert len(starts) == 1
    assert starts[0][0] == "/frozen/python312"


def test_worker_interpreter_is_frozen_and_conflicting_command_paths_fail():
    manifest = {"runs": [{"command": ["/frozen/python312", "train.py"]}],
                "templates": {"gvp": {"command": ["/frozen/python312", "train.py"]}}}
    assert remote_queue.worker_python(manifest) == "/frozen/python312"
    manifest["runs"].append({"command": ["/kernel/python313", "train.py"]})
    with pytest.raises(ValueError, match="conflicting worker interpreters"):
        remote_queue.worker_python(manifest)


def test_pause_observed_after_terminal_completion_drains_before_stop(tmp_path):
    adapter = FakeAdapter(tmp_path, count=1)
    original = adapter.observe
    def paused_terminal():
        state = original()
        if state["pending_persistence"]:
            control.record(tmp_path, "pause", request_id="stop-after-fit", reason="Stop now")
        return state
    adapter.observe = paused_terminal
    result = supervisor.run(tmp_path, adapter, sleep=lambda _: None)
    assert result["reason"] == "user_paused"
    assert adapter.persisted == adapter.launches == ["0"]
