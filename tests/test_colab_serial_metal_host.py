"""Offline ownership/watchdog tests; no Colab client or GPU calls are made."""
from datetime import datetime, timezone
from contextlib import contextmanager, redirect_stdout
import importlib.util
import io
import os
from pathlib import Path
from types import SimpleNamespace
import types
import sys

import pytest
import time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("colab_serial_metal_host", ROOT / "scripts/colab_serial_metal_host.py")
host = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host)
runtime = host.runtime
NAME = "deepmzyme-test-one"


class FakeBackend:
    def __init__(self, clock):
        self.clock = clock
        self.histories = {}
        self.mappings = {}
        self.live = [{"endpoint": "unrelated-browser-vm", "accelerator": "T4", "variant": "GPU"}]
        self.created = []
        self.stopped = []
        self.status_calls = []
        self.create_error = None
        self.omit_history = False
        self.keep_assigned = False
        self.stop_error = None
        self.assigned_gpu = None

    def history(self, name):
        return self.histories.get(name, [])

    def mapping(self, name):
        return self.mappings.get(name)

    def assignments(self):
        return self.live

    def create(self, name, gpu):
        self.created.append(name)
        if not self.omit_history:
            entry = {"endpoint": "owned-endpoint", "accelerator": self.assigned_gpu or gpu, "variant": "GPU"}
            self.histories[name] = [{**entry, "event_type": "session_created",
                                     "timestamp": datetime.fromtimestamp(self.clock.wall, timezone.utc).isoformat()}]
            self.mappings[name] = entry
            self.live.append(entry)
        if self.create_error:
            raise self.create_error

    def status(self, name):
        self.status_calls.append(name)

    def stop(self, name, owned):
        self.stopped.append((name, owned["endpoint"]))
        if not self.keep_assigned:
            self.live = [row for row in self.live if row["endpoint"] != owned["endpoint"]]
            self.mappings.pop(name, None)
        if self.stop_error:
            raise self.stop_error


@pytest.fixture
def context(tmp_path, monkeypatch):
    clock = SimpleNamespace(wall=100000.0, monotonic=50000.0)
    monkeypatch.setattr(host.time, "time", lambda: clock.wall)
    monkeypatch.setattr(host.time, "monotonic", lambda: clock.monotonic)
    runtime.atomic_json(tmp_path / "campaign_manifest.json", {"profile": "serial_metal_v2", "schema": 2})
    host.prepare(tmp_path, NAME, "G4")
    backend = FakeBackend(clock)
    def starter(output, name):
        request = host._request(output, name)
        info = runtime._process_info(os.getpid())
        runtime.atomic_json(host._session(output, name) / "watchdog_process.json", {
            "pid": os.getpid(), "process_start_token": info["start_token"], "token": request["watchdog_token"],
            **runtime._identity()})
        host.watchdog(output, name, request["watchdog_token"], backend=backend, once=True)
    return tmp_path, clock, backend, starter


def test_prepare_is_pure_and_gpu_name_is_exact(tmp_path, monkeypatch):
    runtime.atomic_json(tmp_path / "campaign_manifest.json", {"profile": "test"})
    def forbidden(*args, **kwargs):
        pytest.fail("Preparing a host campaign must not invoke Colab")
    monkeypatch.setattr(host, "CLIBackend", forbidden)
    assert host.prepare(tmp_path, NAME, "G4")["status"] == "prepared_not_allocated"
    with pytest.raises(FileExistsError, match="immutable"):
        host.prepare(tmp_path, NAME, "G4")
    with pytest.raises(ValueError, match="Unsupported accelerator"):
        host.prepare(tmp_path, "deepmzyme-another", "L40")


def test_cli_backend_preserves_virtual_environment_interpreter_symlink(tmp_path):
    interpreter = tmp_path / "tool-env/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    backend = host.CLIBackend(interpreter, tmp_path / "history")
    assert backend.cli_python == str(interpreter.absolute())


def test_allocate_requires_startup_ack_before_new(context):
    output, clock, backend, _ = context
    with pytest.raises(RuntimeError, match="startup did not prove liveness"):
        host.allocate(output, NAME, backend=backend, starter=lambda *args: None)
    assert backend.created == []
    assert host.host_budget(output)["allocated_seconds"] == 0
    assert host._request(output, NAME)["status"] == "cancelled_before_request"


def test_verified_allocation_receipt_has_fresh_bounded_worker_authorization(context):
    output, clock, backend, starter = context
    result = host.allocate(output, NAME, backend=backend, starter=starter)
    assert result["hard_deadline_epoch"] == clock.wall + 4 * 3600
    assert result["training_deadline_epoch"] == clock.wall + 4 * 3600 - 900
    assert result["stop_request_epoch"] == clock.wall + 4 * 3600 - 300
    assert backend.status_calls == [NAME]
    receipt = host.worker_receipt(output, NAME, backend=backend)
    assert receipt["watchdog_verified"]
    assert receipt["endpoint"] == "owned-endpoint"
    assert receipt["expires_epoch"] == clock.wall + 120
    assert receipt["campaign_profile"] == "serial_metal_v2"
    assert receipt["allocation_started_epoch"] == clock.wall
    assert host.host_budget(output)["open_session_ids"] == [NAME]


def test_watchdog_loss_immediately_requests_only_owned_stop(context):
    output, clock, backend, starter = context
    host.allocate(output, NAME, backend=backend, starter=starter)
    clock.wall += host.HEARTBEAT_MAX_AGE + 1
    status = host.status(output, NAME, backend=backend)
    assert not status["work_allowed"]
    assert backend.stopped == [(NAME, "owned-endpoint")]
    assert backend.live == [{"endpoint": "unrelated-browser-vm", "accelerator": "T4", "variant": "GPU"}]
    assert not host.host_budget(output)["open_session_ids"]


def test_lost_create_response_reconciles_owned_history_and_stops(context):
    output, clock, backend, starter = context
    backend.create_error = TimeoutError("lost create response")
    with pytest.raises(TimeoutError):
        host.allocate(output, NAME, backend=backend, starter=starter)
    assert backend.stopped == [(NAME, "owned-endpoint")]
    assert host._request(output, NAME)["status"] == "stopped_verified"


def test_unknown_create_outcome_blocks_next_session_and_remains_charged(context):
    output, clock, backend, starter = context
    backend.create_error = TimeoutError()
    backend.omit_history = True
    with pytest.raises(TimeoutError):
        host.allocate(output, NAME, backend=backend, starter=starter)
    clock.wall += 100
    assert host.host_budget(output)["allocated_seconds"] == 100
    assert backend.stopped == []
    with pytest.raises(RuntimeError, match="preceding owned allocation"):
        host.prepare(output, "deepmzyme-next", "G4")


def test_wrong_hardware_is_stopped_without_authorizing_work(context):
    output, clock, backend, starter = context
    backend.assigned_gpu = "A100"
    with pytest.raises(RuntimeError, match="Assigned hardware differs"):
        host.allocate(output, NAME, backend=backend, starter=starter)
    assert backend.stopped == [(NAME, "owned-endpoint")]
    assert not (host._session(output, NAME) / "allocation_receipt.json").exists()


def test_stop_requires_provider_absence_even_after_successful_cli(context):
    output, clock, backend, starter = context
    host.allocate(output, NAME, backend=backend, starter=starter)
    backend.keep_assigned = True
    clock.wall += 50
    with pytest.raises(RuntimeError, match="remains assigned"):
        host.stop(output, NAME, backend=backend)
    assert host._request(output, NAME)["stopped_epoch"] is None
    assert host.host_budget(output)["allocated_seconds"] == 50
    assert runtime.read_json(host._session(output, NAME) / "stop_required.json")["status"] == "stop_unverified"
    backend.keep_assigned = False
    backend.stop_error = TimeoutError()
    receipt = host.stop(output, NAME, backend=backend)
    assert receipt["provider_verified_stopped"]
    assert receipt["stop_response_error"] == "TimeoutError"
    clock.wall += 50
    assert host.stop(output, NAME, backend=backend) == receipt
    assert host.host_budget(output)["allocated_seconds"] == 50


def test_unrelated_replacement_mapping_is_never_stopped(context):
    output, clock, backend, starter = context
    host.allocate(output, NAME, backend=backend, starter=starter)
    backend.mappings[NAME] = {"endpoint": "unrelated-browser-vm"}
    with pytest.raises(RuntimeError, match="unrelated endpoint"):
        host.stop(output, NAME, backend=backend)
    assert backend.stopped == []
    assert host._request(output, NAME)["stopped_epoch"] is None


def test_watchdog_requests_early_stop_on_monotonic_deadline_despite_wall_rollback(context):
    output, clock, backend, starter = context
    receipt = host.allocate(output, NAME, backend=backend, starter=starter)
    clock.monotonic += 4 * 3600 - 300
    clock.wall += 100
    result = host.watchdog(output, NAME, receipt["watchdog_token"], backend=backend, once=True)
    assert result["provider_verified_stopped"]
    assert backend.stopped == [(NAME, "owned-endpoint")]


def test_last_allocation_uses_only_remaining_twenty_hour_budget(context):
    output, clock, backend, starter = context
    runtime.atomic_json(host._root(output) / "allocations.json", [
        {"session_id": "deepmzyme-prior", "started_epoch": 0, "stopped_epoch": 19 * 3600,
         "status": "stopped_verified"}])
    receipt = host.allocate(output, NAME, backend=backend, starter=starter)
    assert receipt["prior_allocated_seconds"] == 19 * 3600
    assert receipt["hard_deadline_epoch"] == clock.wall + 3600
    clock.wall += 3600
    host.stop(output, NAME, backend=backend)
    host.prepare(output, "deepmzyme-next", "G4")
    with pytest.raises(RuntimeError, match="Cumulative allocation budget"):
        host.allocate(output, "deepmzyme-next", backend=backend, starter=starter)


def test_prior_worker_allocations_are_imported_without_reset(tmp_path):
    runtime.atomic_json(tmp_path / "campaign_manifest.json", {"profile": "test"})
    runtime.atomic_json(tmp_path / "sessions.json", [{
        "session_id": "old", "started_epoch": 100, "stopped_epoch": 200,
        "stop_evidence": {"session_id": "old", "stopped_epoch": 200, "provider_verified_stopped": True}}])
    host.prepare(tmp_path, NAME, "G4")
    assert host.host_budget(tmp_path)["allocated_seconds"] == 100
    runtime.atomic_json(tmp_path / "campaign_manifest.json", {"profile": "different"})
    with pytest.raises(ValueError, match="different campaign manifest"):
        host.prepare(tmp_path, "deepmzyme-next", "G4")


def test_unverified_prior_worker_history_cannot_reset_host_budget(tmp_path):
    runtime.atomic_json(tmp_path / "campaign_manifest.json", {"profile": "test"})
    runtime.atomic_json(tmp_path / "sessions.json", [{"session_id": "old", "started_epoch": 100,
                                                   "stopped_epoch": 200, "stop_evidence": {}}])
    with pytest.raises(RuntimeError, match="verified stop receipts"):
        host.prepare(tmp_path, NAME, "G4")


def test_history_reuse_or_changed_manifest_blocks_create(context):
    output, clock, backend, starter = context
    backend.histories[NAME] = [{"event_type": "old_event"}]
    with pytest.raises(RuntimeError, match="prior CLI history"):
        host.allocate(output, NAME, backend=backend, starter=starter)
    backend.histories.clear()
    runtime.atomic_json(output / "campaign_manifest.json", {"profile": "changed"})
    with pytest.raises(ValueError, match="manifest changed"):
        host.allocate(output, NAME, backend=backend, starter=starter)
    assert backend.created == []


def test_real_local_detached_watchdog_acknowledges_and_exits_after_verified_stop(tmp_path):
    runtime.atomic_json(tmp_path / "campaign_manifest.json", {"profile": "test"})
    host.prepare(tmp_path, NAME, "G4")
    class LiveClock:
        @property
        def wall(self):
            return time.time()
    backend = FakeBackend(LiveClock())
    receipt = host.allocate(tmp_path, NAME, backend=backend)
    record = runtime.read_json(host._session(tmp_path, NAME) / "watchdog_process.json")
    try:
        assert host.watchdog_alive(tmp_path, NAME)
        stopped = host.stop(tmp_path, NAME, backend=backend)
        assert stopped["provider_verified_stopped"]
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            info = runtime._process_info(record["pid"])
            if info is None or info["state"] == "Z":
                break
            time.sleep(0.1)
        else:
            pytest.fail("Detached watchdog did not exit after verified allocation stop")
    finally:
        info = runtime._process_info(record["pid"])
        if info and info["state"] != "Z" and info["start_token"] == record["process_start_token"]:
            os.kill(record["pid"], 9)


@pytest.mark.parametrize("mapped", [None, "owned", "unrelated"])
def test_cli_named_stop_keeps_mapping_locked_through_provider_unassignment(tmp_path, monkeypatch, mapped):
    entries = {} if mapped is None else {NAME: SimpleNamespace(endpoint=mapped)}
    calls = []
    class Store:
        locked = False
        @contextmanager
        def _lock_exclusive(self):
            self.locked = True
            try:
                yield None
            finally:
                self.locked = False
        def _load_raw(self, stream):
            return entries
        def _save_raw(self, stream, value):
            assert self.locked
    store = Store()
    class State:
        @property
        def store(self):
            return self._store
    state = State()
    state._store = store
    def fake_stop(session):
        assert store.locked
        assert state.store.get(session).endpoint == "owned"
        calls.append(session)
        state.store.remove(session)
    modules = {
        "colab_cli": types.ModuleType("colab_cli"),
        "colab_cli.auth": SimpleNamespace(_run_remote_flow=None),
        "colab_cli.common": SimpleNamespace(state=state),
        "colab_cli.state": SimpleNamespace(SessionState=lambda **kwargs: SimpleNamespace(**kwargs)),
        "colab_cli.commands": types.ModuleType("colab_cli.commands"),
        "colab_cli.commands.session": SimpleNamespace(stop=fake_stop),
    }
    modules["colab_cli"].auth = modules["colab_cli.auth"]
    def offline_run(command, **kwargs):
        with monkeypatch.context() as patch:
            for name, module in modules.items():
                patch.setitem(sys.modules, name, module)
            patch.setattr(sys, "argv", ["-c", *command[3:]])
            buffer = io.StringIO()
            try:
                with redirect_stdout(buffer):
                    exec(compile(command[2], "<offline_colab_backend>", "exec"), {})
            except RuntimeError:
                return SimpleNamespace(returncode=1, stdout="")
            return SimpleNamespace(returncode=0, stdout=buffer.getvalue())
    monkeypatch.setattr(host.subprocess, "run", offline_run)
    backend = host.CLIBackend(sys.executable, tmp_path)
    owned = {"endpoint": "owned", "accelerator": "G4", "variant": "GPU"}
    if mapped == "unrelated":
        with pytest.raises(RuntimeError, match="Colab stop_owned failed"):
            backend.stop(NAME, owned)
        assert entries[NAME].endpoint == "unrelated"
        assert calls == []
    else:
        backend.stop(NAME, owned)
        assert calls == [NAME]
        assert entries == {}
