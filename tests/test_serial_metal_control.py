"""Offline pause and host-owner recovery tests; never allocate a provider VM."""
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serial_metal_campaign import control, runtime, workflow


def pause(output, request="stop-1"):
    return control.record(output, "pause", request_id=request, reason="User requested a stop.")


def resume(output, request="resume-1"):
    return control.record(output, "resume", request_id=request, reason="Continue the existing campaign.",
                          authorization="User explicitly authorized resuming the paused GPU campaign.")


def test_legacy_pause_remains_effective_until_explicit_resume(tmp_path):
    path = tmp_path / "USER_REQUESTED_PAUSE_20260917.json"
    runtime.atomic_json(path, {"requires_new_user_instruction_to_resume": True})
    original = path.read_bytes()
    assert control.status(tmp_path)["paused"]
    with pytest.raises(RuntimeError, match="user-paused"):
        control.require_running(tmp_path)
    with pytest.raises(ValueError, match="explicit user"):
        control.record(tmp_path, "resume", request_id="empty", reason="Resume")
    assert resume(tmp_path)["paused"] is False
    assert path.read_bytes() == original
    runtime.atomic_json(path, {"requires_new_user_instruction_to_resume": True, "new_pause": True})
    assert control.status(tmp_path)["paused"]


def test_pause_writes_during_fit_lock_and_replays_do_not_clear_new_stop(tmp_path):
    with runtime._lock(tmp_path):
        assert pause(tmp_path)["paused"]
    assert resume(tmp_path)["paused"] is False
    assert pause(tmp_path, "stop-2")["paused"]
    replay = resume(tmp_path)
    assert replay["paused"]
    assert replay["sequence"] == 3
    with pytest.raises(ValueError, match="different contents"):
        control.record(tmp_path, "pause", request_id="stop-2", reason="Changed")


def test_paused_worker_cannot_create_an_intent_or_attempt(tmp_path):
    pause(tmp_path)
    with pytest.raises(RuntimeError, match="user-paused"):
        runtime.prepare_launch_intent(tmp_path, {})
    with pytest.raises(RuntimeError, match="user-paused"):
        runtime.execute_attempt(tmp_path, {}, tmp_path, lambda *_: pytest.fail("No training"))
    assert not (tmp_path / "attempts.json").exists()
    assert not (tmp_path / "launch_intents.json").exists()


def test_control_receipts_travel_with_state_and_invalidate_old_snapshot(tmp_path):
    runtime.atomic_json(tmp_path / "USER_REQUESTED_PAUSE_old.json", {"status": "paused"})
    pause(tmp_path)
    before = workflow.state_inventory(tmp_path)
    assert "control_events.json" in before
    assert "USER_REQUESTED_PAUSE_old.json" in before
    resume(tmp_path)
    assert workflow.state_inventory(tmp_path) != before


def test_controller_lease_blocks_duplicate_and_releases(tmp_path):
    with control.controller_lease(tmp_path) as first:
        assert first["pid"] == os.getpid()
        with pytest.raises(RuntimeError, match="controller"):
            with control.controller_lease(tmp_path):
                pytest.fail("Duplicate controller acquired lease")
    with control.controller_lease(tmp_path) as second:
        assert second["token"] != first["token"]
        assert second["previous_token"] == first["token"]


@pytest.mark.parametrize("case", ["live", "foreign_host", "dead_pid", "reused_pid", "new_boot"])
def test_lease_recovery_requires_process_and_boot_evidence(tmp_path, case):
    process = runtime._process_info(os.getpid())
    previous = {"status": "active", "pid": os.getpid(), "token": "previous",
                "process_start_token": process["start_token"], **runtime._identity()}
    if case == "foreign_host":
        previous["hostname"] = "unresolved-other-host"
    elif case == "dead_pid":
        previous["pid"] = 999999999
    elif case == "reused_pid":
        previous["process_start_token"] = "different-start-time"
    elif case == "new_boot":
        previous["boot_id"] = "previous-boot"
    runtime.atomic_json(tmp_path / "host_control/controller_lease.json", previous)
    if case in {"live", "foreign_host"}:
        with pytest.raises(RuntimeError, match="still live|another host"):
            with control.controller_lease(tmp_path):
                pytest.fail("Unresolved owner recovered")
    else:
        with control.controller_lease(tmp_path) as current:
            assert current["previous_token"] == "previous"


def test_malformed_pause_cannot_be_interpreted_as_permission(tmp_path):
    runtime.atomic_json(tmp_path / "control_events.json", {"action": "resume"})
    with pytest.raises(ValueError, match="must be a list"):
        control.require_running(tmp_path)
