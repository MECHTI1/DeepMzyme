"""Serial runtime checks with fake allocation clocks and tiny CPU subprocesses."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serial_metal_campaign import runtime

REPO_ROOT = Path(__file__).resolve().parents[1]


def allocation(output, started=None):
    now = time.time()
    return runtime.open_session(output, "session-1", now if started is None else started,
                                {"gpu": "synthetic-cpu-fixture"}, now=now)


def close_fixture(output, session_id, stopped):
    return runtime.close_session(output, session_id, stopped, stop_evidence={
        "provider_verified_stopped": True, "session_id": session_id, "stopped_epoch": stopped})


def command_run(output, run_id="run-1", behavior="success", stage="discovery"):
    code = """
import argparse, json, pathlib, sys, time
p = argparse.ArgumentParser()
p.add_argument('--runs-dir'); p.add_argument('--run-name')
a = p.parse_args()
directory = pathlib.Path(a.runs_dir) / a.run_name
directory.mkdir(parents=True, exist_ok=True)
behavior = BEHAVIOR
if behavior == 'sleep':
    time.sleep(30)
if behavior == 'ignore-term-with-child':
    import signal, subprocess
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    child = subprocess.Popen([sys.executable, '-c', 'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)'])
    (directory / 'child_pid.txt').write_text(str(child.pid))
    time.sleep(30)
if behavior == 'fail' or (behavior == 'retry' and not a.run_name.endswith('__retry1')):
    sys.exit(7)
(directory / 'done.json').write_text(json.dumps({'run_name': a.run_name}))
""".replace("BEHAVIOR", repr(behavior))
    return {"id": run_id, "stage": stage, "block": "synthetic-block", "family": "synthetic",
            "arm": "four_class", "epochs": 1, "run_dir": str(output / "runs" / run_id),
            "command": [sys.executable, "-c", code, "--runs-dir", str(output / "runs"),
                        "--run-name", run_id], "env": {}, "forecast_seconds": 60}


def verify(run, actual_dir):
    value = json.loads((actual_dir / "done.json").read_text())
    assert value["run_name"] == actual_dir.name
    return {"score": 0.5, "run_id": run["id"]}


def persist(output, attempt):
    copy = output / "readback" / attempt["attempt_id"]
    shutil.copytree(attempt["run_dir"], copy)
    return runtime.record_persistence(output, attempt["attempt_id"], {
        "method": "verified_transfer", "readback_root": str(copy),
        "destination_uri": "fixture://immutable/" + attempt["attempt_id"],
        "artifacts": attempt["artifacts"],
    })


def test_campaign_controller_log_is_a_reusable_training_prelaunch_file(tmp_path):
    allocation(tmp_path)
    run_id = "real-training-run-dir-contract"
    code = f"""
import json, pathlib, sys
from types import SimpleNamespace
sys.path.insert(0, {str(REPO_ROOT / 'src')!r})
from training.run import build_run_dir
runs_dir = pathlib.Path(sys.argv[1])
run_name = sys.argv[2]
directory = build_run_dir(SimpleNamespace(runs_dir=runs_dir, run_name=run_name))
(directory / 'done.json').write_text(json.dumps({{'run_name': directory.name}}))
"""
    run = {
        "id": run_id,
        "stage": "operations",
        "block": "smoke",
        "family": "Only-GVP",
        "arm": "gvp_four",
        "epochs": 1,
        "run_dir": str(tmp_path / "runs" / run_id),
        "command": [sys.executable, "-c", code, str(tmp_path / "runs"), run_id],
        "env": {},
        "forecast_seconds": 60,
    }

    attempt = runtime.execute_attempt(tmp_path, run, tmp_path, verify)

    assert attempt["status"] == "completed"
    assert "execution.log" in attempt["artifacts"]


def test_session_inputs_are_immutable_and_intervals_do_not_overlap(tmp_path):
    output = tmp_path / "campaign"
    first = runtime.open_session(output, "one", 100, {"gpu": "fixture"}, now=101)
    assert runtime.open_session(output, "one", 100, {"gpu": "fixture"}, now=102) == first
    with pytest.raises(ValueError, match="immutable"):
        runtime.open_session(output, "one", 101, {"gpu": "fixture"}, now=102)
    with pytest.raises(RuntimeError, match="Close the previous"):
        runtime.open_session(output, "two", 102, {"gpu": "fixture"}, now=102)
    close_fixture(output, "one", 200)
    with pytest.raises(ValueError, match="overlap"):
        runtime.open_session(output, "two", 199, {"gpu": "fixture"}, now=300)
    runtime.open_session(output, "two", 210, {"gpu": "fixture"}, now=300)
    assert runtime.budget_status(output, now=310)["allocated_seconds"] == 200


def test_failed_work_is_charged_and_only_closed_discovery_can_transfer(tmp_path):
    output = tmp_path
    runtime.open_session(output, "one", 0, {"gpu": "fixture"}, now=0)
    close_fixture(output, "one", 4 * 3600)
    runtime.open_session(output, "two", 5 * 3600, {"gpu": "fixture"}, now=5 * 3600)
    runtime.atomic_json(output / "attempts.json", [
        {"attempt_id": "a", "run_id": "a", "session_id": "one", "stage": "discovery", "started_epoch": 0,
         "ended_epoch": 4 * 3600, "status": "failed"},
        {"attempt_id": "b", "run_id": "b", "session_id": "two", "stage": "discovery", "started_epoch": 5 * 3600,
         "ended_epoch": 6 * 3600, "status": "interrupted"},
        {"attempt_id": "c", "run_id": "c", "session_id": "two", "stage": "confirmation", "started_epoch": 6 * 3600,
         "ended_epoch": 8 * 3600, "status": "completed"},
    ])
    status = runtime.budget_status(output, now=9 * 3600)
    assert status["allocated_seconds"] == 8 * 3600
    assert status["used_seconds"] == {"discovery": 0, "confirmation": 2 * 3600, "operations": 6 * 3600}
    assert status["caps_seconds"]["confirmation"] == 10 * 3600
    runtime.freeze_discovery(output, now=9 * 3600)
    transferred = runtime.budget_status(output, now=9 * 3600)
    assert transferred["caps_seconds"]["confirmation"] == 16 * 3600
    assert transferred["discovery_transferred_seconds"] == 6 * 3600
    assert not runtime.admission(output, "discovery", 1, now=9 * 3600)["allowed"]


def test_admission_preserves_session_closeout_full_block_margin_and_confirmation(tmp_path):
    runtime.open_session(tmp_path, "one", 100, {"gpu": "fixture"}, now=100)
    accepted = runtime.admission(tmp_path, "discovery", 1000, now=100)
    assert accepted["allowed"]
    assert accepted["required_seconds"] == 1250
    assert accepted["deadline"] == 100 + 1250
    assert not runtime.admission(tmp_path, "discovery", 11000, now=100)["allowed"]
    assert not runtime.admission(tmp_path, "discovery", 3000, 19 * 3600, now=100)["allowed"]
    assert runtime.admission(tmp_path, "confirmation", 3000, 19 * 3600, now=100)["allowed"]


def test_setup_and_idle_exhaust_operations_budget(tmp_path):
    runtime.open_session(tmp_path, "one", 0, {"gpu": "fixture"}, now=0)
    close_fixture(tmp_path, "one", 4 * 3600)
    runtime.open_session(tmp_path, "two", 5 * 3600, {"gpu": "fixture"}, now=5 * 3600)
    decision = runtime.admission(tmp_path, "discovery", 1, now=5 * 3600)
    assert not decision["allowed"]
    assert "Operations budget" in decision["reason"]


def test_one_attempt_then_persistence_is_required_before_next(tmp_path):
    allocation(tmp_path)
    first = runtime.execute_attempt(tmp_path, command_run(tmp_path), tmp_path, verify)
    assert first["status"] == "completed"
    assert len(runtime.read_json(tmp_path / "attempts.json")) == 1
    assert not (tmp_path / "active_process.json").exists()
    second_run = command_run(tmp_path, "run-2")
    with pytest.raises(RuntimeError, match="Verify persistent artifacts"):
        runtime.execute_attempt(tmp_path, second_run, tmp_path, verify)
    receipt = persist(tmp_path, first)
    assert receipt["attempt_id"] == first["attempt_id"]
    second = runtime.execute_attempt(tmp_path, second_run, tmp_path, verify)
    assert second["status"] == "completed"
    assert len(runtime.read_json(tmp_path / "attempts.json")) == 2


def test_reuse_does_not_rerun_and_rejects_changed_configuration(tmp_path):
    allocation(tmp_path)
    run = command_run(tmp_path)
    first = runtime.execute_attempt(tmp_path, run, tmp_path, verify)
    assert runtime.execute_attempt(tmp_path, run, tmp_path, verify) == first
    assert runtime.execute_attempt(tmp_path, {**run, "forecast_seconds": 123}, tmp_path, verify) == first
    with pytest.raises(ValueError, match="different inputs"):
        runtime.execute_attempt(tmp_path, {**run, "epochs": 2}, tmp_path, verify)
    (Path(first["run_dir"]) / "done.json").write_text("{}")
    with pytest.raises(ValueError, match="artifacts changed"):
        runtime.execute_attempt(tmp_path, run, tmp_path, verify)


def test_retry_is_linked_fresh_directory_and_same_limit(tmp_path):
    allocation(tmp_path)
    run = command_run(tmp_path, behavior="retry")
    first = runtime.execute_attempt(tmp_path, run, tmp_path, verify)
    assert first["status"] == "failed"
    with pytest.raises(RuntimeError, match="Verify persistent artifacts"):
        runtime.execute_attempt(tmp_path, run, tmp_path, verify)
    persist(tmp_path, first)
    second = runtime.execute_attempt(tmp_path, run, tmp_path, verify)
    assert second["status"] == "completed"
    assert second["retry_of"] == first["attempt_id"]
    assert Path(second["run_dir"]).name == "run-1__retry1"
    assert second["command"][-1] == "run-1__retry1"
    assert Path(first["run_dir"]).exists()
    assert first["elapsed_seconds"] > 0
    assert second["elapsed_seconds"] > 0


def test_second_failure_cannot_be_retried(tmp_path):
    allocation(tmp_path)
    run = command_run(tmp_path, behavior="fail")
    for _ in range(2):
        attempt = runtime.execute_attempt(tmp_path, run, tmp_path, verify)
        assert attempt["status"] == "failed"
        persist(tmp_path, attempt)
    with pytest.raises(RuntimeError, match="one linked retry"):
        runtime.execute_attempt(tmp_path, run, tmp_path, verify)


def test_untracked_directory_cannot_be_overwritten(tmp_path):
    allocation(tmp_path)
    run = command_run(tmp_path)
    Path(run["run_dir"]).mkdir(parents=True)
    with pytest.raises(FileExistsError, match="untracked"):
        runtime.execute_attempt(tmp_path, run, tmp_path, verify)
    assert runtime.read_json(tmp_path / "attempts.json", []) == []


@pytest.mark.parametrize("behavior", ["sleep", "ignore-term-with-child"])
def test_hard_deadline_terminates_child_and_charges_failed_stage(tmp_path, monkeypatch, behavior):
    allocation(tmp_path)
    original = runtime.admission
    def short_admission(*args, **kwargs):
        result = original(*args, **kwargs)
        result["deadline"] = time.time() + 0.6
        return result
    monkeypatch.setattr(runtime, "admission", short_admission)
    monkeypatch.setattr(runtime, "TERM_GRACE_SECONDS", 0.1)
    monkeypatch.setattr(runtime, "KILL_GRACE_SECONDS", 0.1)
    monkeypatch.setattr(runtime, "POLL_SECONDS", 0.02)
    started = time.monotonic()
    attempt = runtime.execute_attempt(tmp_path, command_run(tmp_path, behavior=behavior), tmp_path, verify)
    assert time.monotonic() - started < 3
    assert attempt["status"] == "deadline_stopped"
    assert attempt["elapsed_seconds"] > 0
    assert runtime.budget_status(tmp_path)["used_seconds"]["discovery"] == 0
    assert runtime.budget_status(tmp_path)["used_seconds"]["operations"] >= attempt["elapsed_seconds"]
    if behavior == "ignore-term-with-child":
        pid = int((Path(attempt["run_dir"]) / "child_pid.txt").read_text())
        info = runtime._process_info(pid)
        assert info is None or info["state"] == "Z"


def test_receipt_requires_real_readback_and_immutable_matching_inventory(tmp_path):
    allocation(tmp_path)
    attempt = runtime.execute_attempt(tmp_path, command_run(tmp_path), tmp_path, verify)
    bogus = {"method": "verified_transfer", "readback_root": attempt["run_dir"],
             "destination_uri": "fixture://destination", "artifacts": attempt["artifacts"]}
    with pytest.raises(ValueError, match="independent readback"):
        runtime.record_persistence(tmp_path, attempt["attempt_id"], bogus)
    bogus["method"] = "mounted_drive"
    with pytest.raises(ValueError, match="actual Drive mount"):
        runtime.record_persistence(tmp_path, attempt["attempt_id"], bogus)
    receipt = persist(tmp_path, attempt)
    (Path(receipt["readback_root"]) / "done.json").write_text("changed")
    with pytest.raises(ValueError, match="readback hashes"):
        runtime.record_persistence(tmp_path, attempt["attempt_id"], receipt)


def active_fixture(output, process=None, foreign=False):
    now = time.time()
    identity = runtime._identity()
    active = {"attempt_id": "attempt_00001", "session_id": "session-1", "token": "test",
              "controller_pid": 99999999, "controller_start_token": "absent", **identity}
    if process:
        active.update(pid=process.pid, pgid=process.pid,
                      process_start_token=runtime._process_info(process.pid)["start_token"])
    if foreign:
        active["boot_id"] = "different-boot"
    runtime.atomic_json(output / "active_process.json", active)
    directory = output / "runs" / "interrupted"
    directory.mkdir(parents=True)
    (directory / "execution.log").write_text("partial work\n")
    runtime.atomic_json(output / "attempts.json", [{
        "attempt_id": "attempt_00001", "session_id": "session-1", "stage": "discovery",
        "run_id": "run-1", "status": "running", "started_epoch": now, "run_dir": str(directory),
    }])


def test_reconcile_refuses_live_group_then_records_proven_interruption(tmp_path):
    allocation(tmp_path)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    try:
        active_fixture(tmp_path, child)
        with pytest.raises(RuntimeError, match="still live"):
            runtime.reconcile_interrupted(tmp_path)
        with pytest.raises(RuntimeError, match="still live"):
            close_fixture(tmp_path, "session-1", time.time())
    finally:
        os.killpg(child.pid, signal.SIGTERM)
        child.wait()
    record = runtime.reconcile_interrupted(tmp_path)
    assert record["status"] == "interrupted"
    assert record["death_proof"] == "local_process_group_absent"
    assert not (tmp_path / "active_process.json").exists()


def test_foreign_session_requires_provider_stop_evidence(tmp_path, monkeypatch):
    allocation(tmp_path)
    active_fixture(tmp_path, foreign=True)
    with pytest.raises(RuntimeError, match="Cannot prove"):
        runtime.reconcile_interrupted(tmp_path)
    original = runtime._identity()
    monkeypatch.setattr(runtime, "_identity", lambda: {**original, "boot_id": "new-host-boot"})
    stopped = time.time()
    with pytest.raises(RuntimeError, match="provider-verified"):
        runtime.close_session(tmp_path, "session-1", stopped)
    runtime.close_session(tmp_path, "session-1", stopped, stop_evidence={
        "provider_verified_stopped": True, "session_id": "session-1", "stopped_epoch": stopped,
        "receipt": "synthetic-provider-stop",
    })
    record = runtime.reconcile_interrupted(tmp_path)
    assert record["death_proof"] == "provider_verified_session_stopped"
    assert record["ended_epoch"] == stopped


def test_another_controller_cannot_enter_during_fit(tmp_path):
    with runtime._lock(tmp_path):
        with pytest.raises(RuntimeError, match="serial execution lock"):
            runtime.open_session(tmp_path, "another", time.time(), {"gpu": "fixture"})


def test_cross_campaign_worker_lock_and_foreign_open_session_are_rejected(tmp_path, monkeypatch):
    allocation(tmp_path)
    with runtime._worker_lock():
        with pytest.raises(RuntimeError, match="owns the host training worker"):
            runtime.execute_attempt(tmp_path, command_run(tmp_path), tmp_path, verify)
    original = runtime._identity()
    monkeypatch.setattr(runtime, "_identity", lambda: {**original, "boot_id": "other-boot"})
    with pytest.raises(RuntimeError, match="another host or boot"):
        runtime.execute_attempt(tmp_path, command_run(tmp_path), tmp_path, verify)


def test_missing_pid_crash_window_does_not_prove_no_child_was_launched(tmp_path):
    allocation(tmp_path)
    active_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="launch-window child death"):
        runtime.reconcile_interrupted(tmp_path)
    stopped = time.time()
    runtime.close_session(tmp_path, "session-1", stopped, stop_evidence={
        "provider_verified_stopped": True, "session_id": "session-1", "stopped_epoch": stopped,
    })
    assert runtime.reconcile_interrupted(tmp_path)["status"] == "interrupted"


def test_persistence_launch_checks_all_receipts_but_rehashes_only_latest(tmp_path, monkeypatch):
    allocation(tmp_path)
    for run_id in ("one", "two"):
        attempt = runtime.execute_attempt(tmp_path, command_run(tmp_path, run_id), tmp_path, verify)
        persist(tmp_path, attempt)
    calls = []
    original = runtime.artifact_inventory
    def observed(path):
        calls.append(str(path))
        return original(path)
    monkeypatch.setattr(runtime, "artifact_inventory", observed)
    runtime.execute_attempt(tmp_path, command_run(tmp_path, "three"), tmp_path, verify)
    assert str(tmp_path / "runs" / "one") not in calls
    assert str(tmp_path / "runs" / "two") in calls
    latest = runtime.read_json(tmp_path / "attempts.json")[-1]
    persist(tmp_path, latest)
    calls.clear()
    assert len(runtime.verify_persistence(tmp_path)) == 3
    assert str(tmp_path / "runs" / "one") in calls
    first_receipt = tmp_path / "persistence" / "attempt_00001.json"
    payload = runtime.read_json(first_receipt)
    payload["attempt_id"] = "different"
    runtime.atomic_json(first_receipt, payload)
    with pytest.raises(ValueError, match="different attempt"):
        runtime._require_persistence(tmp_path, runtime.read_json(tmp_path / "attempts.json"))


def test_twenty_hour_allocation_ceiling_cannot_be_reset_by_new_session(tmp_path):
    for index in range(5):
        started = index * 5 * 3600
        runtime.open_session(tmp_path, str(index), started, {"gpu": "fixture"}, now=started)
        close_fixture(tmp_path, str(index), started + 4 * 3600)
    assert runtime.budget_status(tmp_path, now=25 * 3600)["total_remaining_seconds"] == 0
    with pytest.raises(RuntimeError, match="cumulative campaign allocation budget"):
        runtime.open_session(tmp_path, "new", 25 * 3600, {"gpu": "fixture"}, now=25 * 3600)


def test_active_seconds_are_unsettled_then_failure_charged_operations_once(tmp_path):
    runtime.open_session(tmp_path, "one", 100, {"gpu": "fixture"}, now=100)
    attempt = {"attempt_id": "one", "run_id": "run", "session_id": "one", "stage": "discovery",
               "status": "running", "started_epoch": 120, "admission": {"deadline": 220}}
    runtime.atomic_json(tmp_path / "attempts.json", [attempt])
    status = runtime.budget_status(tmp_path, now=150)
    assert status["used_seconds"] == {"discovery": 0, "confirmation": 0, "operations": 20}
    assert status["active_elapsed_seconds"] == 30
    assert status["failure_liability_seconds"] == 100
    assert not runtime.admission(tmp_path, "discovery", 1, now=150)["allowed"]
    attempt.update(status="failed", ended_epoch=150)
    runtime.atomic_json(tmp_path / "attempts.json", [attempt])
    status = runtime.budget_status(tmp_path, now=150)
    assert sum(status["used_seconds"].values()) == status["allocated_seconds"] == 50
    assert status["used_seconds"]["operations"] == 50
    assert runtime.budget_status(tmp_path, now=150) == status


def test_failure_liability_preserves_operations_closeout(tmp_path):
    runtime.open_session(tmp_path, "one", 0, {"gpu": "fixture"}, now=0)
    close_fixture(tmp_path, "one", 3 * 3600)
    runtime.open_session(tmp_path, "two", 4 * 3600, {"gpu": "fixture"}, now=4 * 3600)
    accepted = runtime.admission(tmp_path, "discovery", 2000, now=4 * 3600)
    assert accepted["allowed"]
    assert accepted["remaining"] == 3600 - runtime.CLOSEOUT_SECONDS
    assert not runtime.admission(tmp_path, "discovery", 2200, now=4 * 3600)["allowed"]


def test_comparison_reserved_across_sessions_and_discovery_requires_explicit_deferral(tmp_path):
    units = [f"unit-{index}" for index in range(80)]
    reservation = runtime.reserve_comparison(tmp_path, "all-arms", "discovery", units,
                                             {name: 200 for name in units}, now=0)
    assert reservation["required_seconds"] == 20000 > runtime.SESSION_SECONDS
    runtime.open_session(tmp_path, "one", 0, {"gpu": "fixture"}, now=0)
    assert runtime.admission(tmp_path, "discovery", 200, now=0, block_id="all-arms")["allowed"]
    with pytest.raises(RuntimeError, match="outstanding discovery comparisons"):
        runtime.freeze_discovery(tmp_path, now=0)
    with pytest.raises(ValueError, match="immutable"):
        runtime.reserve_comparison(tmp_path, "all-arms", "discovery", units, {name: 201 for name in units}, now=0)
    close_fixture(tmp_path, "one", 10)
    runtime.open_session(tmp_path, "two", 100, {"gpu": "fixture"}, now=100)
    assert runtime.comparison_status(tmp_path, "all-arms", now=100)["remaining_reserved_seconds"] == 20000
    runtime.release_comparison(tmp_path, "all-arms", "Deferred uniformly: insufficient remaining budget")
    closed = runtime.freeze_discovery(tmp_path, now=100)
    assert closed["transferred_seconds"] == runtime.DISCOVERY_SECONDS
    assert runtime.freeze_discovery(tmp_path, now=110) == closed
    assert not runtime.admission(tmp_path, "discovery", 1, now=110)["allowed"]


def test_malformed_closure_or_overlapping_attempts_are_rejected(tmp_path):
    runtime.open_session(tmp_path, "one", 0, {"gpu": "fixture"}, now=0)
    runtime.atomic_json(tmp_path / "discovery_closed.json", {"status": "closed"})
    with pytest.raises(ValueError, match="frozen discovery closure"):
        runtime.budget_status(tmp_path, now=100)
    (tmp_path / "discovery_closed.json").unlink()
    attempts = [{"attempt_id": str(index), "run_id": str(index), "session_id": "one", "stage": "discovery",
                 "status": "completed", "started_epoch": 10 * index, "ended_epoch": 30 + 10 * index}
                for index in range(2)]
    runtime.atomic_json(tmp_path / "attempts.json", attempts)
    with pytest.raises(ValueError, match="Overlapping"):
        runtime.budget_status(tmp_path, now=100)


def test_reserved_run_identity_and_success_release_are_verified(tmp_path):
    allocation(tmp_path)
    run = command_run(tmp_path)
    runtime.reserve_comparison(tmp_path, "comparison", "discovery", [run], {run["id"]: 60})
    with pytest.raises(ValueError, match="frozen logical comparison"):
        runtime.execute_attempt(tmp_path, {**run, "epochs": 2}, tmp_path, verify, block_id="comparison")
    result = runtime.execute_attempt(tmp_path, run, tmp_path, verify, block_id="comparison")
    assert result["status"] == "completed"
    assert runtime.comparison_status(tmp_path, "comparison")["complete"]
    assert runtime.budget_status(tmp_path)["reserved_seconds"]["discovery"] == 0


def test_short_operations_audit_can_finish_without_full_fifteen_second_shutdown_reserve(tmp_path):
    allocation(tmp_path)
    run = command_run(tmp_path, stage="operations")
    run["forecast_seconds"] = 1
    result = runtime.execute_attempt(tmp_path, run, tmp_path, verify)
    assert result["status"] == "completed"
    assert runtime.budget_status(tmp_path)["used_seconds"]["discovery"] == 0


def test_host_training_deadline_and_fresh_guard_are_checked_before_launch(tmp_path):
    allocation(tmp_path)
    run = command_run(tmp_path)
    with pytest.raises(RuntimeError, match="verified host training deadline"):
        runtime.execute_attempt(tmp_path, run, tmp_path, verify, training_deadline_epoch=time.time()+10)
    assert not Path(run["run_dir"]).exists()
    def expired_guard():
        assert Path(run["run_dir"]).is_dir()
        assert (Path(run["run_dir"]) / "execution.log").exists()
        raise RuntimeError("Host receipt expired during artifact verification")
    with pytest.raises(RuntimeError, match="receipt expired"):
        runtime.execute_attempt(tmp_path, run, tmp_path, verify, launch_guard=expired_guard)
    result = runtime.read_json(tmp_path / "attempts.json")[-1]
    assert result["status"] == "failed"
    assert result["charged_stage"] == "operations"
    assert not (Path(run["run_dir"]) / "done.json").exists()
    assert "attempt_outcome.json" in result["artifacts"]


def test_local_session_cannot_close_without_provider_absence(tmp_path):
    allocation(tmp_path)
    for evidence in (None, {}, {"provider_verified_stopped": True}):
        with pytest.raises(RuntimeError, match="provider-verified stop evidence"):
            runtime.close_session(tmp_path, "session-1", time.time(), stop_evidence=evidence)
    assert runtime.read_json(tmp_path / "sessions.json")[0]["stopped_epoch"] is None


def test_immutable_launch_intent_executes_its_reserved_attempt_once(tmp_path):
    allocation(tmp_path)
    run = command_run(tmp_path)
    intent = runtime.prepare_launch_intent(tmp_path, run)
    assert runtime.prepare_launch_intent(tmp_path, run) == intent
    assert not Path(run["run_dir"]).exists()
    assert runtime.read_json(tmp_path / "attempts.json", []) == []
    with pytest.raises(RuntimeError, match="Unresolved launch intent"):
        runtime.execute_attempt(tmp_path, run, tmp_path, verify)
    with pytest.raises(RuntimeError, match="unresolved prior launch intent"):
        runtime.prepare_launch_intent(tmp_path, command_run(tmp_path, "different"))
    result = runtime.execute_attempt(tmp_path, run, tmp_path, verify, launch_intent_id=intent["intent_id"])
    assert result["attempt_id"] == intent["attempt_id"]
    assert result["status"] == "completed"
    assert runtime.read_json(tmp_path / "launch_intents.json") == [intent]
    assert runtime.reconcile_interrupted(tmp_path) is None


def test_lost_vm_intent_becomes_one_interruption_and_one_linked_retry(tmp_path):
    allocation(tmp_path)
    run = command_run(tmp_path)
    intent = runtime.prepare_launch_intent(tmp_path, run)
    with pytest.raises(RuntimeError, match="provider-verified allocation death"):
        runtime.reconcile_interrupted(tmp_path)
    stopped = time.time()
    close_fixture(tmp_path, "session-1", stopped)
    recovered = runtime.reconcile_interrupted(tmp_path)
    assert recovered["attempt_id"] == intent["attempt_id"]
    assert recovered["artifact_loss_detected"]
    assert recovered["launch_status_unknown"]
    assert recovered["status"] == "interrupted"
    assert recovered["charged_stage"] == "operations"
    assert runtime.reconcile_interrupted(tmp_path) is None
    status = runtime.budget_status(tmp_path)
    assert status["used_seconds"]["operations"] == status["allocated_seconds"]
    persist(tmp_path, recovered)
    runtime.open_session(tmp_path, "second", time.time(), {"gpu": "synthetic-cpu-fixture"})
    retry = runtime.prepare_launch_intent(tmp_path, run)
    assert retry["retry_of"] == intent["attempt_id"]
    assert retry["run_dir"].endswith("__retry1")
    # Lose the second VM before its fit ledger reaches persistence as well.
    close_fixture(tmp_path, "second", time.time())
    second = runtime.reconcile_interrupted(tmp_path)
    persist(tmp_path, second)
    runtime.open_session(tmp_path, "third", time.time(), {"gpu": "synthetic-cpu-fixture"})
    with pytest.raises(RuntimeError, match="one linked retry"):
        runtime.prepare_launch_intent(tmp_path, run)
    assert len(runtime.read_json(tmp_path / "attempts.json")) == 2


def test_durable_running_ledger_before_process_token_requires_provider_stop(tmp_path):
    allocation(tmp_path)
    active_fixture(tmp_path)
    (tmp_path / "active_process.json").unlink()
    assert not (tmp_path / "launch_intents.json").exists()
    with pytest.raises(RuntimeError, match="Ownerless running attempt requires provider-verified"):
        runtime.reconcile_interrupted(tmp_path)
    assert runtime.read_json(tmp_path / "attempts.json")[0]["status"] == "running"
    close_fixture(tmp_path, "session-1", time.time())
    recovered = runtime.reconcile_interrupted(tmp_path)
    assert recovered["status"] == "interrupted"
    assert recovered["recovered_without_process_token"]
    assert recovered["death_proof"] == "provider_verified_session_stopped"
    assert recovered["charged_stage"] == "operations"
    assert not recovered["artifact_loss_detected"]
    assert runtime.reconcile_interrupted(tmp_path) is None
    persist(tmp_path, recovered)
    status = runtime.budget_status(tmp_path)
    assert status["allocated_seconds"] == status["used_seconds"]["operations"]
