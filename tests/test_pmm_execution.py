"""Runtime controls use synthetic artifacts and never contact a GPU provider."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from benchmarking import pmm_execution as execution  # noqa: E402


@pytest.fixture
def clock(monkeypatch):
    now = [10_000.0]
    monkeypatch.setattr(execution.time, "time", lambda: now[0])
    return now


def worker(tmp_path, **policy):
    return execution.CampaignExecution(
        tmp_path / "campaign", durable_root=tmp_path / "durable", session_id="owned-l4",
        policy=execution.ExecutionPolicy(deadline_unix=14_000, max_total_seconds=4000,
                                         allocation_started_unix=9000, **policy),
    )


def artifact(root, name="unit"):
    path = root / "runs" / name / "selected_checkpoint.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"fit_status":"completed"}\n')
    return path


def test_one_worker_owns_campaign_even_inside_same_process(tmp_path, clock):
    with worker(tmp_path):
        with pytest.raises(execution.ExecutionBlocked, match="Another worker"):
            with worker(tmp_path):
                pass
    with worker(tmp_path):
        pass


def test_admission_counts_setup_margin_and_closeout(tmp_path, clock):
    with worker(tmp_path) as owner:
        # 3000s remain from the allocation ceiling: 900s reserve + 1.25*1680.
        result = owner.admit("unit", 1680)
        assert result["required_seconds"] == 3000
        owner.record_result("unit", "completed", 100)
        owner.persist([artifact(owner.root)])
        with pytest.raises(execution.ExecutionBlocked, match="needs"):
            owner.admit("next", 1681)


def test_provider_deadline_wins_and_old_session_start_survives_reentry(tmp_path, clock):
    with worker(tmp_path) as owner:
        pass
    clock[0] = 12_500
    with worker(tmp_path) as owner:
        with pytest.raises(execution.ExecutionBlocked):
            owner.admit("late", 100)
        assert owner.state["allocations"]["owned-l4"]["started_unix"] == 9000


def test_terminal_copy_is_required_and_receipt_hashes_real_readback(tmp_path, clock):
    with worker(tmp_path) as owner:
        owner.admit("unit", 100)
        owner.record_result("unit", "completed", 80)
        with pytest.raises(execution.ExecutionBlocked, match="Persist"):
            owner.admit("next", 100)
        source = artifact(owner.root)
        receipt = owner.persist([source.parent])
        saved_reference = owner.state["pending_unit"]["persistence_receipt"]
        copied = owner.durable_root / "runs/unit/selected_checkpoint.json"
        assert copied.read_bytes() == source.read_bytes()
        assert receipt["files"][0]["sha256"] == hashlib.sha256(copied.read_bytes()).hexdigest()
        assert (owner.durable_root / saved_reference).is_file()
    # Closeout does not replace the unit's manifest with a state-only receipt.
    state = json.loads(owner.state_path.read_text())
    assert state["pending_unit"]["persistence_receipt"] == saved_reference
    assert state["allocations"]["owned-l4"]["provider_stop_verified"] is False


def test_failed_unit_is_saved_then_circuit_opens(tmp_path, clock):
    with worker(tmp_path) as owner:
        owner.admit("unit", 100)
        log = owner.root / "runs/unit.log"
        log.parent.mkdir(parents=True)
        log.write_text("CUDA preflight failed\n")
        owner.record_result("unit", "failed", 10)
        owner.persist([log])
        assert owner.should_stop
        with pytest.raises(execution.ExecutionBlocked, match="failed unit"):
            owner.admit("next", 100)
    assert (owner.durable_root / "runs/unit.log").read_text() == log.read_text()


def test_checksum_failure_blocks_following_launch(tmp_path, clock, monkeypatch):
    with worker(tmp_path) as owner:
        owner.admit("unit", 100)
        owner.record_result("unit", "completed", 10)
        source = artifact(owner.root)
        original = execution._sha256

        def corrupt_readback(path):
            return "wrong" if owner.durable_root in path.parents else original(path)

        with monkeypatch.context() as patched:
            patched.setattr(execution, "_sha256", corrupt_readback)
            with pytest.raises(execution.PersistenceError, match="Checksum mismatch"):
                owner.persist([source])
            assert not owner.state["pending_unit"]["persisted"]
            with pytest.raises(execution.ExecutionBlocked):
                owner.admit("next", 100)
        # The exception has been diagnosed; closeout can still save existing artifacts.


def test_exception_preserves_log_and_allows_deliberate_restart(tmp_path, clock):
    with pytest.raises(RuntimeError, match="training interrupted"):
        with worker(tmp_path) as owner:
            owner.admit("unit", 100)
            artifact(owner.root)
            raise RuntimeError("training interrupted")
    state = json.loads(owner.state_path.read_text())
    assert state["pending_unit"]["status"] == "interrupted"
    assert state["pending_unit"]["restart_semantics"] == "restart original seed"
    assert state["pending_unit"]["persisted"]
    with worker(tmp_path) as owner:
        owner.admit("unit", 100)
        owner.record_result("unit", "completed", 10)
        owner.persist([artifact(owner.root)])


def test_abandoned_owner_recovers_terminal_artifacts_before_new_admission(tmp_path, clock):
    owner = worker(tmp_path).__enter__()
    owner.admit("unit", 100)
    artifact(owner.root)
    owner._release()  # Simulate OS-released flock after a worker dies before closeout.
    with worker(tmp_path) as successor:
        assert successor.state["pending_unit"]["status"] == "interrupted"
        assert successor.state["pending_unit"]["persisted"]
        assert (successor.durable_root / "runs/unit/selected_checkpoint.json").is_file()


def test_reject_invalid_forecasts_nested_storage_and_escaped_artifacts(tmp_path, clock):
    with pytest.raises(ValueError, match="non-nested"):
        execution.CampaignExecution(tmp_path, durable_root=tmp_path / "copy", session_id="s",
                                    policy=execution.ExecutionPolicy(20_000, 1000))
    with worker(tmp_path) as owner:
        with pytest.raises(ValueError, match="positive"):
            owner.admit("unit", float("nan"))
        with pytest.raises(ValueError, match="directory name"):
            owner.admit("../escape", 1)
        with pytest.raises(ValueError, match="inside"):
            owner.persist([tmp_path / "outside.txt"])
        with pytest.raises(execution.PersistenceError, match="missing"):
            owner.persist([owner.root / "missing.txt"])


def test_closeout_storage_error_does_not_hide_original_failure(tmp_path, clock, monkeypatch):
    with pytest.raises(RuntimeError, match="original") as caught:
        with worker(tmp_path) as owner:
            owner.admit("unit", 100)
            artifact(owner.root)
            monkeypatch.setattr(owner, "_copy_verified", lambda _: (_ for _ in ()).throw(OSError("disk full")))
            raise RuntimeError("original")
    assert any("disk full" in note for note in caught.value.__notes__)
    assert owner._lock is None


def test_registered_child_result_and_failed_child_cleanup(tmp_path, clock):
    with worker(tmp_path) as owner:
        owner.admit("unit", 10)
        result = owner.run_subprocess([sys.executable, "-c", "print('verified')"],
                                      stdout=subprocess.PIPE, text=True, check=True)
        assert result.stdout.strip() == "verified"
        assert owner.state["active_child"] is None
        with pytest.raises(subprocess.CalledProcessError):
            owner.run_subprocess([sys.executable, "-c", "raise SystemExit(7)"], check=True)
        assert owner.state["active_child"] is None
        owner.record_result("unit", "failed", 1)


def test_child_timeout_preserves_the_closeout_reserve(tmp_path, clock):
    with worker(tmp_path) as owner:
        owner.admit("unit", 10)
        clock[0] = 12_099.99  # 0.01s left before the 900s closeout reserve starts.
        with pytest.raises(subprocess.TimeoutExpired):
            owner.run_subprocess([sys.executable, "-c", "import time; time.sleep(5)"])
        assert owner.state["active_child"] is None
        clock[0] = 12_100
        with pytest.raises(execution.ExecutionBlocked, match="reserve"):
            owner.run_subprocess([sys.executable, "-c", "pass"])
        owner.record_result("unit", "failed", 10)


def test_live_child_retains_flock_after_controller_sigkill(tmp_path):
    root = tmp_path / "campaign"
    durable = tmp_path / "durable"
    source = str(Path(__file__).resolve().parents[1] / "src")
    code = """
import sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[3])
from benchmarking.pmm_execution import CampaignExecution, ExecutionPolicy
with CampaignExecution(Path(sys.argv[1]), durable_root=Path(sys.argv[2]), session_id='original',
        policy=ExecutionPolicy(time.time()+4000, 4000)) as owner:
    owner.admit('unit', 1)
    owner.run_subprocess([sys.executable, '-c', 'import time; time.sleep(30)'])
"""
    controller = subprocess.Popen([sys.executable, "-c", code, str(root), str(durable), source])
    child = None
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            state_path = root / "execution_state.json"
            if state_path.exists():
                child = json.loads(state_path.read_text()).get("active_child")
                if child:
                    break
            time.sleep(0.02)
        assert child, "Controller failed to record the training child"
        controller.kill()
        controller.wait(timeout=5)
        successor = execution.CampaignExecution(
            root, durable_root=durable, session_id="replacement",
            policy=execution.ExecutionPolicy(time.time() + 4000, 4000),
        )
        with pytest.raises(execution.ExecutionBlocked, match="Another worker"):
            with successor:
                pass
        os.killpg(child["pid"], signal.SIGTERM)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                with successor:
                    assert successor.state["pending_unit"]["status"] == "interrupted"
                break
            except execution.ExecutionBlocked:
                time.sleep(0.02)
        else:
            pytest.fail("Ownership was not released after the orphan child exited")
    finally:
        if controller.poll() is None:
            controller.kill()
            controller.wait(timeout=5)
        if child:
            try:
                os.killpg(child["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass


def host_pull_worker(tmp_path):
    return execution.CampaignExecution(
        tmp_path / "campaign", durable_root=tmp_path / "external_disk", session_id="gcp-allocation",
        persistence_mode="host_pull", policy=execution.ExecutionPolicy(14_000, 4000),
    )


def pull_request(owner):
    transfer = json.loads(owner.state_path.read_text())["pending_transfer"]
    manifest = json.loads((owner.root / transfer["manifest_path"]).read_text())
    for relative in [transfer["manifest_path"], *(entry["path"] for entry in manifest["files"])]:
        destination = owner.durable_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(owner.root / relative, destination)
    return transfer


def test_host_pull_requires_external_readback_before_next_worker(tmp_path, clock):
    with host_pull_worker(tmp_path) as owner:
        owner.admit("unit", 10)
        source = artifact(owner.root)
        owner.record_result("unit", "completed", 10)
        result = owner.persist([source.parent])
        assert result["status"] == "awaiting_host_ack"
        assert owner.should_stop
        assert not owner.durable_root.exists(), "Worker must not pretend host storage is mounted"
        with pytest.raises(execution.ExecutionBlocked, match="Host must"):
            owner.admit("next", 10)
    with pytest.raises(execution.ExecutionBlocked, match="Host must"):
        with host_pull_worker(tmp_path):
            pass
    # Worker closeout cannot change the immutable hashes advertised to the host.
    transfer = pull_request(owner)
    ack_path = owner.durable_root / transfer["ack_path"]
    execution.verify_host_pull(owner.durable_root / transfer["manifest_path"], owner.durable_root, ack_path)
    shutil.copyfile(ack_path, owner.root / transfer["ack_path"])
    with host_pull_worker(tmp_path) as successor:
        assert successor.state["pending_unit"]["persisted"]
        assert not successor.should_stop
        assert successor.state["last_host_transfer"]["manifest_path"] == transfer["manifest_path"]


def test_host_pull_rejects_bad_artifact_and_wrong_ack(tmp_path, clock):
    with host_pull_worker(tmp_path) as owner:
        owner.admit("unit", 10)
        source = artifact(owner.root)
        owner.record_result("unit", "completed", 10)
        owner.persist([source])
    transfer = pull_request(owner)
    destination = owner.durable_root / source.relative_to(owner.root)
    destination.write_text("corrupt")
    ack_path = owner.durable_root / transfer["ack_path"]
    with pytest.raises(execution.PersistenceError, match="checksum"):
        execution.verify_host_pull(owner.durable_root / transfer["manifest_path"], owner.durable_root, ack_path)
    assert not ack_path.exists()
    # A claimed acknowledgment for another manifest is not accepted remotely.
    (owner.root / transfer["ack_path"]).write_text(json.dumps({"manifest_sha256": "wrong"}))
    with pytest.raises(execution.PersistenceError, match="acknowledgment"):
        with host_pull_worker(tmp_path):
            pass
