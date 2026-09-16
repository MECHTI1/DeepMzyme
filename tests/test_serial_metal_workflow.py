"""Cross-module queue, budget and persistence tests without GPU execution."""
from copy import deepcopy
from pathlib import Path
import shutil
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serial_metal_campaign import budget, evidence, profile, reporting, runtime, workflow
from test_serial_metal_profile import manifest


@pytest.fixture
def campaign(manifest, tmp_path, monkeypatch):
    value = deepcopy(manifest)
    value.update(output_dir=str(tmp_path), root=str(Path(__file__).resolve().parents[1]), profile=profile.PROFILE)
    value["runs"] = profile.initial_runs(value)
    runtime.atomic_json(tmp_path / "campaign_manifest.json", value)
    runtime.atomic_json(tmp_path / "queue.json", dict(runs=value["runs"], phase="screen"))
    monkeypatch.setattr(profile, "verify_manifest", lambda _: value)
    monkeypatch.setattr(evidence, "verify_preparation", lambda *a, **k: {})
    return tmp_path, value


def scored(run, score=.7):
    labels = reporting.NATIVE_CLASSES if run["scheme"] == "six_class" else reporting.COMMON_CLASSES
    return {**run, "balanced_accuracy": score, "minimum_recall": score,
            "collapsed4_balanced_accuracy": score, "collapsed4_minimum_recall": score,
            "per_class_recall": dict.fromkeys(labels, score),
            "collapsed4_per_class_recall": dict.fromkeys(reporting.COMMON_CLASSES, score),
            "cohort_sha256": "shared-discovery", "selected_epoch": 30, "parameter_count": 100}


def measurements(value, elapsed=60):
    return [{**run, "setup_seconds": 10., "epoch_seconds": (elapsed-10)/50,
             "elapsed_seconds": 10+(elapsed-10)/50} for run in value["runs"] if run["stage"] == "operations"]


def status():
    return dict(remaining_seconds=dict(discovery=21600., confirmation=36000., operations=14400.),
                total_remaining_seconds=72000., discovery_closed=False)


def test_preview_is_explicitly_unfunded_and_counts_all_candidate_protection():
    result = budget.historical_preview()
    assert result["certified_reuse"] == 0
    assert result["counts"]["mandatory_discovery"] == 68
    assert result["counts"]["full_fit_inventory_maximum"] == 258
    assert result["full_coverage_buffered_seconds"]/3600 == pytest.approx(21.5399, abs=.001)
    assert not result["discovery_fits_six_hour_cap"]
    assert result["fallback"]["planned_blocks"] == ["core", "five", "fusion"]
    assert result["fallback"]["deferred_blocks"] == ["ring"]


def test_cost_fallback_is_whole_ordered_prefix():
    costs = dict(core=100, five=30, fusion=100, ring=1)
    result = budget.coverage(costs, 170)
    assert result["planned_blocks"] == ["core", "five"]
    assert result["deferred_blocks"] == ["fusion", "ring"]  # cheap tail cannot leapfrog


def test_forecast_reserves_sixteen_unselected_repeats_and_larger_confirmation(campaign):
    output, value = campaign
    queue = runtime.read_json(output / "queue.json")
    forecast = budget.forecast(output, value, queue, [], measurements(value), 1000, status=status())
    assert forecast["pending_discovery_runs"] == 52
    assert forecast["discovery_raw_seconds"] == pytest.approx(68*60)
    assert forecast["full_training_admitted"]
    bigger = measurements(value)
    for row in bigger:
        if row["parameters"]["capacity"] == "large_256":
            row["epoch_seconds"] *= 2
    more = budget.forecast(output, value, queue, [], bigger, 1000, status=status())
    assert more["discovery_raw_seconds"] > forecast["discovery_raw_seconds"]
    assert more["confirmation_raw_seconds"]["core"] > forecast["confirmation_raw_seconds"]["core"]


def test_forecast_does_not_spend_confirmation_to_repair_discovery(campaign):
    output, value = campaign
    result = budget.forecast(output, value, runtime.read_json(output / "queue.json"), [],
                              measurements(value, elapsed=600), 1000, status=status())
    assert not result["discovery_admitted"]
    assert not result["full_training_admitted"]


def test_advance_requires_large_pair_and_then_generates_sixteen_repeats(campaign, monkeypatch):
    output, value = campaign
    rows = [scored(r) for r in value["runs"] if r["stage"] == "discovery"]
    monkeypatch.setattr(workflow, "collect_results", lambda *a, **k: rows[:-1])
    assert workflow.advance(output)["phase"] == "screen"
    assert not runtime.read_json(output / "queue.json").get("top_two")
    monkeypatch.setattr(workflow, "collect_results", lambda *a, **k: rows)
    result = workflow.advance(output)
    assert result["required_repeats"] == 16
    queue = runtime.read_json(output / "queue.json")
    assert all(len(ids) == 2 for ids in queue["top_two"].values())
    assert len([r for r in queue["runs"] if r["block"] == "repeat"]) == 16
    assert not (output / "attempts.json").exists()  # transition never starts training


def test_admit_reserves_full_screen_and_future_repeats_across_sessions(campaign, monkeypatch):
    output, value = campaign
    now = time.time()
    runtime.open_session(output, "session", now, {"gpu": "synthetic"})
    monkeypatch.setattr(workflow, "collect_results", lambda *a, **k: [])
    monkeypatch.setattr(workflow, "_timings", lambda _: measurements(value))
    forecast = budget.forecast(output, value, runtime.read_json(output / "queue.json"), [],
                              measurements(value), 1000)
    monkeypatch.setattr(workflow, "measured_forecast", lambda *a, **k: forecast)
    result = workflow.admit(output)
    assert sum(len(block["run_ids"]) for block in result["comparisons"]) == 68
    assert runtime.budget_status(output)["reserved_seconds"]["discovery"] == pytest.approx(68*60*1.25)
    assert workflow.admit(output)["comparisons"] == result["comparisons"]


def test_discovery_cannot_restart_after_confirmation_freeze(campaign, monkeypatch):
    output, _ = campaign
    runtime.atomic_json(output / "confirmation_manifest.json", {"candidates": {"frozen": True}})
    monkeypatch.setattr(workflow, "collect_results", lambda *a, **k: [])
    with pytest.raises(ValueError, match="cannot change"):
        workflow.advance(output)


def test_optional_candidate_cannot_reduce_confirmation_coverage(monkeypatch):
    baseline = {"confirmation": {"planned_blocks": ["core", "five", "fusion"]}}
    monkeypatch.setattr(workflow, "measured_forecast", lambda *a, **k:
                        {"confirmation": {"planned_blocks": ["core", "five"]}, "full_training_admitted": True})
    with pytest.raises(ValueError, match="reduce confirmation"):
        workflow._admit_optional("unused", [], baseline)


def test_persistence_rejects_stale_state_and_tampered_readback(campaign, monkeypatch, tmp_path):
    output, _ = campaign
    monkeypatch.setattr(runtime, "_is_drive_mount", lambda _: False)
    snapshot = workflow.export_state(output)
    readback = tmp_path / "independent_readback"
    shutil.copytree(snapshot["source_root"], readback)
    receipt = {**snapshot, "readback_root": str(readback), "destination_uri": "drive://test-only"}
    workflow.verify_state_transfer(output, receipt)
    workflow.require_persistent_state(output)
    runtime.atomic_json(output / "queue.json", {"phase": "changed", "runs": []})
    with pytest.raises(ValueError, match="Export current"):
        workflow.require_persistent_state(output)
    snapshot = workflow.export_state(output)
    shutil.rmtree(readback)
    shutil.copytree(snapshot["source_root"], readback)
    (readback / "queue.json").write_text("{}")
    with pytest.raises(ValueError, match="differs"):
        workflow.verify_state_transfer(output, {**receipt, **snapshot})


def test_operation_plan_requires_shutdown_each_remaining_session():
    values = dict(setup_seconds=10, smoke_seconds=10, audit_seconds=10, persistence_seconds=10,
                  recovery_seconds=10, shutdown_seconds=900, sessions_remaining=2, basis="measured test fixture")
    with pytest.raises(ValueError, match="each remaining shutdown"):
        workflow.validate_operations_plan(values)
    values["shutdown_seconds"] = 1800
    assert workflow.validate_operations_plan(values)["raw_seconds"] == 1850


def test_guard_rejects_stale_or_wrong_allocation(campaign):
    output, _ = campaign
    now = time.time()
    runtime.open_session(output, "session", now-5, {"gpu": "test"})
    receipt = dict(session_id="session", allocation_started_epoch=now-5, endpoint="owned",
                   watchdog_verified=True, campaign_profile=profile.PROFILE,
                   campaign_manifest_sha256=profile.base.digest(output / "campaign_manifest.json"),
                   checked_epoch=now-1, expires_epoch=now+119, hard_stop_epoch=now+1000,
                   training_stop_epoch=now+100)
    path = output / "guard.json"
    runtime.atomic_json(path, receipt)
    workflow.verify_host_guard(output, path)
    runtime.atomic_json(path, {**receipt, "expires_epoch": now-1})
    with pytest.raises(ValueError, match="expired"):
        workflow.verify_host_guard(output, path)


@pytest.mark.parametrize("mounted_drive", [False, True])
def test_state_restore_keeps_prior_receipts_and_interrupted_process_ownership(campaign, monkeypatch, mounted_drive):
    """Restored artifacts remain usable, and a saved interrupted fit is reconcilable."""
    from test_serial_metal_runtime import command_run, persist, verify
    output, _ = campaign
    monkeypatch.setattr(runtime, "_is_drive_mount", lambda _: mounted_drive)
    now = time.time()
    runtime.open_session(output, "recovery", now, {"gpu": "synthetic-cpu-fixture"}, now=now)
    for ident in ("prior-one", "prior-two"):
        result = runtime.execute_attempt(output, command_run(output, ident), output, verify)
        persist(output, result)
    attempts = runtime.read_json(output / "attempts.json")
    directory = output / "runs" / "interrupted"
    directory.mkdir()
    (directory / "execution.log").write_text("partial interrupted work\n")
    attempts.append(dict(attempt_id="attempt_00003", run_id="interrupted", session_id="recovery",
                         stage="discovery", status="running", started_epoch=time.time(), run_dir=str(directory)))
    runtime.atomic_json(output / "attempts.json", attempts)
    active = dict(attempt_id="attempt_00003", session_id="recovery", token="owned-fixture",
                  controller_pid=99999999, controller_start_token="absent", hostname="previous-host", boot_id="previous-boot")
    runtime.atomic_json(output / "active_process.json", active)
    if mounted_drive:
        workflow.require_persistent_state(output)
        assert not (output / "state_persistence.json").exists()
    snapshot = workflow.export_state(output)
    assert "active_process.json" in snapshot["files"]
    assert {f"persistence/attempt_{index:05d}.json" for index in (1, 2)} <= snapshot["files"].keys()
    # Artifacts have separately survived transfer. Replace all live controller
    # files with the immutable snapshot to simulate recovery at the same paths.
    for source in workflow.state_files(output):
        source.unlink()
    shutil.copytree(snapshot["source_root"], output, dirs_exist_ok=True)
    assert runtime.read_json(output / "active_process.json") == active
    assert runtime.verify_persistence(output) == ["attempt_00001", "attempt_00002"]
    stopped = time.time()
    runtime.close_session(output, "recovery", stopped, stop_evidence={
        "provider_verified_stopped": True, "session_id": "recovery", "stopped_epoch": stopped})
    interrupted = runtime.reconcile_interrupted(output)
    assert interrupted["attempt_id"] == "attempt_00003"
    assert interrupted["status"] == "interrupted"
    assert interrupted["death_proof"] == "provider_verified_session_stopped"
    assert len(runtime.read_json(output / "attempts.json")) == 3


@pytest.mark.parametrize("mounted_drive", [False, True])
def test_workflow_transfer_launch_intent_survives_loss_before_attempt_persistence(campaign, monkeypatch, mounted_drive):
    from test_serial_metal_runtime import command_run, verify
    output, value = campaign
    run = command_run(output, "intent-smoke", stage="operations")
    runtime.atomic_json(output / "queue.json", {"runs": [run], "phase": "screen"})
    runtime.open_session(output, "launch", time.time(), {"gpu": "synthetic-cpu-fixture"})
    runtime.atomic_json(output / "readiness" / "launch.json", {"status": "inputs_hardware_verified"})
    monkeypatch.setattr(runtime, "_is_drive_mount", lambda _: mounted_drive)
    monkeypatch.setattr(workflow, "verify_host_guard", lambda *a, **k: {"training_stop_epoch": time.time()+1200})
    monkeypatch.setattr(evidence, "verify_result", lambda _output, row, directory: verify(row, directory))
    if mounted_drive:
        completed = workflow.execute(output, "fixture-guard")
        assert completed["status"] == "completed"
        assert not (output / "launch_intents.json").exists()
        assert (output / "persistence" / f"{completed['attempt_id']}.json").exists()
        return
    with pytest.raises(ValueError, match="Export current campaign state"):
        workflow.execute(output, "fixture-guard")
    intents = runtime.read_json(output / "launch_intents.json")
    assert len(intents) == 1
    assert not (output / "attempts.json").exists()
    snapshot = workflow.export_state(output)
    assert "launch_intents.json" in snapshot["files"]
    readback = output / "durable_prelaunch_readback"
    shutil.copytree(snapshot["source_root"], readback)
    workflow.verify_state_transfer(output, {**snapshot, "readback_root": str(readback), "destination_uri": "fixture://durable"})
    completed = workflow.execute(output, "fixture-guard")
    assert completed["attempt_id"] == intents[0]["attempt_id"]
    assert completed["status"] == "completed"
    assert runtime.read_json(output / "launch_intents.json") == intents
    # VM loss before copying the result removes the running/terminal ledger and
    # artifact directory, while its acknowledged intent survives off the VM.
    for source in workflow.state_files(output):
        source.unlink()
    shutil.rmtree(run["run_dir"])
    shutil.copytree(readback, output, dirs_exist_ok=True)
    assert not (output / "attempts.json").exists()
    stopped = time.time()
    runtime.close_session(output, "launch", stopped, stop_evidence={
        "provider_verified_stopped": True, "session_id": "launch", "stopped_epoch": stopped})
    recovered = runtime.reconcile_interrupted(output)
    assert recovered["attempt_id"] == completed["attempt_id"]
    assert recovered["artifact_loss_detected"] and recovered["launch_status_unknown"]
    assert recovered["charged_stage"] == "operations"
    assert runtime.budget_status(output)["used_seconds"]["discovery"] == 0
