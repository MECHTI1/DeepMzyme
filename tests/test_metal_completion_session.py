"""CPU-only tests of the workstation coordinator: synthetic controller records, fake remote, no cloud calls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import metal_completion_campaign as mc
import metal_completion_session as ms


def put(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) if not isinstance(value, str) else value)
    return path


@pytest.fixture
def prep(tmp_path, monkeypatch):
    root = tmp_path / "prep"
    (root / "code").mkdir(parents=True)
    for name in ("metal_completion_safety.py", "metal_completion_campaign.py"):
        (root / "code" / name).write_bytes(ms.resolve_existing_path(Path(ms.ROOT) / name).read_bytes())
    units = [u.name for u in mc.discovery_units()]
    manifest = put(root / "execution" / "manifest_discovery.json",
                   {"campaign_id": ms.CAMPAIGN_ID, "phase": "discovery", "authorized_grid_units": units,
                    "successor_root": str(root / "execution"),
                    "code_sha256": {"metal_completion_safety.py": ms.sha256_file(root / "code" / "metal_completion_safety.py")}})
    controls = [u.name for u in mc.discovery_units() if u.recipe == "B"]
    put(root / "evidence" / "control_records.json", {"records": {name: {} for name in controls}})
    decision = put(tmp_path / "decision.txt", "I authorize completing the frozen validation-only discovery and confirmation campaign.\n")
    p = ms.Prep(root)
    ms.record_approval(p, phase="discovery", decision_file=decision,
                       decision_sentence="I authorize completing the frozen validation-only discovery",
                       units=ms.new_units(p, "discovery"), gross_ceiling_usd=32.0, max_sessions=8,
                       max_session_seconds=14400, max_total_gpu_seconds=86400, lanes=3,
                       estimated_fit_seconds=2400, reserve_usd=0.5, expires_unix=2_000_000_000,
                       controller_owner="dev@workstation", superseded=[], approval_unix=1_899_900_000,
                       storage_usd_per_day=1.0)
    return p


def controller_dir(tmp_path, *, start=1_900_010_000, maximum=14220, active=True):
    controller = tmp_path / "controller"
    (controller / "state").mkdir(parents=True)
    (controller / "config.env").write_text("PROJECT_ID=p\nZONE=z\nVM_NAME=vm\nSELECTED_INSTANCE_ID=1\n")
    stop = start + maximum
    iso = lambda t: __import__("datetime").datetime.fromtimestamp(t, __import__("datetime").timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: E731
    put(controller / "state" / "current_session.json",
        {"active": active, "project": "p", "zone": "z", "vm_name": "vm", "instance_id": "1",
         "session_id": "session-2", "session_start": iso(start), "termination_ts": iso(stop),
         "max_run_duration_s": maximum})
    put(controller / "state" / "price_cache.json", {"gross_usd_per_hour": 0.879172, "verified_at": iso(start - 3600)})
    events = [
        {"event": "RUNNING", "session_id": "session-1", "session_start": iso(start - 20000), "max_run_duration_s": 14220},
        {"event": "STOPPED", "session_id": "session-1", "session_start": iso(start - 20000),
         "estimated_gross_cost_usd": 3.0, "elapsed_running_s": 12000},
        {"event": "STOPPED", "session_id": "session-0", "session_start": iso(1_800_000_000),
         "estimated_gross_cost_usd": 9.0, "elapsed_running_s": 12000},  # before approval: not counted
        {"event": "RUNNING", "session_id": "session-2", "session_start": iso(start), "max_run_duration_s": maximum},
    ]
    (controller / "state" / "usage_ledger.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    instance = {"status": "RUNNING", "id": "1", "lastStartTimestamp": iso(start + 20),
                "scheduling": {"instanceTerminationAction": "STOP", "maxRunDuration": {"seconds": str(maximum)}}}
    return controller, instance, start, stop


def test_approval_binds_manifest_units_and_decision(prep):
    policy = ms.read_json(prep.policy("discovery"))
    assert policy["execution_authorized"] and policy["manifest_sha256"] == ms.sha256_file(prep.manifest("discovery"))
    assert len(policy["units"]) == 30 and all("__B__" not in u for u in policy["units"])
    assert policy["user_decision"] in prep.decision.read_text()
    assert policy["durable_root"] == str(prep.host_pull)
    with pytest.raises(FileExistsError):
        ms.record_approval(prep, phase="discovery", decision_file=prep.decision, decision_sentence="I authorize",
                           units=policy["units"], gross_ceiling_usd=32.0, max_sessions=8, max_session_seconds=14400,
                           max_total_gpu_seconds=86400, lanes=3, estimated_fit_seconds=2400, reserve_usd=0.5,
                           expires_unix=2e9, controller_owner="x", superseded=[], approval_unix=1.9e9,
                           storage_usd_per_day=1.0)


def test_snapshots_come_from_controller_and_provider_and_pass_the_worker_check(prep, tmp_path):
    controller, instance, start, stop = controller_dir(tmp_path)
    now = start + 600
    out = ms.build_snapshots(prep, "discovery", controller=controller, now=now, instance=instance)
    session, spending = out["session"], out["spending"]
    assert session["provider_hard_stop_verified"] and session["provider_hard_stop_unix"] == stop
    assert session["deadline_unix"] == stop and session["max_total_seconds"] == 14220
    assert spending["spent_before_session_usd"] == 3.0 and spending["allocated_seconds_before_session"] == 14220
    assert spending["session_sha256"] == ms.sha256_file(prep.session)
    runtime = ms.local_admission(prep, "discovery", ms.read_json(prep.policy("discovery"))["units"][0], now=now)
    assert runtime["deadline_unix"] == stop and runtime["lanes"] == 3
    assert runtime["gross_committed_usd"] == pytest.approx(3.0 + 14220 / 3600 * 0.879172 + spending["storage_and_snapshot_forecast_usd"] + 0.5)
    late = stop - 3000
    ms.build_snapshots(prep, "discovery", controller=controller, now=late, instance=instance)
    with pytest.raises(ms.CoordinatorError, match="Insufficient time"):
        ms.local_admission(prep, "discovery", ms.read_json(prep.policy("discovery"))["units"][0], now=late)


@pytest.mark.parametrize("change", ["action", "duration", "status", "inactive", "stale_price"])
def test_snapshots_refuse_unverified_provider_stop_or_stale_inputs(prep, tmp_path, change):
    controller, instance, start, stop = controller_dir(tmp_path, active=change != "inactive")
    if change == "action":
        instance["scheduling"]["instanceTerminationAction"] = "DELETE"
    elif change == "duration":
        instance["scheduling"]["maxRunDuration"] = {"seconds": "99999"}
    elif change == "status":
        instance["status"] = "STOPPING"
    elif change == "stale_price":
        price = ms.read_json(controller / "state" / "price_cache.json")
        price["verified_at"] = "2020-01-01T00:00:00Z"
        put(controller / "state" / "price_cache.json", price)
    with pytest.raises(ms.CoordinatorError):
        ms.build_snapshots(prep, "discovery", controller=controller, now=start + 600, instance=instance)


def test_wrapper_script_records_exit_file_last_and_rotates_earlier_files(prep):
    unit = ms.read_json(prep.policy("discovery"))["units"][0]
    script = ms.launch_script(prep, "discovery", unit, 2, "session-2")
    assert f"{prep.launch}/{unit}.rc.tmp" in script and f"mv {prep.launch}/{unit}.rc.tmp {prep.launch}/{unit}.rc" in script
    assert "--action run" in script and "--policy" in script and "--lane 2" in script and "setsid nohup" in script
    assert "n=$((n+1))" in script and ".launch.json" in script and "boot_id" in script
    assert script.index("echo $? >") < script.index("echo launched")


def test_place_lane_copies_only_artifacts_and_verifies_bytes(prep):
    pulled = prep.host_pull / "lane1"
    put(pulled / "runs" / "unit" / "selected_checkpoint.json", {"a": 1})
    put(pulled / "run_status_unit.json", {"status": "completed"})
    put(pulled / "persistence_receipts" / "x.json", {"receipt": True})
    (pulled / ".host_pull.lock").write_text("")
    result = ms.place_lane(prep, 1)
    assert result["copied"] == 2
    assert (prep.lane(1) / "runs" / "unit" / "selected_checkpoint.json").is_file()
    assert not (prep.lane(1) / "persistence_receipts").exists()
    assert ms.place_lane(prep, 1)["copied"] == 0


def test_collect_map_requires_exit_zero_and_completed_status(prep):
    units = ms.read_json(prep.policy("discovery"))["units"]
    put(prep.launch / f"{units[0]}.rc", "0\n")
    put(prep.lane(0) / f"run_status_{units[0]}.json", {"status": "completed", "run_name": units[0]})
    put(prep.launch / f"{units[1]}.rc", "1\n")
    put(prep.lane(1) / f"run_status_{units[1]}.json", {"status": "failed", "run_name": units[1]})
    put(prep.launch / f"{units[2]}.rc", "0\n")
    result = ms.collect_map(prep, "discovery")
    assert set(result) == {units[0]}
    assert result[units[0]]["run_dir"] == str(prep.lane(0) / "runs" / units[0])
    assert all(Path(v).is_absolute() for v in result[units[0]].values())


class FakeRemote:
    """Probe states are scripted; launches and pulls are recorded."""

    def __init__(self, states):
        self.states, self.calls = list(states), []

    def probe_state(self):
        return self.states.pop(0) if len(self.states) > 1 else self.states[0]


def lane_info(**kw):
    return {"exists": True, "lock_free": True, "pending_unit": None, "pending_transfer": False,
            "ack_uploaded": False, "active_child_alive": False, **kw}


def test_session_loop_launches_pulls_reruns_once_and_ends(prep, monkeypatch):
    units = ms.read_json(prep.policy("discovery"))["units"][:2]
    u0, u1 = units
    free = {"lane0": lane_info(), "lane1": lane_info(), "lane2": lane_info()}
    states = [
        {"statuses": {}, "claims": [], "archived": {}, "launches": {}, "lanes": free},
        {"statuses": {}, "claims": [], "archived": {}, "lanes": {**free, "lane0": lane_info(lock_free=False), "lane1": lane_info(lock_free=False)},
         "launches": {u0: {"lane": 0, "state": "running", "rc": None}, u1: {"lane": 1, "state": "running", "rc": None}}},
        {"statuses": {u0: {"status": "completed"}, u1: {"status": "failed"}}, "claims": [], "archived": {},
         "lanes": {**free, "lane0": lane_info(pending_transfer=True, pending_unit={"run_name": u0, "status": "completed"}),
                   "lane1": lane_info(pending_transfer=True, pending_unit={"run_name": u1, "status": "failed"})},
         "launches": {u0: {"lane": 0, "state": "exited", "rc": "0"}, u1: {"lane": 1, "state": "exited", "rc": "1"}}},
        {"statuses": {u0: {"status": "completed"}, u1: {"status": "failed"}}, "claims": [], "archived": {},
         "lanes": free, "launches": {u0: {"lane": 0, "state": "exited", "rc": "0"}, u1: {"lane": 1, "state": "exited", "rc": "1"}}},
        {"statuses": {u0: {"status": "completed"}, u1: {"status": "completed"}}, "claims": [], "archived": {u1: 1},
         "lanes": free, "launches": {u0: {"lane": 0, "state": "exited", "rc": "0"}, u1: {"lane": 0, "state": "exited", "rc": "0"}}},
    ]
    remote = FakeRemote(states)
    events = []
    monkeypatch.setattr(ms, "probe", lambda r, p, lanes=ms.LANES: remote.probe_state())
    monkeypatch.setattr(ms, "launch", lambda r, p, phase, unit, lane: events.append(("launch", unit, lane)) or
                        {"unit": unit, "lane": lane, "session_id": "s", "launched_at": "t"})
    monkeypatch.setattr(ms, "pull_lanes", lambda r, p, lanes: events.append(("pull", tuple(lanes))) or
                        {"pull": {"lanes": {f"lane{k}": {"status": "acknowledged"} for k in lanes}}, "placement": {}})
    monkeypatch.setattr(ms, "archive_failed", lambda r, p, unit: events.append(("archive", unit)) or {"archived": unit})
    monkeypatch.setattr(ms, "pull_root_records", lambda r, p: events.append(("root",)))
    monkeypatch.setattr(ms, "host_cuda_ready", lambda r: True)
    summary = ms.run_session(remote, prep, "discovery", [u0, u1], lanes=(0, 1, 2), poll=0, sleep=lambda s: None)
    kinds = [e[0] for e in events]
    assert kinds.count("launch") == 3 and ("launch", u1, 0) in events  # the failed unit was rerun once
    assert ("archive", u1) in events and ("pull", (0,)) in events and ("pull", (1,)) in events
    assert summary["remaining_queue"] == [] and events[-1] == ("root",)
    assert summary["ended"][u0]["rc"] == "0"


def test_session_refuses_without_cuda_and_recovers_interrupted_lanes_before_launching(prep, monkeypatch):
    units = ms.read_json(prep.policy("discovery"))["units"][:1]
    monkeypatch.setattr(ms, "host_cuda_ready", lambda r: False)
    with pytest.raises(ms.CoordinatorError, match="CUDA"):
        ms.run_session(object(), prep, "discovery", list(units), poll=0, sleep=lambda s: None)
    free = {"lane0": lane_info(), "lane1": lane_info(), "lane2": lane_info()}
    dead = lane_info(pending_unit={"run_name": "ghost", "status": "running", "persisted": False})
    states = [
        {"statuses": {}, "claims": [], "archived": {}, "launches": {}, "lanes": {**free, "lane0": dead}},
        {"statuses": {}, "claims": [], "archived": {}, "launches": {}, "lanes": free},
        {"statuses": {units[0]: {"status": "completed"}}, "claims": [], "archived": {}, "lanes": free,
         "launches": {units[0]: {"lane": 0, "state": "exited", "rc": "0"}}},
    ]
    remote = FakeRemote(states)
    events = []
    monkeypatch.setattr(ms, "host_cuda_ready", lambda r: True)
    monkeypatch.setattr(ms, "probe", lambda r, p, lanes=ms.LANES: remote.probe_state())
    monkeypatch.setattr(ms, "recover_lane", lambda r, p, lane: events.append(("recover", lane)) or
                        {"pull": {"pull": {"lanes": {f"lane{lane}": {"status": "acknowledged"}}}}})
    monkeypatch.setattr(ms, "launch", lambda r, p, phase, unit, lane: events.append(("launch", unit, lane)) or
                        {"unit": unit, "lane": lane, "session_id": "s", "launched_at": "t"})
    monkeypatch.setattr(ms, "pull_lanes", lambda r, p, lanes: {"pull": {"lanes": {}}, "placement": {}})
    monkeypatch.setattr(ms, "pull_root_records", lambda r, p: None)
    ms.run_session(remote, prep, "discovery", list(units), poll=0, sleep=lambda s: None)
    assert events[0] == ("recover", 0) and ("launch", units[0], 0) in events


def test_lane_with_uploaded_acknowledgement_is_launchable(prep, monkeypatch):
    units = ms.read_json(prep.policy("discovery"))["units"][:1]
    acked = lane_info(pending_transfer=True, ack_uploaded=True,
                      pending_unit={"run_name": "earlier", "status": "interrupted", "persisted": False})
    states = [
        {"statuses": {}, "claims": [], "archived": {}, "launches": {}, "lanes": {"lane0": acked}},
        {"statuses": {units[0]: {"status": "completed"}}, "claims": [], "archived": {}, "lanes": {"lane0": lane_info()},
         "launches": {units[0]: {"lane": 0, "state": "exited", "rc": "0"}}},
    ]
    remote = FakeRemote(states)
    events = []
    monkeypatch.setattr(ms, "host_cuda_ready", lambda r: True)
    monkeypatch.setattr(ms, "probe", lambda r, p, lanes=ms.LANES: remote.probe_state())
    monkeypatch.setattr(ms, "launch", lambda r, p, phase, unit, lane: events.append(("launch", unit, lane)) or
                        {"unit": unit, "lane": lane, "session_id": "s", "launched_at": "t"})
    monkeypatch.setattr(ms, "pull_lanes", lambda r, p, lanes: {"pull": {"lanes": {}}, "placement": {}})
    monkeypatch.setattr(ms, "pull_root_records", lambda r, p: None)
    ms.run_session(remote, prep, "discovery", list(units), lanes=(0,), poll=0, sleep=lambda s: None)
    assert events == [("launch", units[0], 0)]
