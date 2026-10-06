"""Workstation step-C launcher: lane readiness, duplicate refusal, detached starts, recovery with 'wait', session
time, verified per-lane pulls that block further work, and the evidence copy (no VM contact)."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import pmm_v3_step_b as b  # noqa: E402
import pmm_v3_step_c as c  # noqa: E402

LF4 = "gvp_late_fusion__four_class__baseline__fold0__seed42"
ESM4 = "only_esm__four_class__baseline__fold0__seed42"
ESM5 = "only_esm__five_class__baseline__fold0__seed42"
GVP4 = "only_gvp__four_class__baseline__fold0__seed42"
LF5 = "gvp_late_fusion__five_class__baseline__fold0__seed42"
PROBE = "probe-full-fp32__" + LF4


def lane(**overrides):
    info = {"exists": True, "lock_free": True, "active_child_alive": False, "pending_transfer": True,
            "ack_uploaded": True, "pending_unit": {"run_name": "x", "status": "completed", "persisted": False}}
    return {**info, **overrides}


def vm(*, regression="passed", **overrides):
    """A VM snapshot after step B: setting recorded (3 lanes, LF4 reused), every lane idle and acknowledged."""
    statuses = {PROBE: {"status": "completed", "lane": 0, "gate_passed": None, "family": "gvp_late_fusion",
                        "target": "four_class"}}
    if regression is not None:
        statuses[c.REGRESSION_NAME] = {"status": "completed" if regression == "passed" else regression, "lane": 0,
                                       "gate_passed": regression == "passed", "family": None, "target": None}
    state = {"settings": {"amp": False, "concurrent_lanes": 3, "reuse": {LF4: PROBE}}, "statuses": statuses,
             "claims": sorted(statuses), "archived": {}, "launches": {},
             "lanes": {f"lane{k}": lane() for k in range(3)}, "width": 3}
    for key, value in overrides.items():
        state[key] = value
    return state


def session(remaining=14000.0, deadline=4e9):
    return {"session_id": "s-1", "allocation_started": 1000.0, "deadline": deadline, "max_seconds": 14220.0,
            "remaining_seconds": remaining}


class FakeVM:
    """Answers the launcher's probes from a sequence of snapshots and records every script."""

    def __init__(self, *snapshots, launch_error=None, contact_errors=0):
        self.snapshots, self.scripts = list(snapshots), []
        self.launch_error, self.contact_errors, self.transport = launch_error, contact_errors, None

    def current(self):
        return self.snapshots[0] if len(self.snapshots) == 1 else self.snapshots.pop(0)

    def run(self, script, *, check=True, timeout=None):
        self.scripts.append(script)
        assert len(self.scripts) < 400, "runaway loop: a refusal or a wait exit no longer works"
        if "setsid nohup" in script:
            if self.launch_error:
                raise self.launch_error
            return subprocess.CompletedProcess([], 0, stdout="launched\n", stderr="")
        if "boot_id" in script and "launch.json" in script:  # STATE_PROBE, the first read of every poll
            if self.contact_errors:
                self.contact_errors -= 1
                raise b.LauncherError("remote command failed (255): connection reset")
            self.state = self.current()
            out = {k: v for k, v in self.state.items() if k not in ("lanes", "width")}
        elif "fcntl" in script:  # LANE_PROBE
            out = self.state["lanes"]
        else:
            out = {"archived_to": "failed_attempts/x/attempt1"}
        return subprocess.CompletedProcess([], 0, stdout=json.dumps(out), stderr="")

    def launched(self):
        return [s for s in self.scripts if "setsid nohup" in s]


def launch_record(unit, lane_index, state):
    return {"unit": unit, "lane": lane_index, "kind": "unit", "session_id": "s-1", "state": state,
            "rc": "0" if state == "exited" else None}


# ---------------------------------------------------------------------------
# Unit list, offline plan and import hygiene
# ---------------------------------------------------------------------------

def test_unit_list_matches_the_frozen_campaign_module():
    import pmm_v3_campaign as v3
    from benchmarking import pmm_ion_campaign as v2

    assert c.STEP_C_UNITS == tuple(unit.name for unit in v3.step_units("C"))
    assert c.REGRESSION_NAME == v3.REGRESSION_NAME
    assert c.USES_ESM == {family: spec["uses_esm"] for family, spec in v2.FAMILIES.items()}
    assert sorted(c.SUGGESTED_ORDER) == sorted(set(c.STEP_C_UNITS) - {LF4})


def test_launcher_never_imports_torch():
    code = "import sys; sys.path.insert(0, %r); import pmm_v3_step_c; print('torch' in sys.modules)" % str(ROOT)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False"


# ---------------------------------------------------------------------------
# Lane readiness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("info, message", [
    (lane(pending_unit={"run_name": "u", "status": "running", "persisted": False}, pending_transfer=False),
     "running or was interrupted"),
    (lane(lock_free=False), "lane lock"),
    (lane(active_child_alive=True), "training child"),
    (lane(pending_unit={"run_name": "u", "status": "failed", "persisted": False}, pending_transfer=False),
     "without a persisted copy")])
def test_a_busy_target_lane_is_refused(info, message):
    state = vm(lanes={"lane0": lane(), "lane1": info, "lane2": lane()})
    with pytest.raises(c.LauncherError, match=message):
        c.plan_units([(ESM4, 1)], state)
    assert c.plan_units([(ESM4, 2)], state)[0]["lane"] == 2  # another, idle lane is fine


def test_a_lane_with_a_launch_still_starting_is_refused():
    state = vm(launches={GVP4: launch_record(GVP4, 1, "starting")}, claims=[GVP4])
    with pytest.raises(c.LauncherError, match="lane1: .* no exit code yet"):
        c.plan_units([(ESM4, 1)], state)
    assert c.plan_units([(ESM4, 0)], state)


# ---------------------------------------------------------------------------
# Refusal of duplicate and out-of-plan launches; regression first and alone
# ---------------------------------------------------------------------------

def failed(status="failed"):
    return {"status": status, "lane": 1, "gate_passed": None, "family": "only_esm", "target": "four_class"}


@pytest.mark.parametrize("requests, change, message", [
    ([(ESM4, 0)], lambda s: s["statuses"].update({ESM4: {**failed(), "status": "completed"}}), "never rerun"),
    ([(ESM4, 0)], lambda s: s["claims"].append(ESM4), "claimed without a terminal status"),
    ([(ESM4, 0)], lambda s: s["launches"].update({ESM4: launch_record(ESM4, 1, "running")}), "use 'wait'"),
    ([(ESM4, 0)], lambda s: s["statuses"].update({ESM4: failed()}), "archive-failed"),
    ([(ESM4, 0)], lambda s: (s["statuses"].update({ESM4: failed()}), s["archived"].update({ESM4: 1})),
     "failed after its one rerun"),
    ([(LF4, 0)], lambda s: None, "reused from step B"),
    (["only_gvp__four_class__meanagg__fold0__seed42@0"], lambda s: None, "not a step C unit"),
    ([(ESM4, 0), (ESM4, 1)], lambda s: None, "listed twice"),
    ([(ESM4, 0), (ESM5, 0)], lambda s: None, "share a lane"),
    ([(ESM4, 3)], lambda s: None, "outside the recorded 3 lanes"),
    ([(ESM4, 0), (ESM5, 1), (GVP4, 2), (LF5, 0)], lambda s: None, "1-3 units")])
def test_duplicate_and_out_of_plan_launches_are_refused(requests, change, message):
    state = vm()
    change(state)
    requests = [c.parse_request(r) if isinstance(r, str) else r for r in requests]
    remote = FakeVM(state)
    with pytest.raises(c.LauncherError, match=message):
        c.cmd_launch(remote, session(), requests, wait=False, sleep=lambda s: None)
    assert not remote.launched()


def test_a_rerun_after_archive_failed_is_allowed_once():
    state = vm(archived={ESM4: 1}, launches={ESM4: launch_record(ESM4, 1, "exited")})
    assert c.plan_units([(ESM4, 1)], state)[0]["unit"] == ESM4


@pytest.mark.parametrize("regression, extra, message", [
    (None, {}, "run 'regression' first"),
    ("failed_regression_gate", {}, "STOP"),
    ("failed_independent_replay", {}, "STOP"),
    (None, {"launches": {c.REGRESSION_NAME: launch_record(c.REGRESSION_NAME, 0, "running")}}, "still running")])
def test_units_wait_for_a_passed_regression_gate(regression, extra, message):
    with pytest.raises(c.LauncherError, match=message):
        c.plan_units([(ESM4, 1)], vm(regression=regression, **extra))


def test_the_regression_runs_first_and_alone():
    planned = c.plan_regression(vm(regression=None))
    assert planned == [{"unit": c.REGRESSION_NAME, "lane": 0, "fit_seconds": float(c.REGRESSION_FIT_SECONDS)}]
    busy = vm(regression=None, launches={ESM4: launch_record(ESM4, 2, "running")})
    with pytest.raises(c.LauncherError, match="goes alone"):
        c.plan_regression(busy)
    with pytest.raises(c.LauncherError, match="lane2: .*running or was interrupted"):
        c.plan_regression(vm(regression=None, lanes={"lane0": lane(), "lane1": lane(), "lane2": lane(
            pending_unit={"run_name": "u", "status": "running", "persisted": False}, pending_transfer=False)}))
    for outcome in ("passed", "failed_regression_gate"):
        with pytest.raises(c.LauncherError, match="never repeated"):
            c.plan_regression(vm(regression=outcome))
    command = c.unit_command(c.REGRESSION_NAME, 0, session(), fit_seconds=3300)
    assert command[command.index("--action") + 1] == "regression" and "--unit" not in command
    assert command[command.index("--v2-root") + 1] == b.REMOTE_V2 and command[command.index("--lane") + 1] == "0"


# ---------------------------------------------------------------------------
# Session expiry and admission time
# ---------------------------------------------------------------------------

def test_launches_are_refused_without_enough_session_time():
    remote = FakeVM(vm())
    with pytest.raises(c.LauncherError, match="hard stop"):
        c.cmd_launch(remote, session(remaining=3000.0), [(ESM4, 0)], sleep=lambda s: None)
    assert not remote.launched()
    remote = FakeVM(vm(regression=None))
    with pytest.raises(c.LauncherError, match="hard stop"):
        c.cmd_regression(remote, session(remaining=4000.0), sleep=lambda s: None)
    assert not remote.launched()


def test_session_expiry_is_refused_by_the_shared_reader(tmp_path):
    root = tmp_path / "ctl"
    (root / "state").mkdir(parents=True)
    (root / "config.env").write_text("PROJECT_ID=p\nZONE=z\nVM_NAME=v\nSELECTED_INSTANCE_ID=1\n")
    (root / "state" / "current_session.json").write_text(json.dumps({
        "active": True, "project": "p", "zone": "z", "vm_name": "v", "instance_id": "1", "session_id": "s",
        "session_start": "2026-10-06T08:00:00Z", "termination_ts": "2026-10-06T11:57:00Z",
        "max_run_duration_s": 14220}))
    with pytest.raises(b.LauncherError, match="hard stop has passed"):
        b.read_session(root, now=b._stamp("2026-10-06T12:00:00Z"))


def test_forecasts_charge_a_cold_cache_set_until_a_fit_built_it():
    statuses = vm()["statuses"]
    assert c.forecast_seconds(ESM4, statuses) == c.FIT_SECONDS["only_esm"]  # step-B late fusion built (four, ESMC)
    assert c.forecast_seconds(GVP4, statuses) == c.FIT_SECONDS["only_gvp"] + c.COLD_CACHE_SECONDS  # no ESMC
    assert c.forecast_seconds(ESM5, statuses) == c.FIT_SECONDS["only_esm"] + c.COLD_CACHE_SECONDS
    statuses[LF5] = {"status": "completed", "family": "gvp_late_fusion", "target": "five_class"}
    assert c.forecast_seconds(ESM5, statuses) == c.FIT_SECONDS["only_esm"]
    assert c.plan_units([(ESM5, 0)], vm(), fit_seconds=1234)[0]["fit_seconds"] == 1234.0


# ---------------------------------------------------------------------------
# Detached starts (the real shell script and state probe, run locally)
# ---------------------------------------------------------------------------

def test_launch_script_shape():
    planned = [{"unit": ESM4, "lane": 1, "fit_seconds": 1800.0}, {"unit": GVP4, "lane": 2, "fit_seconds": 3800.0}]
    script = c.launch_script("c-x", planned, session())
    assert script.count("setsid nohup") == 1 and script.rstrip().endswith("echo launched")
    assert script.index("setsid") > script.index("mv \"$f\"")  # earlier attempts' files kept, then launch
    for item in planned:
        command = c.unit_command(item["unit"], item["lane"], session(), fit_seconds=item["fit_seconds"])
        assert command[command.index("--unit") + 1] == item["unit"]
        assert command[command.index("--lane") + 1] == str(item["lane"])
        assert command[command.index("--estimated-fit-seconds") + 1] == repr(item["fit_seconds"])
        assert command[command.index("--persistence-mode") + 1] == "host_pull"
        assert command[command.index("--durable-root") + 1] == str(b.DURABLE)
        assert "--amp" not in command and "--epochs" not in command  # units follow the recorded setting
        for suffix in (".launch.json", ".boot", ".pid", ".rc"):
            assert f"{item['unit']}{suffix}" in script


class LocalRemote:
    """Runs the launcher's scripts with local bash/python against a scratch 'VM' tree."""

    def __init__(self):
        from pmm_v3_host_pull import LocalTransport

        self.transport = LocalTransport()

    def run(self, script, *, check=True, timeout=None):
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        return result


@pytest.fixture
def local_vm(tmp_path, monkeypatch):
    v3, code = tmp_path / "vm_v3", tmp_path / "vm_code"
    (v3 / "lanes").mkdir(parents=True)
    code.mkdir()
    (v3 / "execution_settings.json").write_text(json.dumps({"amp": False, "concurrent_lanes": 3, "reuse": {}}))
    (code / "run_pmm_v3_campaign.py").write_text(
        "import sys, time\nunit = sys.argv[sys.argv.index('--unit') + 1]\ntime.sleep(3)\n"
        "sys.exit(0 if unit.startswith('only_esm') else 1)\n")
    for name, value in (("REMOTE_V3", str(v3)), ("REMOTE_CODE", str(code)), ("REMOTE_PY", sys.executable)):
        monkeypatch.setattr(b, name, value)
    return v3


def test_units_start_detached_and_the_probe_tells_running_from_ended(local_vm):
    remote = LocalRemote()
    planned = [{"unit": ESM4, "lane": 1, "fit_seconds": 1800.0}, {"unit": GVP4, "lane": 2, "fit_seconds": 3800.0}]
    started = time.monotonic()
    out = remote.run(c.launch_script("c-test", planned, session())).stdout
    assert "launched" in out and time.monotonic() - started < 3  # returned before the runners ended
    for _ in range(20):  # the detached shell records each member's PID; the probe then reads 'running'
        first = c.vm_state(remote)["launches"]
        if first[ESM4]["state"] == first[GVP4]["state"] == "running":
            break
        time.sleep(0.05)
    assert first[ESM4]["state"] == first[GVP4]["state"] == "running"
    assert (local_vm / "step_c" / "launch" / f"{ESM4}.pid").is_file()
    for _ in range(100):
        launches = c.vm_state(remote)["launches"]
        if all(record["state"] == "exited" for record in launches.values()):
            break
        time.sleep(0.2)
    assert launches[ESM4]["rc"] == "0" and launches[GVP4]["rc"] == "1" and launches[GVP4]["lane"] == 2
    # A second launch of the same name keeps the first attempt's files.
    remote.run(c.launch_script("c-test2", planned[:1], session()))
    assert (local_vm / "step_c" / "launch" / f"{ESM4}.rc.1").read_text().strip() == "0"
    # A launch recorded under another boot (VM stopped) is lost, never awaited.
    (local_vm / "step_c" / "launch" / f"{GVP4}.rc").unlink()
    (local_vm / "step_c" / "launch" / f"{GVP4}.boot").write_text("another-boot\n")
    assert c.vm_state(remote)["launches"][GVP4]["state"] == "lost"


# ---------------------------------------------------------------------------
# Recovery with 'wait' after a dropped connection (never relaunches)
# ---------------------------------------------------------------------------

def test_a_dropped_launch_connection_is_recovered_by_wait_without_relaunching(monkeypatch):
    pulled = []
    monkeypatch.setattr(b, "cmd_pull", lambda remote, lanes: (pulled.append(lanes),
                                                                {"lanes": {f"lane{lanes[0]}": {"status": "acknowledged"}},
                                                                 "ok": True})[1])
    dropped = FakeVM(vm(), launch_error=subprocess.TimeoutExpired("ssh", 600))
    with pytest.raises(c.LauncherError, match="never relaunch"):
        c.cmd_launch(dropped, session(), [(ESM4, 1)], sleep=lambda s: None)
    assert len(dropped.launched()) == 1  # one attempt, never retried
    running = vm(launches={ESM4: launch_record(ESM4, 1, "running")}, claims=[ESM4])
    ended = vm(launches={ESM4: launch_record(ESM4, 1, "exited")}, claims=[ESM4],
               lanes={"lane0": lane(), "lane1": lane(ack_uploaded=False), "lane2": lane()})
    ended["statuses"][ESM4] = {"status": "completed", "lane": 1, "gate_passed": None, "family": "only_esm",
                               "target": "four_class"}
    remote = FakeVM(running, running, ended, contact_errors=2)  # two failed polls, then running, then ended
    result = c.cmd_wait(remote, session(), sleep=lambda s: None)
    assert result["ended"][ESM4]["exit_code"] == "0" and result["ended"][ESM4]["status"] == "completed"
    assert pulled == [[1]] and not remote.launched()


def test_wait_gives_up_cleanly_and_never_waits_for_a_lost_launch(monkeypatch):
    monkeypatch.setattr(b, "cmd_pull", lambda remote, lanes: pytest.fail("nothing to pull"))
    remotes = [FakeVM(vm(), contact_errors=c.MAX_CONTACT_FAILURES)]
    with pytest.raises(c.LauncherError, match="Resume with 'wait'"):
        c.cmd_wait(remotes[-1], session(), names=[ESM4], sleep=lambda s: None)
    remotes.append(FakeVM(vm(), contact_errors=1))
    with pytest.raises(c.LauncherError, match="hard stop passed"):  # contact lost after the hard stop
        c.cmd_wait(remotes[-1], session(deadline=1.0), names=[ESM4], sleep=lambda s: None)
    remotes.append(FakeVM(vm(launches={ESM4: launch_record(ESM4, 1, "lost")})))
    assert c.cmd_wait(remotes[-1], session(), names=[ESM4], sleep=lambda s: None)["ended"][ESM4]["launch"] == "lost"
    remotes.append(FakeVM(vm(launches={ESM4: launch_record(ESM4, 1, "running")})))
    with pytest.raises(c.LauncherError, match="hard stop passed"):
        c.cmd_wait(remotes[-1], session(deadline=1.0), sleep=lambda s: None)
    remotes.append(FakeVM(vm()))
    with pytest.raises(c.LauncherError, match="No launch record"):
        c.cmd_wait(remotes[-1], session(), names=[ESM4], sleep=lambda s: None)
    assert not any(remote.launched() for remote in remotes)


def test_a_lost_record_in_a_busy_lane_still_counts_as_running():
    state = vm(launches={ESM4: launch_record(ESM4, 1, "lost")},
               lanes={"lane0": lane(), "lane2": lane(),
                      "lane1": lane(lock_free=False, pending_transfer=False,
                                    pending_unit={"run_name": ESM4, "status": "running", "persisted": False})})
    remote = FakeVM(state)
    assert c.vm_state(remote)["launches"][ESM4]["state"] == "running"


def test_wait_any_returns_after_the_first_unit_to_refill_its_lane(monkeypatch):
    monkeypatch.setattr(b, "cmd_pull", lambda remote, lanes: {"lanes": {f"lane{lanes[0]}": {"status": "acknowledged"}},
                                                               "ok": True})
    state = vm(launches={ESM4: launch_record(ESM4, 0, "exited"), GVP4: launch_record(GVP4, 1, "running")},
               lanes={"lane0": lane(ack_uploaded=False), "lane1": lane(), "lane2": lane()})
    remote = FakeVM(state)
    result = c.cmd_wait(remote, session(), names=[ESM4, GVP4], any_unit=True, sleep=lambda s: None)
    assert not remote.launched() and list(result["ended"]) == [ESM4] and result["still_running"] == [GVP4] and "lane0" in result["pulls"]


# ---------------------------------------------------------------------------
# Verified per-lane host pull: a failure blocks further work
# ---------------------------------------------------------------------------

def test_a_failed_pull_fails_wait_and_blocks_every_later_launch(monkeypatch):
    monkeypatch.setattr(b.host_pull, "pull", lambda *a, **k: {"ok": False, "lanes": {
        "lane1": {"status": "error", "error": "rsync failed"}}})
    blocked = vm(launches={ESM4: launch_record(ESM4, 1, "exited")},
                 lanes={"lane0": lane(), "lane1": lane(ack_uploaded=False), "lane2": lane()})
    waiting = FakeVM(blocked)
    with pytest.raises(c.LauncherError, match="rsync failed"):
        c.cmd_wait(waiting, session(), names=[ESM4], sleep=lambda s: None)
    assert not waiting.launched()
    remote = FakeVM(blocked)
    with pytest.raises(c.LauncherError, match="lane1: an ended unit awaits its verified host pull"):
        c.cmd_launch(remote, session(), [(GVP4, 0)], sleep=lambda s: None)  # even into another lane
    assert not remote.launched()
    with pytest.raises(c.LauncherError, match="awaits its verified host pull"):
        c.plan_regression(vm(regression=None, lanes=blocked["lanes"]))


def test_launch_waits_and_pulls_each_lane_by_default(monkeypatch):
    pulled = []
    monkeypatch.setattr(b, "cmd_pull", lambda remote, lanes: (pulled.append(lanes[0]),
                                                                {"lanes": {f"lane{lanes[0]}": {"status": "ok"}},
                                                                 "ok": True})[1])
    before = vm()
    after = copy.deepcopy(before)
    after["launches"] = {ESM4: launch_record(ESM4, 0, "exited"), GVP4: launch_record(GVP4, 1, "exited")}
    after["lanes"] = {"lane0": lane(ack_uploaded=False), "lane1": lane(ack_uploaded=False), "lane2": lane()}
    remote = FakeVM(before, after)
    result = c.cmd_launch(remote, session(), [(ESM4, 0), (GVP4, 1)], sleep=lambda s: None)
    assert len(remote.launched()) == 1 and sorted(pulled) == [0, 1] and set(result["ended"]) == {ESM4, GVP4}


# ---------------------------------------------------------------------------
# Evidence copy before vm-stop; archive and assessment helpers
# ---------------------------------------------------------------------------

def test_evidence_copy_is_checked_on_both_ends(local_vm, tmp_path):
    launch = local_vm / "step_c" / "launch"
    launch.mkdir(parents=True)
    (launch / f"{ESM4}.launch.json").write_text(json.dumps(launch_record(ESM4, 0, "exited")))
    (launch / f"{ESM4}.rc").write_text("0\n")
    (local_vm / "assessments" / "C_1").mkdir(parents=True)
    (local_vm / "assessments" / "C_1" / "assessment.json").write_text("{}")
    (local_vm / "lanes" / "lane0").mkdir()
    (local_vm / "lanes" / "lane0" / "execution_events.jsonl").write_text("{}\n")
    (local_vm / "step_b").mkdir()
    (local_vm / "step_b" / "host.jsonl").write_text("{}\n")
    (local_vm / "campaign_manifest.json").write_text("{}")
    out = c.cmd_evidence(LocalRemote(), destination=tmp_path / "evidence")
    manifest = json.loads((tmp_path / "evidence" / "evidence_manifest.json").read_text())
    assert out["files"] == len(manifest["files"]) and not manifest["mismatched"]
    for name in (f"campaign/step_c/launch/{ESM4}.rc", "campaign/assessments/C_1/assessment.json",
                 "campaign/lanes/lane0/execution_events.jsonl", "campaign/campaign_manifest.json"):
        assert name in manifest["files"]
    assert not any(name.startswith("campaign/step_b/") for name in manifest["files"])  # step B was copied already
    assert out["active_launches"] == [] and out["awaiting_pull"] == []


def test_archive_failed_refuses_the_regression_and_running_units():
    with pytest.raises(c.LauncherError, match="diagnosis"):
        c.cmd_archive_failed(FakeVM(vm()), c.REGRESSION_NAME)
    with pytest.raises(c.LauncherError, match="running launch"):
        c.cmd_archive_failed(FakeVM(vm(launches={ESM4: launch_record(ESM4, 0, "running")})), ESM4)
    remote = FakeVM(vm(statuses={**vm()["statuses"], ESM4: failed()}))
    assert c.cmd_archive_failed(remote, ESM4)["archived_to"].endswith("attempt1")
    assert "--action archive-failed" in remote.scripts[-1] and ESM4 in remote.scripts[-1]


def test_assessment_copy_is_checked(local_vm, tmp_path):
    (Path(b.REMOTE_CODE) / "pmm_v3_assessment.py").write_text(
        "import json, sys\nfrom pathlib import Path\nroot = Path(sys.argv[sys.argv.index('--campaign-dir') + 1])\n"
        "out = root / 'assessments' / 'C_20261006T000000Z'\nout.mkdir(parents=True)\n"
        "(out / 'assessment.json').write_text(json.dumps({'result': {'step': 'C'}}))\n"
        "print(json.dumps({'assessment': str(out)}))\n")
    result = c.cmd_assess(LocalRemote(), evidence_root=tmp_path / "copies")
    assert result["result"] == {"step": "C"} and Path(result["copy"]).is_file()


# ---------------------------------------------------------------------------
# Review fixes (2026-10-06): archive after pull, interrupted lanes, hints, evidence safety, pull retries
# ---------------------------------------------------------------------------

def test_archive_failed_is_refused_while_a_lane_awaits_its_pull():
    state = vm(statuses={**vm()["statuses"], ESM4: failed()},
               lanes={"lane0": lane(), "lane1": lane(ack_uploaded=False), "lane2": lane()})
    remote = FakeVM(state)
    with pytest.raises(c.LauncherError, match="await their host pull"):
        c.cmd_archive_failed(remote, ESM4)
    assert not any("--action archive-failed" in s for s in remote.scripts)
    assert "after its lane is pulled" in c.unit_blockers(ESM4, state)[0]


def interrupted_vm(tmp_path, monkeypatch):
    """A real lane whose worker admitted a unit and was killed (no exit, no status, lock released by the kernel)."""
    v3, host = tmp_path / "vm_v3", tmp_path / "host"
    (v3 / "lanes").mkdir(parents=True)
    (v3 / "execution_settings.json").write_text(json.dumps({"amp": False, "concurrent_lanes": 3, "reuse": {}}))
    for name, value in (("REMOTE_V3", str(v3)), ("REMOTE_CODE", str(ROOT)), ("REMOTE_PY", sys.executable),
                        ("DURABLE", host)):
        monkeypatch.setattr(b, name, value)
    worker = ("import importlib.util, os, sys\nfrom pathlib import Path\n"  # file-path load, as RECOVER_LANE does
              f"spec = importlib.util.spec_from_file_location('e', {str(ROOT / 'src/benchmarking/pmm_execution.py')!r})\n"
              "e = importlib.util.module_from_spec(spec); sys.modules['e'] = e; spec.loader.exec_module(e)\n"
              f"lane = Path({str(v3 / 'lanes' / 'lane1')!r})\n"
              f"ex = e.CampaignExecution(lane, durable_root=Path({str(host / 'lane1')!r}), session_id='s-1', "
              "persistence_mode='host_pull', policy=e.ExecutionPolicy(4e9, 4000))\n"
              f"ex.__enter__(); ex.admit({ESM4!r}, 10)\n"
              f"(lane / 'runs' / {ESM4!r}).mkdir(parents=True); (lane / 'runs' / {ESM4!r} / 'partial.txt').write_text('x')\n"
              "os._exit(9)\n")
    assert subprocess.run([sys.executable, "-c", worker]).returncode == 9
    return v3, host


def test_an_interrupted_lane_is_recovered_by_the_runner_logic_then_pulled(tmp_path, monkeypatch):
    v3, host = interrupted_vm(tmp_path, monkeypatch)
    remote = LocalRemote()
    state = c.vm_state(remote)
    assert c.interrupted_lanes(state) == {1: ESM4}
    with pytest.raises(c.LauncherError, match="recover-lane 1"):
        c.plan_units([(GVP4, 1)], {**state, "statuses": {**state["statuses"], **vm()["statuses"]}})
    with pytest.raises(c.LauncherError, match="recover-lane"):
        c.cmd_archive_failed(remote, ESM4)
    result = c.cmd_recover_lane(remote, session(), 1)
    assert result["recovery"]["recovered"] and result["pull"]["lanes"]["lane1"]["status"] == "acknowledged"
    assert (host / "lane1" / "runs" / ESM4 / "partial.txt").is_file()
    after = c.vm_state(remote)
    assert c.interrupted_lanes(after) == {} and c.awaiting_pull(after) == [] and not b.lane_blockers(after["lanes"])
    with pytest.raises(c.LauncherError, match="nothing to recover"):
        c.cmd_recover_lane(remote, session(), 1)


def test_hints_follow_the_vm_statuses():
    blocked = vm(regression=None, launches={c.REGRESSION_NAME: {**launch_record(c.REGRESSION_NAME, 0, "exited"),
                                                                  "rc": "3"}})
    ended = {c.REGRESSION_NAME: {"launch": "exited", "exit_code": "3", "status": None, "gate_passed": None}}
    assert "did not start" in c.next_hint(blocked, ended)
    assert c.next_hint(vm(regression="failed_regression_gate"), {}).startswith("STOP")
    with_failure = vm(statuses={**vm()["statuses"], ESM4: failed()})
    assert "failed units" in c.next_hint(with_failure, {})
    assert c.summary(with_failure)["failed_units"] == [ESM4]
    assert c.rc_meaning("137") == "killed by signal 9" and c.rc_meaning(None).startswith("no exit code")
    assert "STOP" in c.rc_meaning("1", regression=True)


def test_wait_after_a_drop_reports_units_that_already_ended(monkeypatch):
    monkeypatch.setattr(b, "cmd_pull", lambda remote, lanes: {"lanes": {f"lane{lanes[0]}": {"status": "ok"}}, "ok": True})
    state = vm(statuses={**vm()["statuses"], ESM4: failed()}, launches={ESM4: launch_record(ESM4, 1, "exited")},
               lanes={"lane0": lane(), "lane1": lane(ack_uploaded=False), "lane2": lane()})
    remote = FakeVM(state)
    result = c.cmd_wait(remote, session(), sleep=lambda s: None)
    assert not remote.launched()
    assert result["ended"] == {} and "lane1" in result["pulls"] and result["failed_units"] == [ESM4]
    assert "failed units" in result["next"]


def test_a_transient_pull_error_is_retried(monkeypatch):
    calls = []

    def flaky(remote, lanes):
        calls.append(lanes[0])
        if len(calls) == 1:
            raise b.LauncherError("Host pull failed for some lane: rsync failed")
        return {"lanes": {f"lane{lanes[0]}": {"status": "acknowledged"}}, "ok": True}

    monkeypatch.setattr(b, "cmd_pull", flaky)
    awaiting = vm(launches={ESM4: launch_record(ESM4, 1, "exited")},
                  lanes={"lane0": lane(), "lane1": lane(ack_uploaded=False), "lane2": lane()})
    pulled = vm(launches={ESM4: launch_record(ESM4, 1, "exited")})
    remote = FakeVM(awaiting, pulled)
    result = c.cmd_wait(remote, session(), names=[ESM4], sleep=lambda s: None)
    assert calls == [1] and result["ended"][ESM4]["exit_code"] == "0" and not remote.launched()


def test_evidence_is_copied_but_refuses_a_stop_while_units_run(local_vm, tmp_path):
    launch = local_vm / "step_c" / "launch"
    launch.mkdir(parents=True)
    (launch / f"{ESM4}.launch.json").write_text(json.dumps(launch_record(ESM4, 0, "running")))
    (launch / f"{ESM4}.boot").write_text(Path("/proc/sys/kernel/random/boot_id").read_text())
    (launch / f"{ESM4}.rc.tmp").write_text("0\n")  # transient file, never copied
    (local_vm / "campaign_manifest.json").write_text("{}")
    member = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", ESM4])  # a live member
    try:
        (launch / f"{ESM4}.pid").write_text(f"{member.pid}\n")
        assert c.vm_state(LocalRemote())["launches"][ESM4]["state"] == "running"
        with pytest.raises(c.LauncherError, match="NOT safe to vm-stop"):
            c.cmd_evidence(LocalRemote(), destination=tmp_path / "ev1")
        assert (tmp_path / "ev1" / "evidence_manifest.json").is_file()  # the copy was still made
        out = c.cmd_evidence(LocalRemote(), destination=tmp_path / "ev2", allow_running=True)
        assert out["safe_to_stop"] is False and out["active_launches"] == [ESM4]
        assert not any(name.endswith(".rc.tmp") for name in json.loads(
            (tmp_path / "ev2" / "evidence_manifest.json").read_text())["files"])
    finally:
        member.kill()
        member.wait()
    assert c.vm_state(LocalRemote())["launches"][ESM4]["state"] == "lost"  # dead member, no exit code


def test_a_launch_that_never_started_becomes_lost_after_the_grace(local_vm):
    launch = local_vm / "step_c" / "launch"
    launch.mkdir(parents=True)
    record = launch / f"{ESM4}.launch.json"
    record.write_text(json.dumps(launch_record(ESM4, 0, "starting")))
    (launch / f"{ESM4}.boot").write_text(Path("/proc/sys/kernel/random/boot_id").read_text())
    assert c.vm_state(LocalRemote())["launches"][ESM4]["state"] == "starting"
    old = time.time() - c.STARTING_GRACE_SECONDS - 5
    __import__("os").utime(record, (old, old))
    assert c.vm_state(LocalRemote())["launches"][ESM4]["state"] == "lost"
    assert c.unit_blockers(ESM4, {**vm(), "launches": c.vm_state(LocalRemote())["launches"], "claims": []}) == []


def test_a_batch_is_admitted_only_if_its_slowest_unit_fits():
    remote = FakeVM(vm())  # ESM4 warm (1,800 s) fits in 4,000 s; GVP4 cold (2,300 + 2,000 s) does not
    with pytest.raises(c.LauncherError, match="hard stop"):
        c.cmd_launch(remote, session(remaining=4000.0), [(ESM4, 0), (GVP4, 1)], wait=False, sleep=lambda s: None)
    assert not remote.launched()
    assert c.cmd_launch(FakeVM(vm()), session(remaining=4000.0), [(ESM4, 0)], wait=False)["launched"]


@pytest.mark.parametrize("lanes, reason", [
    ({"lane0": lane(), "lane1": lane(ack_uploaded=False), "lane2": lane()}, "awaiting_pull"),
    ({"lane0": lane(), "lane1": lane(), "lane2": lane(pending_transfer=False, pending_unit={
        "run_name": ESM4, "status": "running", "persisted": False})}, "interrupted_lanes"),
    ({"lane0": lane(lock_free=False), "lane1": lane(), "lane2": lane()}, "busy_lanes")])
def test_evidence_refuses_a_stop_for_every_unsafe_lane_state(monkeypatch, lanes, reason):
    monkeypatch.setattr(b, "cmd_evidence", lambda remote, **kw: {"evidence_dir": "copied", "files": 1})
    with pytest.raises(c.LauncherError, match=f"NOT safe to vm-stop.*{reason}"):
        c.cmd_evidence(FakeVM(vm(lanes=lanes)))
    assert c.cmd_evidence(FakeVM(vm()))["safe_to_stop"] is True


def test_evidence_copies_every_pattern_to_the_step_c_folder_and_rejects_a_corrupt_copy(local_vm, tmp_path,
                                                                                         monkeypatch):
    from pmm_v3_host_pull import LocalTransport

    samples = ["campaign_manifest.json", "execution_settings.json", "fold_class_weights.json", "claims/u.json",
               "step_c/launch/u.rc", "assessments/C_1/assessment.json", "lanes/lane0/execution_state.json",
               "lanes/lane0/execution_events.jsonl", "lanes/lane0/run_status_u.json", "lanes/lane0/commands/u.json",
               "lanes/lane0/persistence_receipts/t.ack.json", "failed_attempts/u/attempt1/archive_receipt.json"]
    assert len(samples) == len(c.EVIDENCE_PATTERNS)
    for name in samples:
        (local_vm / name).parent.mkdir(parents=True, exist_ok=True)
        content = {"run_name": "u", "status": "completed"} if "run_status" in name else {}
        (local_vm / name).write_text(json.dumps(content))
    monkeypatch.setattr(c, "EVIDENCE", tmp_path / "step_c_evidence")
    out = c.cmd_evidence(LocalRemote())
    assert Path(out["evidence_dir"]).parent == tmp_path / "step_c_evidence"
    copied = json.loads((Path(out["evidence_dir"]) / "evidence_manifest.json").read_text())["files"]
    assert sorted(copied) == sorted("campaign/" + name for name in samples)

    class Corrupting(LocalTransport):
        def fetch(self, remote_root, members, local_root):
            super().fetch(remote_root, members, local_root)
            (Path(local_root) / members[0]).write_text("corrupt")

    corrupt = LocalRemote()
    corrupt.transport = Corrupting()
    with pytest.raises(c.LauncherError, match="Evidence copies differ"):
        c.cmd_evidence(corrupt, destination=tmp_path / "bad")
    assert json.loads((tmp_path / "bad" / "evidence_manifest.json").read_text())["mismatched"]


def test_a_corrupt_assessment_copy_is_rejected(local_vm, tmp_path):
    (Path(b.REMOTE_CODE) / "pmm_v3_assessment.py").write_text(
        "import json, sys\nfrom pathlib import Path\nroot = Path(sys.argv[sys.argv.index('--campaign-dir') + 1])\n"
        "out = root / 'assessments' / 'C_x'\nout.mkdir(parents=True)\n"
        "(out / 'assessment.json').write_text(json.dumps({'result': {}}))\nprint(json.dumps({'assessment': str(out)}))\n")
    remote = LocalRemote()
    remote.transport.read_bytes = lambda path: b'{"result": {"tampered": true}}'
    with pytest.raises(c.LauncherError, match="differs from the VM file"):
        c.cmd_assess(remote, evidence_root=tmp_path / "copies")


def test_archive_and_recovery_refusals():
    with pytest.raises(c.LauncherError, match="not a step C unit"):
        c.cmd_archive_failed(FakeVM(vm()), "only_gvp__four_class__meanagg__fold0__seed42")
    stranded = vm(lanes={"lane0": lane(), "lane1": lane(), "lane2": lane(pending_transfer=False, pending_unit={
        "run_name": ESM4, "status": "running", "persisted": False})})

    class NotRecovered(FakeVM):
        def run(self, script, *, check=True, timeout=None):
            if "CampaignExecution" in script:
                return subprocess.CompletedProcess([], 0, stdout='{"recovered": false}', stderr="")
            return super().run(script, check=check, timeout=timeout)

    with pytest.raises(c.LauncherError, match="did not record"):
        c.cmd_recover_lane(NotRecovered(stranded), session(), 2)


def test_the_command_line_reads_the_session_first_and_wires_its_flags(monkeypatch):
    def expired():
        raise b.LauncherError("The session has not started or its hard stop has passed")

    monkeypatch.setattr(b, "read_session", expired)
    monkeypatch.setattr(b, "check_ssh_config", lambda: pytest.fail("no VM contact without a valid session"))
    with pytest.raises(b.LauncherError, match="hard stop"):
        c.main(["launch", f"{ESM4}@0", "--no-wait"])
    calls = {}
    monkeypatch.setattr(b, "read_session", lambda: session())
    monkeypatch.setattr(b, "check_ssh_config", lambda: Path("cfg"))
    monkeypatch.setattr(b, "Remote", lambda cfg: "remote")
    monkeypatch.setattr(c, "cmd_launch", lambda remote, s, requests, **kw: calls.setdefault("launch", (requests, kw)))
    monkeypatch.setattr(c, "cmd_wait", lambda remote, s, **kw: calls.setdefault("wait", kw))
    monkeypatch.setattr(c, "cmd_recover_lane", lambda remote, s, k: calls.setdefault("recover", k))
    monkeypatch.setattr(c, "cmd_evidence", lambda remote, **kw: calls.setdefault("evidence", kw))
    c.main(["launch", f"{ESM4}@0", f"{GVP4}@2", "--no-wait", "--fit-seconds", "1234"])
    c.main(["wait", ESM4, "--any"])
    c.main(["recover-lane", "2"])
    c.main(["evidence", "--allow-running"])
    assert calls["launch"] == ([(ESM4, 0), (GVP4, 2)], {"fit_seconds": 1234.0, "wait": False})
    assert calls["wait"] == {"names": [ESM4], "any_unit": True} and calls["recover"] == 2
    assert calls["evidence"] == {"allow_running": True}
