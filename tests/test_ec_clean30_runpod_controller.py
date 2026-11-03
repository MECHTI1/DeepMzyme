"""A real detached watchdog drives a local fake CLI; no cloud or GPU is used."""
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import signal
import shutil
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from ec_clean30 import runpod_controller as rc  # noqa: E402


FAKE = r'''#!/usr/bin/env python3
import datetime,fcntl,json,pathlib,sys
p=pathlib.Path(__file__).with_suffix('.json')
with p.with_suffix('.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 state=json.loads(p.read_text());args=sys.argv[1:];assert args.pop(0)=='pod';action=args.pop(0)
 state.setdefault('calls',[]).append(action)
 error=None
 if action=='list': out=state.get('pods',[])
 elif action=='create':
  state['create_calls']=state.get('create_calls',0)+1
  options=dict(zip(args[::2],args[1::2])); now=state.get('created_at',datetime.datetime.now(datetime.timezone.utc).isoformat())
  pod={'id':'ownedpod123','name':options['--name'],'createdAt':now,'desiredStatus':'RUNNING','runtimeStatus':'running',
       'gpuCount':1,'gpuId':state.get('actual_gpu',options['--gpu-id']),'imageName':options['--image'],
       'machine':{'dataCenterId':state.get('actual_dc',options['--data-center-ids']),'secureCloud':True},
       'volumeInGb':int(options['--volume-in-gb']),'volumeMountPath':options['--volume-mount-path'],
       'costPerHr':state.get('actual_rate',.89),'ssh':{'ip':'127.0.0.1','port':2222},
       'env':{'RUNPOD_API_KEY':'rpa_should_never_enter_receipts',
              'NVIDIA_TF32_OVERRIDE':state.get('actual_tf32',json.loads(options['--env'])['NVIDIA_TF32_OVERRIDE'])}}
  state.setdefault('pods',[]).append(pod);out=state['pods'][0] if state.get('foreign_create_response') else pod
  if state.get('lose_create_response'):error={'code':'network_error','error':'response lost rpa_fake_secret'}
 elif action=='get':
  out=next(x for x in state['pods'] if x['id']==args[0])
 elif action=='stop':
  state['stop_calls']=state.get('stop_calls',0)+1
  state.setdefault('stopped_ids',[]).append(args[0])
  out=next(x for x in state['pods'] if x['id']==args[0])
  if state.get('transient_stops',0)>0:
   state['transient_stops']-=1;error={'code':'network_error','error':'transient stop failure'}
  else:out.update(desiredStatus='EXITED',runtimeStatus='stopped');out={'message':'stop requested'}
 elif action=='delete':
  state.setdefault('deleted_ids',[]).append(args[0]);state['pods']=[x for x in state['pods'] if x['id']!=args[0]]
  out={'message':'deleted'}
 else:raise AssertionError(action)
 p.write_text(json.dumps(state))
 if error:print(json.dumps(error),file=sys.stderr);sys.exit(1)
 print(json.dumps(out))
'''


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    cli = tmp_path / "fake_runpodctl"
    # Use the verified interpreter, without relying on the shell's Python.
    cli.write_text(FAKE.replace("#!/usr/bin/env python3", f"#!{sys.executable}"))
    cli.chmod(0o700)
    api = cli.with_suffix(".json")
    api.write_text(json.dumps({"pods": []}))
    credential = tmp_path / "credentials.json"
    credential.write_text(json.dumps({"api_key": "rpa_fake_secret"}))
    credential.chmod(0o600)
    key = tmp_path / "ssh_key"
    key.write_text("synthetic key, never used for SSH")
    key.chmod(0o600)
    config = {"provider": "runpod", "hardware_profile": "runpod_rtx4090", "state_root": str(tmp_path / "state"),
              "runpod": {"cli_path": str(cli), "credential_file": str(credential), "pod_name": "g2-unique-fixture",
                         "gpu_id": "NVIDIA GeForce RTX 4090", "data_center_id": "EUR-IS-2",
                         "image": "example/image@sha256:" + "a" * 64, "min_cuda_version": "13.0",
                         "container_disk_gb": 50, "volume_gb": 20, "volume_mount_path": "/workspace",
                         "ssh_alias": "runpod-fixture", "ssh_private_key": str(key), "requested_hours": 2,
                         "max_runtime_seconds": 7020, "gross_ceiling_usd": 5, "stop_reserve_seconds": 120,
                         "runtime_environment": {"NVIDIA_TF32_OVERRIDE": "0"},
                         "retention_policy": {"retention_seconds": 86400, "cleanup_reviewed": True,
                                              "action": "verified_backup_then_delete", "owner": "fixture owner",
                                              "evidence": "synthetic approved cleanup policy"},
                         "quote": {"observed_at": datetime.now(timezone.utc).isoformat(), "source": "synthetic quote",
                                   "max_age_seconds": 900, "hourly_compute_usd": .89, "running_storage_hourly_usd": .01,
                                   "stopped_storage_reserve_usd": .25, "stopped_storage_hourly_usd": .01,
                                   "egress_reserve_usd": .25,
                                   "prior_charge_usd": .6, "margin_fraction": .15}}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    controllers = []

    def make(**api_values):
        api.write_text(json.dumps({"pods": [], **api_values}))
        path.write_text(json.dumps(config))
        controller = rc.RunpodController(config, path)
        controllers.append(controller)
        return controller

    yield config, path, api, credential, make
    for controller in controllers:
        if controller.state_path.exists():
            state = json.loads(controller.state_path.read_text())
            if state.get("pod_id") and state.get("phase") not in {"stopped", "deleted"}:
                controller.stop()
            pid = state.get("watchdog_pid")
            if pid:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass


def test_cost_admission_accounts_for_prior_storage_egress_and_margin(fixture):
    config, _, _, _, _ = fixture
    result = rc.validate_config(config)
    assert result["estimated_gross_usd"] == pytest.approx(.6 + (2 * .9 + .25 + .25) * 1.15)
    assert result["provider_enforced_stop"] is False
    config["runpod"]["quote"]["prior_charge_usd"] = 4
    with pytest.raises(rc.RunpodError, match="gross envelope"):
        rc.validate_config(config)


def test_continuous_heartbeat_covers_a_simulated_long_blocking_transfer(fixture, monkeypatch):
    _, _, api, _, make = fixture
    controller = make()
    session = controller.start()
    started = time.time()
    elapsed = [0]
    advanced = threading.Event()
    real_event = threading.Event
    real_thread = threading.Thread

    class AcceleratedEvent:
        def __init__(self): self.event = real_event()
        def is_set(self): return self.event.is_set()
        def set(self): return self.event.set()
        def wait(self, interval):
            if self.event.wait(.002): return True
            elapsed[0] = min(1050, elapsed[0] + interval)
            if elapsed[0] == 1050: advanced.set()
            return False

    monkeypatch.setattr(rc, 'threading', SimpleNamespace(Event=AcceleratedEvent, Thread=real_thread))
    monkeypatch.setattr(rc, 'time', SimpleNamespace(time=lambda: started + elapsed[0],
                                                   monotonic=time.monotonic, sleep=time.sleep))
    with controller.maintain_heartbeat(interval=30) as heartbeat:
        # No foreground transport call runs during this simulated 1050-second block.
        assert advanced.wait(3), 'The background ticker did not progress while the foreground waited.'
        beat = json.loads(controller.heartbeat_path.read_text())
        assert elapsed[0] > 900 and 0 <= started + elapsed[0] - beat['updated_unix'] <= 30
        assert beat['pid'] == os.getpid() and beat['session_id'] == session['session_id']
        heartbeat.check()
        state = controller._state()
        assert all(state[key] == session[key] for key in ('allocation_started_unix', 'deadline_unix', 'hard_stop_unix'))
        with pytest.raises(rc.RunpodError, match='before verified shutdown'):
            heartbeat.close()
    assert not heartbeat.thread.is_alive()
    assert json.loads(api.read_text())['pods'][0]['runtimeStatus'] == 'stopped'


def test_continuous_heartbeat_failure_closes_admission_and_still_stops(fixture, monkeypatch):
    _, _, api, _, make = fixture
    controller = make()
    controller.start()
    original = rc._write
    writes = [0]

    def broken_owner_disk(path, value):
        if path == controller.heartbeat_path:
            writes[0] += 1
            if writes[0] > 1: raise OSError('owner disk failed rpa_private_credential')
        original(path, value)

    monkeypatch.setattr(rc, '_write', broken_owner_disk)
    with pytest.raises(rc.RunpodError, match='Continuous owner heartbeat failed'):
        with controller.maintain_heartbeat(interval=.005) as heartbeat:
            assert heartbeat.finished.wait(2)
            assert controller._state()['stop_requested']
            before = list(json.loads(api.read_text())['calls'])
            with pytest.raises(rc.RunpodError, match='Continuous owner heartbeat failed'):
                controller.session()
            assert json.loads(api.read_text())['calls'] == before
            heartbeat.check()
    assert controller._state()['phase'] == 'stopped'
    assert not heartbeat.thread.is_alive()
    assert 'rpa_private_credential' not in controller.state_path.read_text()


def test_heartbeat_context_refuses_another_coordinator_process(fixture, monkeypatch):
    _, _, api, _, make = fixture
    controller = make()
    controller.start()
    with controller.maintain_heartbeat(interval=30) as heartbeat:
        original = rc.os.getpid
        monkeypatch.setattr(rc.os, 'getpid', lambda: heartbeat.pid + 100)
        try:
            with pytest.raises(rc.RunpodError, match='another coordinator process'):
                controller.heartbeat()
        finally:
            monkeypatch.setattr(rc.os, 'getpid', original)
    assert json.loads(api.read_text())['pods'][0]['runtimeStatus'] == 'stopped'


def test_heartbeat_context_keeps_ticker_when_stop_cannot_be_verified(fixture, monkeypatch):
    _, _, _, _, make = fixture
    controller = make()
    controller.start()
    stop = controller.stop
    monkeypatch.setattr(controller, 'stop', lambda: (_ for _ in ()).throw(rc.RunpodError('STOP unavailable')))
    try:
        with pytest.raises(rc.RunpodError, match='STOP unavailable'):
            with controller.maintain_heartbeat(interval=.01) as heartbeat:
                pass
        assert heartbeat.thread.is_alive() and not heartbeat.finished.is_set()
    finally:
        monkeypatch.setattr(controller, 'stop', stop)
        controller.stop()
        heartbeat.close()


def test_independent_watchdog_stops_after_losing_continuous_owner(fixture, monkeypatch):
    _, path, _, _, make = fixture
    controller = make()
    state = {'phase': 'running', 'session_id': 'lost-owner', 'deadline_unix': 9000}
    controller._cached_state = state
    controller.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
    controller.heartbeat_path.write_text(json.dumps({'session_id': 'lost-owner', 'updated_unix': 0, 'pid': 123}))
    snapshot = controller.root / 'watchdog-fixture.json'
    snapshot.write_text(json.dumps({'config': controller.config, 'config_path': str(path)}))
    stopped = []
    monkeypatch.setattr(controller, '_cleanup_state', lambda: dict(state))
    monkeypatch.setattr(controller, 'stop', lambda: stopped.append('owned-stop'))
    monkeypatch.setattr(rc, 'RunpodController', lambda *args, **kwargs: controller)
    monkeypatch.setattr(rc, 'time', SimpleNamespace(time=lambda: 1001))
    rc._watchdog(snapshot)
    assert stopped == ['owned-stop'] and state['deadline_unix'] == 9000


@pytest.mark.parametrize("field,value,match", [("requested_hours", 2.1, "runtime"),
    ("max_runtime_seconds", 7200, "runtime"), ("gross_ceiling_usd", 5.01, "ceiling"),
    ("image", "example/mutable:latest", "digest"), ("gpu_id", "NVIDIA H100", "hardware profile"),
    ("volume_gb", 0, "volume_gb"), ("stop_reserve_seconds", 60, "shutdown-request reserve"),
    ("runtime_environment", {"NVIDIA_TF32_OVERRIDE": "1"}, "full-FP32"),
    ("runtime_environment", {"NVIDIA_TF32_OVERRIDE": "0", "UNREVIEWED": "1"}, "full-FP32")])
def test_config_refuses_widening(fixture, field, value, match):
    config, _, _, _, _ = fixture
    config["runpod"][field] = value
    with pytest.raises(rc.RunpodError, match=match):
        rc.validate_config(config)


def test_stale_quote_and_unsafe_credentials_never_create(fixture):
    config, _, api, credential, make = fixture
    config["runpod"]["quote"]["observed_at"] = "2000-01-01T00:00:00Z"
    controller = make()
    with pytest.raises(rc.RunpodError, match="stale"):
        controller.start()
    config["runpod"]["quote"]["observed_at"] = datetime.now(timezone.utc).isoformat()
    credential.chmod(0o644)
    controller = make()
    with pytest.raises(rc.RunpodError, match="0600"):
        controller.start()
    assert not json.loads(api.read_text()).get("create_calls")


def test_inline_credentials_are_rejected_before_any_receipt(fixture):
    config, _, _, _, _ = fixture
    config["runpod"]["api_key"] = "rpa_do_not_persist_this"
    with pytest.raises(rc.RunpodError, match="Credentials belong"):
        rc.validate_config(config)


def test_storage_retention_requires_review_and_full_reserve(fixture):
    config, _, _, _, _ = fixture
    config["runpod"]["retention_policy"]["cleanup_reviewed"] = False
    with pytest.raises(rc.RunpodError, match="reviewed finite"):
        rc.validate_config(config)
    config["runpod"]["retention_policy"]["cleanup_reviewed"] = True
    config["runpod"]["quote"]["stopped_storage_reserve_usd"] = .01
    with pytest.raises(rc.RunpodError, match="retention period"):
        rc.validate_config(config)


def test_one_create_even_when_response_is_lost_and_receipts_redact_env(fixture):
    _, _, api, _, make = fixture
    controller = make(lose_create_response=True)
    session = controller.start()
    assert controller.session(session["session_id"])["pod_id"] == "ownedpod123"
    assert controller.write_ssh_config().is_file()
    with pytest.raises(rc.RunpodError, match="already has a launch intent"):
        controller.start()
    assert json.loads(api.read_text())["create_calls"] == 1
    controller.stop()
    receipt_text = "\n".join(p.read_text() for p in controller.root.glob("*.json"))
    assert "rpa_fake_secret" not in receipt_text
    assert "rpa_should_never_enter_receipts" not in receipt_text
    assert json.loads(controller.state_path.read_text())["stop_receipt"]["runtime_status"] == "stopped"


@pytest.mark.parametrize("absolute_seconds,lost_response", [(2000, False), (2000, True), (10000, False)])
def test_fresh_absolute_window_is_bound_before_watchdog_and_survives_recovery(fixture, monkeypatch, absolute_seconds, lost_response):
    config, path, api, _, make = fixture
    absolute = time.time() + absolute_seconds
    config["runpod"]["absolute_hard_stop_unix"] = absolute
    controller = make(lose_create_response=lost_response)
    original_arm = controller._arm_watchdog
    observed = {}
    offset = [0]
    real_time = time.time
    monkeypatch.setattr(rc, "time", SimpleNamespace(time=lambda: real_time() + offset[0],
                                                   monotonic=time.monotonic, sleep=time.sleep))

    def arm_then_advance():
        observed.update(controller._state())
        original_arm()
        offset[0] = 3

    monkeypatch.setattr(controller, "_arm_watchdog", arm_then_advance)
    session = controller.start()
    for key in ("allocation_started_unix", "hard_stop_unix", "deadline_unix", "max_total_seconds"):
        assert session[key] == observed[key]
    assert session["hard_stop_unix"] == min(absolute, observed["allocation_started_unix"] + 7020)
    assert session["hard_stop_unix"] - session["deadline_unix"] == 120
    assert session["max_total_seconds"] == session["hard_stop_unix"] - session["allocation_started_unix"]
    assert session["cost_admission"]["max_total_seconds"] == 7020
    recovered = rc.RunpodController(config, path).session(session)
    assert recovered["deadline_unix"] == session["deadline_unix"]
    assert json.loads(api.read_text())["create_calls"] == 1


@pytest.mark.parametrize("seconds", [-1, 120, 1019])
def test_fresh_absolute_deadline_without_closeout_reserve_refuses_before_intent(fixture, seconds):
    config, _, api, _, make = fixture
    config["runpod"]["absolute_hard_stop_unix"] = time.time() + seconds
    controller = make()
    with pytest.raises(rc.RunpodError, match="900-second closeout reserve"):
        controller.start()
    assert not controller.state_path.exists() and not controller.snapshot_path.exists()
    assert not json.loads(api.read_text()).get("create_calls")


def test_fresh_absolute_reserve_is_rechecked_immediately_before_post(fixture, monkeypatch):
    config, _, api, _, make = fixture
    config["runpod"]["absolute_hard_stop_unix"] = time.time() + 1030
    controller = make()
    original_call = controller._call
    real_time = time.time

    def advance_at_post(*args, **kwargs):
        if args[0] == "create":
            monkeypatch.setattr(rc, "time", SimpleNamespace(time=lambda: real_time() + 20,
                                                           monotonic=time.monotonic, sleep=time.sleep))
        return original_call(*args, **kwargs)

    monkeypatch.setattr(controller, "_call", advance_at_post)
    with pytest.raises(rc.RunpodError, match="900-second closeout reserve"):
        controller.start()
    assert not json.loads(api.read_text()).get("create_calls")
    state = controller._state()
    assert state["create_cancelled_before_post"] and state["phase"] == "stopped"


def test_fresh_bind_earlier_provider_creation_shortens_the_intent_window(fixture):
    _, _, _, _, make = fixture
    created_at = datetime.fromtimestamp(time.time() - 3, timezone.utc).isoformat()
    created = rc._epoch(created_at)
    controller = make(created_at=created_at)
    session = controller.start()
    assert session["allocation_started_unix"] == created
    assert session["hard_stop_unix"] == created + 7020
    assert session["deadline_unix"] == created + 7020 - 120
    assert session["max_total_seconds"] == 7020


@pytest.mark.parametrize("api_change,match", [({"actual_rate": 2.0}, "price"),
                                            ({"actual_gpu": "NVIDIA H100"}, "GPU type"),
                                            ({"actual_dc": "other-region"}, "data center"),
                                            ({"actual_tf32": "1"}, "runtime environment")])
def test_invalid_allocation_is_owned_before_validation_and_stopped(fixture, api_change, match):
    _, _, api, _, make = fixture
    controller = make(**api_change)
    with pytest.raises(rc.RunpodError, match=match):
        controller.start()
    state = json.loads(controller.state_path.read_text())
    assert state["pod_id"] == "ownedpod123" and state["phase"] == "stopped"
    assert json.loads(api.read_text())["pods"][0]["runtimeStatus"] == "stopped"


def test_other_active_gpu_blocks_before_watchdog_and_create(fixture):
    _, _, api, _, make = fixture
    controller = make(pods=[{"id": "other", "name": "other", "gpuCount": 1,
                             "desiredStatus": "RUNNING", "runtimeStatus": "running"}])
    with pytest.raises(rc.RunpodError, match="Another Runpod GPU"):
        controller.start()
    assert not controller.state_path.exists()
    assert not json.loads(api.read_text()).get("create_calls")


def adoption_fixture(config, *, ready=True, started=None):
    started = time.time() - 60 if started is None else started
    stamp = datetime.fromtimestamp(started, timezone.utc).isoformat()
    config["runpod"].update(adopt_pod_id="handedpod123", adopt_started_at=stamp,
                           absolute_hard_stop_unix=started + 6000)
    cfg = config["runpod"]
    return {"id": cfg["adopt_pod_id"], "name": cfg["pod_name"], "createdAt": stamp, "startedAt": stamp,
            "desiredStatus": "RUNNING", "runtimeStatus": "running", "gpuCount": 1, "gpuId": cfg["gpu_id"],
            "machine": {"dataCenterId": cfg["data_center_id"], "secureCloud": True}, "costPerHr": .89,
            "imageName": cfg["image"] if ready else "example/unconfigured:tag",
            "volumeInGb": cfg["volume_gb"] if ready else 0, "volumeMountPath": "/workspace",
            "env": cfg["runtime_environment"] if ready else {}, "ssh": {"ip": "127.0.0.1", "port": 2222}}


def test_named_adoption_arms_guard_without_creating_or_restarting(fixture):
    config, _, api, _, make = fixture
    pod = adoption_fixture(config, ready=False)
    controller = make(pods=[pod])
    result = controller.adopt("handedpod123")
    assert result["pending_qualification"] and result["pod_id"] == "handedpod123"
    assert result["allocation_started_unix"] == rc._epoch(pod["startedAt"])
    assert result["hard_stop_unix"] == config["runpod"]["absolute_hard_stop_unix"]
    assert result["deadline_unix"] == result["hard_stop_unix"] - 120
    assert not json.loads(api.read_text()).get("create_calls")
    with pytest.raises(rc.RunpodError, match="image differs"):
        controller.session()
    assert controller.stop()["pod_id"] == "handedpod123"


def test_adoption_configuration_cannot_create_even_at_low_level(fixture):
    config, _, api, _, make = fixture
    pod = adoption_fixture(config)
    controller = make(pods=[pod])
    with pytest.raises(rc.RunpodError, match="can never create"):
        controller.start()
    with pytest.raises(rc.RunpodError, match="can never create"):
        controller._call("create")
    assert not json.loads(api.read_text()).get("create_calls") and not controller.state_path.exists()


def test_adoption_recovery_never_resets_original_clock(fixture):
    config, _, api, _, make = fixture
    pod = adoption_fixture(config)
    controller = make(pods=[pod])
    first = controller.adopt("handedpod123", qualify=True)
    later = json.loads(api.read_text())
    later["pods"][0]["startedAt"] = datetime.now(timezone.utc).isoformat()
    api.write_text(json.dumps(later))
    recovered = controller.adopt("handedpod123", qualify=True)
    for key in ("session_id", "allocation_started_unix", "deadline_unix", "hard_stop_unix"):
        assert recovered[key] == first[key]


@pytest.mark.parametrize("wrong", ["argument", "name", "start", "cloud"])
def test_foreign_adoption_never_binds_or_stops_a_resource(fixture, wrong):
    config, _, api, _, make = fixture
    pod = adoption_fixture(config)
    argument = "handedpod123"
    if wrong == "argument":
        argument = "not-authorized"
    elif wrong == "name":
        pod["name"] = "other"
    elif wrong == "start":
        pod["startedAt"] = "2000-01-01T00:00:00Z"
    else:
        pod["machine"]["secureCloud"] = False
    controller = make(pods=[pod])
    with pytest.raises(rc.RunpodError):
        controller.adopt(argument)
    assert not controller.owner_path.exists()
    assert not json.loads(api.read_text()).get("stop_calls")


def test_adoption_qualification_failure_stops_only_the_handed_over_pod(fixture):
    config, _, api, _, make = fixture
    pod = adoption_fixture(config, ready=False)
    controller = make(pods=[pod])
    with pytest.raises(rc.RunpodError, match="image differs"):
        controller.adopt("handedpod123", qualify=True)
    state = json.loads(controller.state_path.read_text())
    assert state["adopted"] and state["phase"] == "stopped"
    assert json.loads(api.read_text())["stopped_ids"] == ["handedpod123"]


def test_adoption_accepts_missing_location_and_conservative_created_fallback(fixture):
    config, _, api, _, make = fixture
    pod = adoption_fixture(config)
    config["runpod"].update(data_center_id="ANY_SECURE", precision_environment_origin="transport")
    start = rc._epoch(config["runpod"]["adopt_started_at"])
    pod.pop("startedAt")
    pod["createdAt"] = datetime.fromtimestamp(start + .004, timezone.utc).isoformat()
    pod["machine"]["dataCenterId"] = ""
    pod["env"] = {}
    controller = make(pods=[pod])
    result = controller.adopt("handedpod123", qualify=True)
    assert result["allocation_started_unix"] == start
    state = json.loads(controller.state_path.read_text())
    assert state["pod"]["placement"]["data_center_id"] is None
    assert state["precision_environment_origin"] == "transport"
    assert not json.loads(api.read_text()).get("create_calls")


def test_adoption_absolute_limit_cannot_extend_provider_start_window(fixture):
    config, _, _, _, _ = fixture
    adoption_fixture(config)
    config["runpod"]["absolute_hard_stop_unix"] = rc._epoch(config["runpod"]["adopt_started_at"]) + 7021
    with pytest.raises(rc.RunpodError, match="reset or extend"):
        rc.validate_config(config)


def test_adoption_price_failure_is_bound_and_stopped(fixture):
    config, _, api, _, make = fixture
    pod = adoption_fixture(config)
    pod["costPerHr"] = 2
    controller = make(pods=[pod])
    with pytest.raises(rc.RunpodError, match="price exceeds"):
        controller.adopt("handedpod123")
    assert json.loads(controller.state_path.read_text())["phase"] == "stopped"
    assert json.loads(api.read_text())["stopped_ids"] == ["handedpod123"]


def test_config_drift_closes_admission_but_does_not_block_owned_stop(fixture):
    config, path, _, _, make = fixture
    controller = make()
    controller.start()
    config["runpod"]["pod_name"] = "changed"
    path.write_text(json.dumps(config))
    with pytest.raises(rc.RunpodError, match="configuration changed"):
        controller.session()
    assert controller.stop()["status"] == "EXITED"


def test_partial_owner_receipt_still_allows_exact_owned_stop(fixture, monkeypatch):
    _, _, api, _, make = fixture
    controller = make()
    original_write = rc._write
    failed = False

    def fail_second_binding_write(path, value):
        nonlocal failed
        if path == controller.state_path and value.get("phase") == "allocated" and not failed:
            failed = True
            raise OSError("synthetic disk failure after immutable ownership write")
        original_write(path, value)

    monkeypatch.setattr(rc, "_write", fail_second_binding_write)
    with pytest.raises(OSError, match="disk failure"):
        controller.start()
    assert failed and controller._owned_id() == "ownedpod123"
    assert json.loads(api.read_text())["pods"][0]["runtimeStatus"] == "stopped"


def test_foreign_create_response_is_never_bound_or_stopped(fixture):
    _, _, api, _, make = fixture
    foreign = {"id": "foreignpod", "name": "existing-stopped", "gpuCount": 0,
               "desiredStatus": "EXITED", "runtimeStatus": "stopped", "createdAt": "2000-01-01T00:00:00Z"}
    controller = make(pods=[foreign], foreign_create_response=True)
    assert controller.start()["pod_id"] == "ownedpod123"
    controller.stop()
    state = json.loads(api.read_text())
    assert state["create_calls"] == 1
    assert set(state["stopped_ids"]) == {"ownedpod123"}


def test_stale_precreate_watchdog_cannot_be_resurrected(fixture, monkeypatch):
    _, _, api, _, make = fixture
    controller = make()
    original_arm = controller._arm_watchdog

    def arm_then_close():
        original_arm()
        controller.stop()

    monkeypatch.setattr(controller, "_arm_watchdog", arm_then_close)
    with pytest.raises(rc.RunpodError, match="intent is closed"):
        controller.start()
    assert not json.loads(api.read_text()).get("create_calls")


@pytest.mark.parametrize("boundary", ["closing", "stopped", "deleted", "deadline"])
def test_session_cannot_admit_a_stale_provider_response_after_close(fixture, monkeypatch, boundary):
    _, _, api, _, make = fixture
    controller = make()
    session = controller.start()
    verified_at = controller._state()["provider_verified_at"]
    original_pods = controller._pods

    def read_then_close():
        # The provider response was valid when read, but shutdown wins before
        # session() publishes its result, as with a concurrent stop watchdog.
        rows = original_pods()
        if boundary in {"stopped", "deleted"}:
            controller.stop()
            if boundary == "deleted":
                controller._cleanup_update(phase="deleted")
        elif boundary == "closing":
            controller._cleanup_update(phase="closing_after_failure", stop_requested=True)
        else:
            controller._cleanup_update(deadline_unix=time.time() - 1)
        return rows

    monkeypatch.setattr(controller, "_pods", read_then_close)
    try:
        with pytest.raises(rc.RunpodError, match="admission closed during provider verification"):
            controller.session(session["session_id"])
        state = controller._state()
        assert state["provider_verified_at"] == verified_at
        if boundary != "deadline":
            assert state["stop_requested"] and state["phase"] != "running"
        else:
            assert state["deadline_unix"] < time.time()
    finally:
        controller.stop()
    assert json.loads(api.read_text())["pods"][0]["runtimeStatus"] == "stopped"


@pytest.mark.parametrize("problem", ["closing", "deadline", "watchdog"])
def test_final_prepost_check_blocks_late_or_unprotected_create(fixture, monkeypatch, problem):
    _, _, api, _, make = fixture
    controller = make()
    original_call = controller._call

    def delayed_call(*args, **kwargs):
        if args[0] == "create":
            if problem == "closing":
                controller._update(stop_requested=True)
            elif problem == "deadline":
                controller._update(deadline_unix=time.time() - 1)
            else:
                controller.watch_path.unlink()
        return original_call(*args, **kwargs)

    monkeypatch.setattr(controller, "_call", delayed_call)
    with pytest.raises(rc.RunpodError, match="admission closed|watchdog"):
        controller.start()
    assert not json.loads(api.read_text()).get("create_calls")
    state = controller._state()
    assert state["create_cancelled_before_post"] and state["phase"] == "stopped"


def test_full_receipt_disk_does_not_prevent_provider_stop(fixture, monkeypatch):
    _, _, api, _, make = fixture
    controller = make()
    controller.start()

    def fail_write(*_):
        raise OSError("synthetic full receipt disk")

    monkeypatch.setattr(rc, "_write", fail_write)
    assert controller.stop()["runtime_status"] == "stopped"
    assert json.loads(api.read_text())["pods"][0]["runtimeStatus"] == "stopped"


def test_real_detached_watchdog_stops_at_short_deadline_after_transient_failure(fixture):
    config, _, api, _, make = fixture
    config["runpod"].update(max_runtime_seconds=10, stop_reserve_seconds=6)
    controller = make(transient_stops=1)
    session = controller.start()
    state = json.loads(controller.state_path.read_text())
    assert state["watchdog_pid"] != os.getpid()
    assert os.getsid(state["watchdog_pid"]) != os.getsid(0)
    until = time.monotonic() + 12
    while time.monotonic() < until:
        state = json.loads(controller.state_path.read_text())
        if state["phase"] == "stopped":
            break
        time.sleep(.05)
    assert state["phase"] == "stopped"
    assert rc._epoch(state["stop_receipt"]["verified_at"]) <= session["hard_stop_unix"]
    provider = json.loads(api.read_text())
    assert provider["create_calls"] == 1 and provider["stop_calls"] >= 2
    assert provider["pods"][0]["runtimeStatus"] == "stopped"


def preservation(controller, primary, mirror, *, workload=False):
    primary.mkdir(parents=True, exist_ok=True)
    mirror.mkdir(parents=True, exist_ok=True)
    identity = {"pod_id": controller._owned_id(), "session_id": controller._state()["session_id"],
                "config_sha256": controller.config_sha}
    files = []

    def add(name, value):
        content = value if isinstance(value, bytes) else json.dumps(value).encode()
        for root in (primary, mirror):
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        row = {"path": name, "size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        files.append(row)
        return row

    manifests = []
    if workload:
        payload = add("runs/unit/model.bin", b"synthetic preserved checkpoint")
        manifest = add("persistence_receipts/unit.json", {"schema_version": 1, "persistence_mode": "host_pull",
                       "destination_root": str(primary), "ack_path": "persistence_receipts/unit.ack.json",
                       "files": [payload]})
        add("persistence_receipts/unit.ack.json", {"schema_version": 1, "manifest_sha256": manifest["sha256"],
            "verified_root": str(primary), "file_count": 1})
        manifests.append(manifest["path"])
    remote = list(files)
    add("remote_inventory.json", {"format": "deepmzyme_ec_clean30_runpod_remote_inventory_v1", **identity,
        "pending_transfer": None, "pending_unit": {"status": "completed", "persisted": True} if workload else None,
        "files": remote})
    path = controller.root / "preservation.json"
    path.write_text(json.dumps({"format": "deepmzyme_ec_clean30_runpod_preservation_v1", **identity,
        "primary_root": str(primary), "mirror_root": str(mirror), "files": files,
        "remote_inventory_path": "remote_inventory.json", "host_pull_manifests": manifests}))
    return path


@pytest.fixture
def mirror_device(tmp_path, monkeypatch):
    # Synthetic devices exercise validation without requiring another real disk in CPU fixtures.
    mirror = tmp_path / "mirror"
    original_stat = Path.stat

    def distinct(path, *args, **kwargs):
        value = original_stat(path, *args, **kwargs)
        if path == mirror:
            fields = list(value)
            fields[2] += 1
            return os.stat_result(fields)
        return value

    monkeypatch.setattr(Path, "stat", distinct)
    return mirror


@pytest.mark.parametrize("workload", [False, True])
def test_closeout_deletes_only_after_both_copies_and_acknowledgments(fixture, tmp_path, mirror_device, workload):
    _, _, api, _, make = fixture
    controller = make()
    controller.start()
    if workload:
        controller.admit_workload()
    path = preservation(controller, tmp_path / "primary", mirror_device, workload=workload)
    result = controller.closeout(path)
    assert result["status"] == "DELETED" and result["provider_absence_verified"]
    assert json.loads(api.read_text())["deleted_ids"] == ["ownedpod123"]
    assert controller.closeout(path)["status"] == "DELETED"
    assert json.loads(api.read_text())["deleted_ids"] == ["ownedpod123"]


@pytest.mark.parametrize("problem", ["mirror", "identity", "incomplete", "pending", "ack", "empty_admitted"])
def test_failed_preservation_never_deletes(fixture, tmp_path, mirror_device, problem):
    _, _, api, _, make = fixture
    controller = make()
    controller.start()
    controller.admit_workload()
    path = preservation(controller, tmp_path / "primary", mirror_device, workload=problem != "empty_admitted")
    proof = json.loads(path.read_text())
    if problem == "mirror":
        (mirror_device / "runs/unit/model.bin").write_bytes(b"corrupted")
    elif problem == "identity":
        proof["pod_id"] = "another-pod"
    elif problem == "incomplete":
        proof["files"] = [entry for entry in proof["files"] if entry["path"] != "runs/unit/model.bin"]
    elif problem == "pending":
        inventory = json.loads((tmp_path / "primary/remote_inventory.json").read_text())
        inventory["pending_unit"] = {"status": "running", "persisted": False}
        content = json.dumps(inventory).encode()
        for root in (tmp_path / "primary", mirror_device):
            (root / "remote_inventory.json").write_bytes(content)
        row = next(entry for entry in proof["files"] if entry["path"] == "remote_inventory.json")
        row.update(size_bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
    elif problem == "ack":
        proof["host_pull_manifests"] = []
    path.write_text(json.dumps(proof))
    with pytest.raises(rc.RunpodError):
        controller.closeout(path)
    assert not json.loads(api.read_text()).get("deleted_ids")


def ephemeral_adoption(config, primary, mirror):
    pod = adoption_fixture(config)
    mirror.mkdir(parents=True, exist_ok=True)
    config.update(durable_root=str(primary), mirror_root=str(mirror))
    cfg = config["runpod"]
    cfg.update(storage_mode="ephemeral_host_pull", volume_gb=0)
    cfg["quote"].update(stopped_storage_hourly_usd=0, stopped_storage_reserve_usd=0)
    cfg["retention_policy"]["action"] = "host_pull_before_stop"
    pod.update(volumeInGb=0, containerDiskInGb=cfg["container_disk_gb"])
    pod.pop("volumeMountPath")
    return pod


@pytest.fixture
def real_mirror(tmp_path):
    root = Path("/dev/shm")
    if not root.is_dir() or root.stat().st_dev == tmp_path.stat().st_dev:
        pytest.skip("A second local device is needed for the real detached ephemeral watchdog test.")
    path = Path(tempfile.mkdtemp(prefix="deepmzyme-controller-mirror-", dir=root))
    yield path
    shutil.rmtree(path)


@pytest.mark.parametrize("problem", ["unnamed", "same_device", "stopped_cost", "volume", "policy"])
def test_ephemeral_storage_is_exactly_scoped_to_independent_host_pull_adoption(fixture, tmp_path, mirror_device, problem):
    config, _, _, _, _ = fixture
    ephemeral_adoption(config, tmp_path / "primary", mirror_device)
    if problem == "unnamed":
        config["runpod"].pop("adopt_pod_id")
    elif problem == "same_device":
        config["mirror_root"] = str(tmp_path / "another-primary")
    elif problem == "stopped_cost":
        config["runpod"]["quote"]["stopped_storage_hourly_usd"] = .01
    elif problem == "volume":
        config["runpod"]["volume_gb"] = 1
    else:
        config["runpod"]["retention_policy"]["action"] = "verified_backup_then_delete"
    with pytest.raises(rc.RunpodError):
        rc.validate_config(config)


def test_ephemeral_named_adoption_uses_host_pull_and_stops_without_delete(fixture, tmp_path, real_mirror):
    config, _, api, _, make = fixture
    pod = ephemeral_adoption(config, tmp_path / "primary", real_mirror)
    controller = make(pods=[pod])
    session = controller.adopt("handedpod123", qualify=True)
    assert session["cost_admission"]["ceiling_scope"] == "execution_only_no_stopped_storage"
    controller.admit_workload()
    path = preservation(controller, tmp_path / "primary", real_mirror, workload=True)
    result = controller.closeout(path)
    assert result["status"] == "STOPPED" and result["provider_stop_verified"]
    assert result["ephemeral_data_erased_on_stop"] and result["deletion_authorized"] is False
    provider = json.loads(api.read_text())
    assert not provider.get("deleted_ids") and provider["pods"][0]["runtimeStatus"] == "stopped"


def test_failed_ephemeral_preservation_still_stops_and_never_qualifies(fixture, tmp_path, real_mirror):
    config, _, api, _, make = fixture
    pod = ephemeral_adoption(config, tmp_path / "primary", real_mirror)
    controller = make(pods=[pod])
    controller.adopt("handedpod123", qualify=True)
    controller.admit_workload()
    path = preservation(controller, tmp_path / "primary", real_mirror, workload=True)
    (real_mirror / "runs/unit/model.bin").unlink()
    with pytest.raises(rc.RunpodError, match="missing or differ"):
        controller.closeout(path)
    provider = json.loads(api.read_text())
    assert provider["pods"][0]["runtimeStatus"] == "stopped" and not provider.get("deleted_ids")
    assert "closeout_receipt" not in controller._state()


def test_container_restart_does_not_reset_adopted_allocation_clock(fixture):
    config, _, _, _, make = fixture
    pod = adoption_fixture(config)
    original = rc._epoch(config["runpod"]["adopt_started_at"])
    pod["startedAt"] = datetime.now(timezone.utc).isoformat()
    controller = make(pods=[pod])
    result = controller.adopt("handedpod123", qualify=True)
    assert result["allocation_started_unix"] == original
    assert result["hard_stop_unix"] == config["runpod"]["absolute_hard_stop_unix"]


def test_restarted_timestamp_cannot_rebase_an_older_allocation(fixture):
    config, _, api, _, make = fixture
    pod = adoption_fixture(config)
    original = rc._epoch(pod["createdAt"])
    restarted = original + 60
    pod["startedAt"] = datetime.fromtimestamp(restarted, timezone.utc).isoformat()
    config["runpod"].update(adopt_started_at=pod["startedAt"], absolute_hard_stop_unix=restarted + 6000)
    controller = make(pods=[pod])
    with pytest.raises(rc.RunpodError, match="clock cannot reset"):
        controller.adopt("handedpod123")
    assert not controller.owner_path.exists() and not controller.state_path.exists()
    assert not json.loads(api.read_text()).get("create_calls")


def test_persistent_named_adoption_also_has_no_deletion_authority(fixture, tmp_path, mirror_device):
    config, _, api, _, make = fixture
    pod = adoption_fixture(config)
    controller = make(pods=[pod])
    controller.adopt("handedpod123", qualify=True)
    path = preservation(controller, tmp_path / "primary", mirror_device)
    assert controller.closeout(path)["status"] == "STOPPED"
    assert not json.loads(api.read_text()).get("deleted_ids")
