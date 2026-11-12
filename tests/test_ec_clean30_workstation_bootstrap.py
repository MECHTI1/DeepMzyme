"""CPU checks for ordered preparation and cache-backed Runpod admission."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import os
from pathlib import Path
from types import SimpleNamespace
import threading
import time

import pytest

from ec_clean30 import phase3, runpod_bootstrap as bootstrap, runpod_controller, workstation as ws


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_incomplete_runtime_reference_fails_before_any_archive_or_price_read(monkeypatch, tmp_path):
    config = {"provider": "runpod", "stage": [{"remote": "/archive"}],
              "runtime_environments": {"fit": {"packages": {"torch": "fixture"}}, "esm1b": {}}}
    monkeypatch.setattr(ws, "_sha", lambda *a: pytest.fail("archive read before structural refusal"))
    monkeypatch.setattr(runpod_controller, "validate_config", lambda *a: pytest.fail("price check before structural refusal"))
    with pytest.raises(ws.WorkstationError, match="locked, relocatable archive"):
        ws.preflight(config, tmp_path / "config.json")


@pytest.mark.parametrize("change,reused", [("quote", True), ("placement", True), ("input", False),
                                           ("runtime", False), ("deadline", False), ("source", False)])
def test_prepared_input_proof_has_a_closed_metadata_whitelist(tmp_path, monkeypatch, change, reused):
    config = {"provider": "runpod", "artifact_cache": {"fixture": True}, "state_root": str(tmp_path / "state"),
              "execution_manifest": str(tmp_path / "manifest.json"), "stage": [{"sha256": "a" * 64}],
              "runtime_environments": {"fit": "locked"},
              "runpod": {"data_center_id": "EU-RO-1", "max_runtime_seconds": 7020,
                         "quote": {"observed_at": "before", "source": "fixture", "hourly_compute_usd": .89}}}
    path = tmp_path / "config.json"
    write(path, config)
    source = {"published": True, "git_head": "a" * 40}
    calls = []
    monkeypatch.setattr(ws, "_validate_runpod_structure", lambda c: None)
    monkeypatch.setattr(ws, "_local_storage", lambda c: None)
    monkeypatch.setattr(runpod_controller, "validate_config", lambda c, *, fresh: {"status": "ready"} if fresh
                        else pytest.fail("a launch preflight relaxed quote freshness"))
    monkeypatch.setattr(phase3, "verify_execution_source", lambda *a, **kw: dict(source))
    monkeypatch.setattr(ws, "preflight", lambda *a, **kw: calls.append("full_input_check") or
                        {"status": "ready", "source": dict(source), "staging_peak_bytes": 1})
    original = ws._prepared_preflight(config, path, published=True)
    assert not original["input_proof_reused"]
    if change == "quote":
        config["runpod"]["quote"].update(observed_at="after", source="fresh")
    elif change == "placement":
        config["runpod"]["data_center_id"] = "EUR-IS-1"
    elif change == "input":
        config["stage"][0]["sha256"] = "b" * 64
    elif change == "runtime":
        config["runtime_environments"]["fit"] = "different"
    elif change == "deadline":
        config["runpod"]["max_runtime_seconds"] = 7000
    else:
        source["git_head"] = "b" * 40
    write(path, config)
    result = ws._prepared_preflight(config, path, published=True)
    assert result["input_proof_reused"] is reused
    assert len(calls) == (1 if reused else 2)
    assert result["config_sha256"] == ws._sha(path)


def test_bootstrap_delegation_refusal_precedes_checkout_or_payload(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    helper = repo / "src/ec_clean30/container_caps.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("# authenticated helper\n")
    monkeypatch.setattr(phase3, "REPO_ROOT", repo)
    events = []

    class Transport:
        def upload(self, local, remote):
            events.append(("upload", Path(local).name))

        def stat_file(self, remote):
            return {"sha256": ws._sha(helper), "size_bytes": helper.stat().st_size}

        def remote(self, argv, **kwargs):
            events.append(("remote", argv))
            raise ws.WorkstationError("container delegation refused")

    with pytest.raises(ws.WorkstationError, match="delegation refused"):
        ws._bootstrap_runpod({"bootstrap_python": "/usr/bin/python3", "cgroup_root": "/sys/fs/cgroup",
                              "state_root": str(tmp_path / "state")}, Transport(),
                             {"source_sha256": {"src/ec_clean30/container_caps.py": ws._sha(helper)}},
                             {"session_id": "session", "pod_id": "owned-pod"})
    assert events[0] == ("upload", "container_caps.py")
    assert len(events) == 2 and "prepare" in events[1][1]


def test_main_runtime_lock_refusal_precedes_remaining_large_transfers(tmp_path, monkeypatch):
    main, other = {"remote": "/main", "kind": "tar"}, {"remote": "/clean", "kind": "tar"}
    config = {"stage": [other, main], "runtime_environments": {"fit": {"archive": "/main"}},
              "remote_python": "/main/bin/python", "state_root": str(tmp_path)}
    events = []
    monkeypatch.setattr(ws, "_stage_checkout", lambda *a: events.append("checkout"))
    monkeypatch.setattr(bootstrap, "hydrate_public_artifact", lambda *a: False)
    monkeypatch.setattr(ws, "_stage_file", lambda c, t, i: events.append(i["remote"]))
    monkeypatch.setattr(ws, "_unpack_stage", lambda *a: None)
    monkeypatch.setattr(ws, "_container_probe", lambda *a: None)
    monkeypatch.setattr(ws, "_verify_runtime_lock", lambda *a: (_ for _ in ()).throw(ws.WorkstationError("runtime lock differs")))
    with pytest.raises(ws.WorkstationError, match="runtime lock differs"):
        ws._stage_runpod_runtime(config, SimpleNamespace(deadline_unix=9999999999), {}, lambda: None)
    assert events == ["checkout", "/main"]


def test_cache_seed_mode_and_execution_are_exclusive(tmp_path):
    with pytest.raises(SystemExit) as result:
        ws.main(["--config", str(tmp_path / "config.json"), "--seed-cache", "--run"])
    assert result.value.code == 2


@pytest.fixture
def cached_driver(tmp_path, monkeypatch):
    """An owned allocation with inert controller and transport boundaries."""
    events = []
    main, clean = tmp_path / "main.tar.gz", tmp_path / "clean.tar.gz"
    main.write_bytes(b"main archive")
    clean.write_bytes(b"clean archive")
    config = {"provider": "runpod", "artifact_cache": {"fixture": True},
              "state_root": str(tmp_path / "state"), "gcp_controller_root": str(tmp_path / "gcp"),
              "gcp_stopped_identity": {"name": "retained"}, "bootstrap_python": "/usr/bin/python3",
              "stage": [{"local": str(main), "remote": "/main", "kind": "file"},
                        {"local": str(clean), "remote": "/clean", "kind": "file"}],
              "runtime_environments": {"fit": {"archive": "/main"}, "esm1b": {"archive": "/clean"}},
              "remote_python": "/main/bin/python", "profiles": {"fit": {"cap_prefix": ["limits"]}},
              "remote_repo": "/repo", "hardware_profile": "approved"}
    path = tmp_path / "config.json"
    write(path, config)
    source = {"format": phase3.BOOTSTRAP_FORMAT, "published": True}
    ready = {"source": source, "staging_peak_bytes": 1}
    begun = time.time()
    session_record = {"session_id": "owned-session", "pod_id": "owned-pod", "allocation_started_unix": begun,
                      "deadline_unix": begun + 6900, "hard_stop_unix": begun + 7020}

    class Controller:
        def start(self):
            events.append("start")
            return self.session()

        def session(self, previous=None):
            assert previous in (None, "owned-session")
            return dict(session_record)

        @contextmanager
        def maintain_heartbeat(self, *, interval):
            assert interval == 30
            events.append("heartbeat_enter")
            try:
                yield SimpleNamespace(check=lambda: events.append("heartbeat_check"))
            finally:
                events.append("heartbeat_exit")

        def write_ssh_config(self):
            return tmp_path / "ssh_config"

        def admit_workload(self):
            events.append("admit")

        def stop(self):
            events.append("stop")
            return {"desiredStatus": "EXITED", "runtimeStatus": "stopped"}

    class Transport:
        def remote(self, argv, **kwargs):
            if "import shutil; print(shutil.disk_usage('/workspace').free)" in argv:
                return str(1000 * ws.GIB)
            if "cuda proof" in argv:
                return "{}"
            return "ready"

    def preserve(*args, max_download_bytes):
        assert type(max_download_bytes) is int and max_download_bytes > 0
        events.append("preserve")
        return None

    monkeypatch.setattr(ws, "_prepared_preflight", lambda *a, **kw: ready)
    monkeypatch.setattr(bootstrap, "verify_seeded_cache", lambda *a, **kw: events.append("cache_verified"))
    monkeypatch.setattr(bootstrap, "staging_scratch_bytes", lambda *a: 1)
    monkeypatch.setattr(ws.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="GCP TERMINATED"))
    monkeypatch.setattr(ws, "_status", lambda *a: "TERMINATED")
    monkeypatch.setattr(ws, "_bootstrap_runpod", lambda *a: events.append("bootstrap"))
    monkeypatch.setattr(ws, "_runtime_preflight", lambda *a: events.append("runtime_preflight") or ("manifest", "probe"))
    monkeypatch.setattr(ws, "_execute_queue", lambda *a: events.append("queue"))
    monkeypatch.setattr(ws, "_preserve_runpod_outputs", preserve)
    return SimpleNamespace(config=config, path=path, ready=ready, events=events,
                           controller=Controller(), transport=Transport(), session=session_record)


def test_cold_cache_refuses_before_any_allocation_or_provider_command(cached_driver, monkeypatch):
    case = cached_driver

    def refuse(*args, **kwargs):
        case.events.append("cache_refused")
        raise RuntimeError("cache descriptor missing")

    monkeypatch.setattr(bootstrap, "verify_seeded_cache", refuse)
    monkeypatch.setattr(ws.subprocess, "run", lambda *a, **kw: pytest.fail("provider command before warm-cache proof"))
    with pytest.raises(RuntimeError, match="cache descriptor missing"):
        ws._run_runpod_g2(case.config, case.path, controller=case.controller, transport=case.transport)
    assert case.events == ["cache_refused"]
    state_path = Path(case.config["state_root"]) / "driver_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    assert not state.get("start_attempted")


def test_v5_gate_refusal_precedes_controller_construction_and_provider(cached_driver, monkeypatch):
    case = cached_driver
    from ec_clean30 import runpod_controller
    case.config["execution_scope"] = "g2_then_fold0_v5"
    case.config["g2_max_download_bytes"] = 10 * ws.GIB
    case.ready["source"]["format"] = phase3.EXECUTION_V5_FORMAT
    write(case.path, case.config)
    def forbidden(*args, **kwargs):
        pytest.fail("controller or provider reached before launch authorization")
    monkeypatch.setattr(runpod_controller, "RunpodController", forbidden)
    monkeypatch.setattr(ws.subprocess, "run", forbidden)
    def refuse(config, path, ready):
        assert config is case.config and path == case.path and ready is case.ready
        raise ws.WorkstationError("authorization pending")
    with pytest.raises(ws.WorkstationError, match="authorization pending"):
        ws._run_runpod_g2(case.config, case.path, pre_allocation=refuse, prepare_phase4=lambda *a: {},
                         prepare_g2=lambda *a: pytest.fail("preparation reached before launch authorization"),
                         complete_g2=lambda *a: pytest.fail("completion reached before launch authorization"))
    assert case.events == []


def test_v5_cannot_use_legacy_entry_or_omit_callbacks(cached_driver):
    case = cached_driver
    case.config["execution_scope"] = "g2_then_fold0_v5"
    with pytest.raises(ws.WorkstationError, match="launch dispatcher"):
        ws.run_g2(case.config, case.path)
    with pytest.raises(ws.WorkstationError, match="launch gate"):
        ws._run_runpod_g2(case.config, case.path, controller=case.controller, transport=case.transport)
    assert case.events == []


def test_v5_continues_before_single_owned_closeout(cached_driver, monkeypatch):
    case = cached_driver
    case.config["execution_scope"] = "g2_then_fold0_v5"
    case.config["g2_max_download_bytes"] = 10 * ws.GIB
    case.ready["source"]["format"] = phase3.EXECUTION_V5_FORMAT
    case.ready["source"]["execution_manifest_sha256"] = "a" * 64
    write(case.path, case.config)
    prepare = lambda *a: {}
    def gate(config, path, ready):
        case.events.append("launch_gate")
        return {"config_sha256": ws._sha(path), "execution_manifest_sha256": "a" * 64}
    monkeypatch.setattr(ws, "_stage_runpod_runtime", lambda *a: case.events.append("stage"))
    monkeypatch.setattr(ws, "_prepare_g2_v5", lambda *a: case.events.append("pretrain"))
    monkeypatch.setattr(ws, "_complete_g2_v5", lambda *a: case.events.append("freeze1c_v2"))
    def continuation(config, transport, state, path, session, ready, manifest, probe, check, callback):
        assert transport is case.transport and callback is prepare and session["session_id"] == "owned-session"
        assert check()["pod_id"] == "owned-pod"
        case.events.append("phase4")
    monkeypatch.setattr(ws, "_continue_runpod_phase4", continuation)
    result = ws._run_runpod_g2(case.config, case.path, controller=case.controller, transport=case.transport,
                              pre_allocation=gate, prepare_phase4=prepare,
                              prepare_g2=lambda *a: {}, complete_g2=lambda *a: {})
    assert case.events.count("start") == case.events.count("stop") == 1
    assert case.events.index("launch_gate") < case.events.index("cache_verified") < case.events.index("start")
    assert (case.events.index("pretrain") < case.events.index("queue") < case.events.index("freeze1c_v2")
            < case.events.index("phase4") < case.events.index("preserve") < case.events.index("stop"))
    assert result["provider_stop_verified"]


def test_cached_route_requires_execution_v4_before_cache_or_allocation(cached_driver):
    case = cached_driver
    case.ready["source"]["format"] = phase3.RUNPOD_FORMAT
    with pytest.raises(ws.WorkstationError, match="execution-v4 manifest"):
        ws._run_runpod_g2(case.config, case.path, controller=case.controller, transport=case.transport)
    assert case.events == []


@pytest.mark.parametrize("failure", ["bootstrap", "main_lock", "remaining_transfer"])
def test_cached_preparation_failure_closes_owned_pod_without_admission(cached_driver, monkeypatch, failure):
    case = cached_driver

    def refuse(*args, **kwargs):
        raise ws.WorkstationError("qualification refused")

    if failure == "bootstrap":
        monkeypatch.setattr(ws, "_bootstrap_runpod", refuse)
        monkeypatch.setattr(ws, "_stage_runpod_runtime", lambda *a: pytest.fail("staging after bootstrap refusal"))
    else:
        monkeypatch.setattr(ws, "_stage_checkout", lambda *a: case.events.append("checkout"))
        monkeypatch.setattr(bootstrap, "hydrate_public_artifact", lambda *a: False)

        def stage_file(config, transport, item):
            if failure == "remaining_transfer" and item["remote"] == "/clean":
                refuse()
            case.events.append(item["remote"])

        monkeypatch.setattr(ws, "_stage_file", stage_file)
        monkeypatch.setattr(ws, "_unpack_stage", lambda *a: None)
        monkeypatch.setattr(ws, "_container_probe", lambda *a: case.events.append("main_caps"))
        monkeypatch.setattr(ws, "_verify_runtime_lock", refuse if failure == "main_lock" else lambda *a: None)
        monkeypatch.setattr(phase3, "cuda_probe_program", lambda *a: "cuda proof")
        monkeypatch.setattr(phase3, "validate_cuda_evidence", lambda *a: None)
    with pytest.raises(ws.WorkstationError, match="qualification refused"):
        ws._run_runpod_g2(case.config, case.path, controller=case.controller, transport=case.transport)
    assert case.events.index("cache_verified") < case.events.index("start")
    assert case.events.count("start") == case.events.count("stop") == 1
    assert case.events.index("preserve") < case.events.index("stop") < case.events.index("heartbeat_exit")
    assert not {"runtime_preflight", "admit", "queue", "/clean"}.intersection(case.events)
    if failure != "bootstrap":
        assert case.events.count("checkout") == case.events.count("/main") == 1
    state = json.loads((Path(case.config["state_root"]) / "driver_state.json").read_text())
    assert state["closed"] is True and state["provider_stop_verified"] is True
    assert state["status"] == "interrupted"
    assert not (Path(case.config['state_root']) / 'operator_hold.json').exists()


def _operator_case(case, monkeypatch, *, now=None, sleep=None):
    case.config['runpod'] = {'keep_alive_for_operator': True}
    case.config['execution_manifest'] = str(case.path.with_name('manifest.json'))
    write(case.path, case.config)
    clock = [case.session['allocation_started_unix'] + 1 if now is None else now]

    def advance(seconds):
        assert 0 <= seconds <= 15
        if sleep is not None: sleep(clock)
        clock[0] += seconds

    monkeypatch.setattr(ws, 'time', SimpleNamespace(time=lambda: clock[0], sleep=advance))
    monkeypatch.setattr(ws.phase3, 'verify_execution_source', lambda *a, **kw: dict(case.ready['source']))
    monkeypatch.setattr(ws, '_local_storage', lambda *a: case.events.append('local_storage'))
    monkeypatch.setattr(ws, '_stage_runpod_runtime', lambda *a: case.events.append('staging'))
    return clock


def test_keepalive_retries_qualification_on_same_pod_then_holds_completed_queue(cached_driver, monkeypatch):
    case = cached_driver
    root = Path(case.config['state_root'])
    attempts, held_sessions = [], []

    def bootstrap_attempt(*args):
        attempts.append(True)
        if len(attempts) == 1: raise ws.WorkstationError('repairable qualification failure')

    def queue(config, transport, state, *args):
        case.events.append('queue')
        state.update(completed=[name for _, name in ws.QUEUE], status='measurements_complete')

    def control(clock):
        hold = ws._operator_json(root / 'operator_hold.json')
        assert 'stop' not in case.events and 'heartbeat_exit' not in case.events
        held_sessions.append({key: hold[key] for key in case.session})
        action = 'retry-qualification' if len(attempts) == 1 else 'closeout'
        assert hold['retry_allowed'] == (action == 'retry-qualification')
        ws.request_operator(case.config, case.path, action)
        assert (root / 'operator_request.json').stat().st_mode & 0o777 == 0o600

    _operator_case(case, monkeypatch, sleep=control)
    monkeypatch.setattr(ws, '_bootstrap_runpod', bootstrap_attempt)
    monkeypatch.setattr(ws, '_execute_queue', queue)
    result = ws._run_runpod_g2(case.config, case.path, controller=case.controller, transport=case.transport)
    assert attempts == [True, True] and held_sessions == [case.session, case.session]
    assert case.events.count('start') == case.events.count('stop') == case.events.count('queue') == 1
    assert result['qualification_retry_count'] == 1 and result['completed'] == [name for _, name in ws.QUEUE]
    assert result['closed'] and result['status'] == 'measurements_complete'
    assert result['qualification_errors'] == ['WorkstationError: repairable qualification failure']
    assert not ws._operator_json(root / 'operator_hold.json')['active']


def test_keepalive_deadline_preserves_closeout_reserve_and_never_restarts(cached_driver, monkeypatch):
    case = cached_driver
    until = case.session['deadline_unix'] - ws.RESERVE_SECONDS
    clock = _operator_case(case, monkeypatch, now=until - 20)
    monkeypatch.setattr(ws, '_bootstrap_runpod', lambda *a: (_ for _ in ()).throw(ws.WorkstationError('qualification failed')))
    with pytest.raises(ws.WorkstationError, match='qualification failed'):
        ws._run_runpod_g2(case.config, case.path, controller=case.controller, transport=case.transport)
    assert clock[0] == until and case.events.count('start') == case.events.count('stop') == 1
    state = json.loads((Path(case.config['state_root']) / 'driver_state.json').read_text())
    assert state['operator_hold']['exit_action'] == 'closeout-reserve'
    assert state['deadline_unix'] == case.session['deadline_unix'] and state['closed']
    assert 'admit' not in case.events


def test_ssh_qualification_failure_enters_hold_before_any_cleanup(cached_driver, monkeypatch):
    case = cached_driver
    root = Path(case.config['state_root'])
    holds = []

    def control(clock):
        if not (root / 'operator_hold.json').exists(): return
        assert 'stop' not in case.events and 'bootstrap' not in case.events
        holds.append(True)
        ws.request_operator(case.config, case.path, 'closeout')

    _operator_case(case, monkeypatch, sleep=control)
    monkeypatch.setattr(case.transport, 'remote', lambda *a, **kw:
                        (_ for _ in ()).throw(ws.WorkstationError('SSH not reachable')))
    with pytest.raises(ws.WorkstationError, match='startup bound'):
        ws._run_runpod_g2(case.config, case.path, controller=case.controller, transport=case.transport)
    assert holds == [True] and case.events.count('start') == case.events.count('stop') == 1


def test_admitted_failure_can_hold_but_never_retry_or_overwrite_units(cached_driver, monkeypatch, tmp_path):
    case = cached_driver
    root = Path(case.config['state_root'])
    output = tmp_path / 'F1_complete'
    output.write_text('verified prior result')
    visits = []

    def queue(config, transport, state, *args):
        visits.append('queue')
        state['completed'] = ['F1']
        raise ws.WorkstationError('later unit failed')

    def control(clock):
        assert 'stop' not in case.events and output.read_text() == 'verified prior result'
        hold = ws._operator_json(root / 'operator_hold.json')
        assert not hold['retry_allowed']
        if visits[-1] == 'queue':
            with pytest.raises(ws.WorkstationError, match='forbidden after workload admission'):
                ws.request_operator(case.config, case.path, 'retry-qualification')
            request = {key: hold[key] for key in ('hold_id', 'config_sha256', 'session_id', 'pod_id',
                                                'deadline_unix', 'source_identity_sha256')}
            ws._operator_write(root / 'operator_request.json', {**request, 'action': 'retry-qualification',
                                                               'created_unix': clock[0]})
            visits.append('forged_retry')
        else:
            assert not ws._operator_json(root / 'operator_control_result.json')['accepted']
            ws.request_operator(case.config, case.path, 'closeout')

    _operator_case(case, monkeypatch, sleep=control)
    monkeypatch.setattr(ws, '_execute_queue', queue)
    with pytest.raises(ws.WorkstationError, match='later unit failed'):
        ws._run_runpod_g2(case.config, case.path, controller=case.controller, transport=case.transport)
    assert visits == ['queue', 'forged_retry'] and case.events.count('start') == 1
    state = json.loads((root / 'driver_state.json').read_text())
    assert state['completed'] == ['F1'] and not state.get('qualification_retry_count')
    assert output.read_text() == 'verified prior result'


@pytest.mark.parametrize('change', ['hold_id', 'session_id', 'pod_id', 'config_sha256', 'deadline_unix',
                                   'source_identity_sha256', 'stale', 'future', 'permissions', 'source_changed'])
def test_operator_hold_rejects_foreign_stale_or_changed_source_controls(cached_driver, monkeypatch, change):
    case = cached_driver
    root = Path(case.config['state_root'])
    visits = []

    def control(clock):
        assert 'stop' not in case.events
        if not visits:
            ws.request_operator(case.config, case.path, 'retry-qualification')
            path = root / 'operator_request.json'
            request = ws._operator_json(path)
            if change == 'stale': request['created_unix'] -= 61
            elif change == 'future': request['created_unix'] += 100
            elif change == 'permissions': pass
            elif change == 'source_changed':
                monkeypatch.setattr(ws.phase3, 'verify_execution_source', lambda *a, **kw: {'published': True, 'different': True})
            else: request[change] = 'foreign'
            ws._operator_write(path, request)
            if change == 'permissions': path.chmod(0o644)
            visits.append('invalid')
        else:
            result = ws._operator_json(root / 'operator_control_result.json')
            assert result['accepted'] is False and result['reason']
            ws.request_operator(case.config, case.path, 'closeout')

    _operator_case(case, monkeypatch, sleep=control)
    monkeypatch.setattr(ws, '_bootstrap_runpod', lambda *a: (_ for _ in ()).throw(ws.WorkstationError('qualification failed')))
    with pytest.raises(ws.WorkstationError, match='qualification failed'):
        ws._run_runpod_g2(case.config, case.path, controller=case.controller, transport=case.transport)
    assert visits == ['invalid'] and case.events.count('start') == case.events.count('stop') == 1
    state = json.loads((root / 'driver_state.json').read_text())
    assert not state.get('qualification_retry_count') and 'admit' not in case.events


def test_operator_request_requires_fresh_private_hold_and_never_calls_provider(cached_driver, monkeypatch):
    case = cached_driver
    clock = _operator_case(case, monkeypatch)
    root = Path(case.config['state_root'])
    root.mkdir(parents=True)
    root.chmod(0o700)
    ws._operator_write(root / 'operator_hold.json', {'active': True, 'config_sha256': ws._sha(case.path),
        'updated_unix': clock[0] - 46, 'hold_until_unix': clock[0] + 100})
    with pytest.raises(ws.WorkstationError, match='stale'):
        ws.request_operator(case.config, case.path, 'closeout')
    assert not (root / 'operator_request.json').exists() and case.events == []


def test_operator_action_is_exclusive_with_run_and_seed(tmp_path):
    for action in ('--run', '--seed-cache'):
        with pytest.raises(SystemExit) as result:
            ws.main(['--config', str(tmp_path / 'config.json'), '--operator', 'closeout', action])
        assert result.value.code == 2


def test_private_operator_controls_use_nvme_when_data_status_has_nonposix_modes(cached_driver, monkeypatch, tmp_path):
    case = cached_driver
    case.config['mirror_root'] = str(tmp_path / 'nvme' / 'host')
    root = tmp_path / 'nvme' / 'operator'
    status_root = Path(case.config['state_root'])
    visits = []

    def control(clock):
        state = status_root / 'driver_state.json'
        state.chmod(0o777)  # Simulate Data1's synthetic status-file mode.
        status_root.chmod(0o777)
        hold = ws._operator_json(root / 'operator_hold.json')
        assert hold['control_root'] == str(root)
        assert root.stat().st_mode & 0o777 == 0o700
        assert (root / 'operator_hold.json').stat().st_mode & 0o777 == 0o600
        assert not (status_root / 'operator_hold.json').exists()
        visits.append(True)
        ws.request_operator(case.config, case.path, 'closeout')
        assert (root / 'operator_request.json').stat().st_mode & 0o777 == 0o600

    _operator_case(case, monkeypatch, sleep=control)
    monkeypatch.setattr(ws, '_bootstrap_runpod', lambda *a: (_ for _ in ()).throw(ws.WorkstationError('repairable')))
    with pytest.raises(ws.WorkstationError, match='repairable'):
        ws._run_runpod_g2(case.config, case.path, controller=case.controller, transport=case.transport)
    assert visits == [True] and case.events.count('stop') == 1


@pytest.mark.parametrize('problem', ['symlink', 'nonposix_directory', 'nonposix_file'])
def test_private_control_permissions_are_proved_before_allocation(cached_driver, monkeypatch, tmp_path, problem):
    case = cached_driver
    case.config['mirror_root'] = str(tmp_path / 'nvme' / 'host')
    _operator_case(case, monkeypatch)
    root = tmp_path / 'nvme' / 'operator'
    root.parent.mkdir()
    if problem == 'symlink':
        target = tmp_path / 'foreign'
        target.mkdir()
        root.symlink_to(target, target_is_directory=True)
    else:
        root.mkdir(mode=0o755)
        if problem == 'nonposix_directory':
            monkeypatch.setattr(ws.os, 'chmod', lambda *a: None)
        else:
            original = ws._operator_write

            def synthetic_file_modes(path, value):
                original(path, value)
                path.chmod(0o777)

            monkeypatch.setattr(ws, '_operator_write', synthetic_file_modes)
    with pytest.raises(ws.WorkstationError, match='Operator|private owned'):
        ws._run_runpod_g2(case.config, case.path, controller=case.controller, transport=case.transport)
    assert case.events == []


def test_staging_checks_main_first_and_limits_remaining_writers_to_two(tmp_path, monkeypatch):
    main = {"remote": "/main", "kind": "file"}
    others = [{"remote": f"/other{index}", "kind": "file"} for index in range(4)]
    config = {"stage": [*others[:2], main, *others[2:]], "runtime_environments": {"fit": {"archive": "/main"}},
              "profiles": {"fit": {"cap_prefix": ["limits"]}}, "remote_python": "/main/bin/python",
              "remote_repo": "/repo", "hardware_profile": "approved", "state_root": str(tmp_path)}
    events, started = [], []
    active = peak = 0
    lock = threading.Lock()
    two_started, release = threading.Event(), threading.Event()

    def stage_file(config, transport, item):
        nonlocal active, peak
        if item is main:
            events.append("main")
            return
        with lock:
            active += 1
            peak = max(active, peak)
            started.append(item["remote"])
            events.append(item["remote"])
            if len(started) == 2:
                two_started.set()
        try:
            assert release.wait(10), "Test did not release the bounded staging workers."
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(ws, "_stage_checkout", lambda *a: events.append("checkout"))
    monkeypatch.setattr(bootstrap, "hydrate_public_artifact", lambda *a: False)
    monkeypatch.setattr(ws, "_stage_file", stage_file)
    monkeypatch.setattr(ws, "_unpack_stage", lambda *a: None)
    monkeypatch.setattr(ws, "_container_probe", lambda *a: events.append("caps"))
    monkeypatch.setattr(ws, "_verify_runtime_lock", lambda *a: events.append("lock"))
    monkeypatch.setattr(phase3, "cuda_probe_program", lambda *a: "cuda proof")
    monkeypatch.setattr(phase3, "validate_cuda_evidence", lambda *a: events.append("cuda"))
    transport = SimpleNamespace(deadline_unix=time.time() + 600, remote=lambda *a, **kw: "{}")
    with ThreadPoolExecutor(max_workers=1) as driver:
        future = driver.submit(ws._stage_runpod_runtime, config, transport, {"published": True}, lambda: None)
        try:
            assert two_started.wait(10), "Independent staging writers never started."
            with lock:
                assert active == 2 and len(started) == 2
            assert events[:5] == ["checkout", "main", "caps", "lock", "cuda"]
        finally:
            release.set()
        future.result(timeout=10)
    assert events.count("checkout") == events.count("main") == 1
    assert peak == 2 and active == 0
    assert sorted(started) == sorted(item["remote"] for item in others)
    receipt = json.loads((tmp_path / "staging_complete.json").read_text())
    assert receipt["workers"] == 2 and receipt["main_runtime_first"] is True


def test_keepalive_staging_failure_joins_existing_writer_before_recovery(tmp_path, monkeypatch):
    main = {'remote': '/main', 'kind': 'file'}
    others = [{'remote': '/failing', 'kind': 'file'}, {'remote': '/still_running', 'kind': 'file'}]
    config = {'runpod': {'keep_alive_for_operator': True}, 'stage': [main, *others],
        'runtime_environments': {'fit': {'archive': '/main'}}, 'profiles': {'fit': {'cap_prefix': ['limits']}},
        'remote_python': '/main/python', 'remote_repo': '/repo', 'hardware_profile': 'approved', 'state_root': str(tmp_path)}
    both_started = threading.Barrier(2)
    release, shutdown_entered = threading.Event(), threading.Event()
    events, shutdown_waits = [], []

    class TrackedPool(ThreadPoolExecutor):
        def shutdown(self, wait=True, *, cancel_futures=False):
            shutdown_waits.append(wait)
            shutdown_entered.set()
            return super().shutdown(wait=wait, cancel_futures=cancel_futures)

    def stage(config, transport, item):
        if item is main: return
        both_started.wait(timeout=10)
        if item['remote'] == '/failing': raise ws.WorkstationError('staging failed')
        assert release.wait(10)
        events.append('writer_settled')

    def drive():
        try:
            ws._stage_runpod_runtime(config, SimpleNamespace(deadline_unix=time.time() + 600,
                remote=lambda *a, **kw: '{}'), {}, lambda: None)
        finally:
            events.append('recovery_can_begin')

    monkeypatch.setattr(ws, 'ThreadPoolExecutor', TrackedPool)
    monkeypatch.setattr(ws, '_stage_checkout', lambda *a: None)
    monkeypatch.setattr(bootstrap, 'hydrate_public_artifact', lambda *a: False)
    monkeypatch.setattr(ws, '_stage_file', stage)
    monkeypatch.setattr(ws, '_unpack_stage', lambda *a: None)
    monkeypatch.setattr(ws, '_container_probe', lambda *a: None)
    monkeypatch.setattr(ws, '_verify_runtime_lock', lambda *a: None)
    monkeypatch.setattr(phase3, 'cuda_probe_program', lambda *a: 'probe')
    monkeypatch.setattr(phase3, 'validate_cuda_evidence', lambda *a: None)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(drive)
        try:
            assert shutdown_entered.wait(10)
            assert shutdown_waits == [True] and not future.done()
            assert 'recovery_can_begin' not in events
        finally:
            release.set()
        with pytest.raises(ws.WorkstationError, match='staging failed'): future.result(timeout=10)
    assert events == ['writer_settled', 'recovery_can_begin']


def test_posix_preparation_failure_precedes_payload_and_preserves_closeout(cached_driver, monkeypatch):
    case = cached_driver

    def refuse(*args):
        case.events.append('posix_refused')
        raise ws.WorkstationError('POSIX storage refused')

    monkeypatch.setattr(ws, '_prepare_runpod_storage', refuse)
    monkeypatch.setattr(ws, '_stage_runpod_runtime', lambda *a: pytest.fail('payload before POSIX proof'))
    with pytest.raises(ws.WorkstationError, match='POSIX storage refused'):
        ws._run_runpod_g2(case.config, case.path, controller=case.controller, transport=case.transport)
    assert 'posix_refused' in case.events
    assert case.events.index('preserve') < case.events.index('stop')
    assert not {'admit', 'queue'}.intersection(case.events)


def test_only_kernel_delegation_refusals_classify_as_host_rejections(tmp_path, monkeypatch):
    from ec_clean30 import container_caps
    local = phase3.REPO_ROOT / "src/ec_clean30/container_caps.py"
    source = {"source_sha256": {"src/ec_clean30/container_caps.py": ws._sha(local)}}
    config = {"bootstrap_python": "/py", "cgroup_root": "/sys/fs/cgroup", "state_root": str(tmp_path)}
    session = {"session_id": "s1", "pod_id": "p1"}

    class Transport:
        def __init__(self, error):
            self.error = error

        def upload(self, *a):
            pass

        def stat_file(self, remote):
            return {"sha256": ws._sha(local), "size_bytes": local.stat().st_size}

        def remote(self, argv, **kw):
            raise ws.WorkstationError(self.error)

    with pytest.raises(ws.HostRejected):
        ws._bootstrap_runpod(config, Transport(ws.HOST_REJECTION_PREFIX + " v1 controls are not writable"), source, session)
    with pytest.raises(ws.WorkstationError) as other:
        ws._bootstrap_runpod(config, Transport("Remote command failed (2): ssh lost"), source, session)
    assert not isinstance(other.value, ws.HostRejected)

    class Evidence(Transport):
        def remote(self, argv, **kw):
            return json.dumps({"parent_limits_unchanged": True} if "prepare" in argv else {"fixture": "weak"})

    monkeypatch.setattr(container_caps, "validate_probe_evidence",
                        lambda e: (_ for _ in ()).throw(container_caps.CapError("swap present")))
    with pytest.raises(ws.HostRejected, match="swap present"):
        ws._bootstrap_runpod(config, Evidence(""), source, session)


def _v5_host_case(case, monkeypatch, rejections):
    case.config["execution_scope"] = "g2_then_fold0_v5"
    case.config["g2_max_download_bytes"] = 10 * ws.GIB
    case.ready["source"]["format"] = phase3.EXECUTION_V5_FORMAT
    case.ready["source"]["execution_manifest_sha256"] = "a" * 64
    write(case.path, case.config)
    events, begun = case.events, time.time()

    class Host:
        def __init__(self, n):
            self.n, self.root = n, Path(case.config["state_root"]) / ("." if n == 1 else f"host_try_{n}")

        def start(self, replacing=None):
            events.append(("start", self.n, replacing and replacing["pod_id"]))
            return self.session()

        def session(self, previous=None):
            return {"session_id": f"s{self.n}", "pod_id": f"p{self.n}", "allocation_started_unix": begun,
                    "deadline_unix": begun + 6900, "hard_stop_unix": begun + 7020}

        @contextmanager
        def maintain_heartbeat(self, *, interval):
            events.append(("heartbeat_enter", self.n))
            try:
                yield SimpleNamespace(check=lambda: None)
            finally:
                events.append(("heartbeat_exit", self.n))

        def write_ssh_config(self):
            return Path(case.config["state_root"]) / "ssh_config"

        def admit_workload(self):
            events.append(("admit", self.n))

        def stop(self):
            events.append(("stop", self.n))
            return {"desiredStatus": "EXITED", "runtimeStatus": "stopped"}

        def release_rejected_host(self, reason):
            assert reason.startswith("HostRejected: " + ws.HOST_REJECTION_PREFIX)
            events.append(("release", self.n))
            return {"status": "DELETED", "pod_id": f"p{self.n}"}

        def replacement_released(self):
            events.append(("released", self.n))

    tries = iter(range(rejections))

    def bootstrap(config, transport, source, session):
        events.append(("bootstrap", session["pod_id"]))
        if next(tries, None) is not None:
            raise ws.HostRejected(ws.HOST_REJECTION_PREFIX + " fixture host")

    monkeypatch.setattr(ws, "_bootstrap_runpod", bootstrap)
    monkeypatch.setattr(ws, "_host_controller", lambda config, path, n: Host(n))
    monkeypatch.setattr(ws, "_host_transport", lambda config, controller: case.transport)
    monkeypatch.setattr(ws, "_stage_runpod_runtime", lambda *a: events.append("stage"))
    monkeypatch.setattr(ws, "_prepare_g2_v5", lambda *a: events.append("pretrain"))
    monkeypatch.setattr(ws, "_complete_g2_v5", lambda *a: events.append("freeze1c_v2"))
    monkeypatch.setattr(ws, "_continue_runpod_phase4", lambda *a: events.append("phase4"))
    gate = lambda config, path, ready: {"config_sha256": ws._sha(path), "execution_manifest_sha256": "a" * 64}
    return Host(1), dict(pre_allocation=gate, prepare_phase4=lambda *a: {}, prepare_g2=lambda *a: {},
                         complete_g2=lambda *a: {})


def test_v5_host_rejection_replaces_the_host_inside_the_same_window(cached_driver, monkeypatch):
    case = cached_driver
    first, callbacks = _v5_host_case(case, monkeypatch, rejections=1)
    result = ws._run_runpod_g2(case.config, case.path, controller=first, transport=case.transport, **callbacks)
    e = case.events
    assert ("start", 1, None) in e and ("start", 2, "p1") in e
    # The rejected pod is released only after the replacement create returned, then the overlap ends.
    assert (e.index(("bootstrap", "p1")) < e.index(("start", 2, "p1")) < e.index(("release", 1))
            < e.index(("heartbeat_exit", 1)) < e.index(("released", 2)) < e.index(("bootstrap", "p2")))
    assert ("stop", 1) not in e and e.count(("stop", 2)) == 1
    assert e.index(("admit", 2)) < e.index("queue") < e.index("phase4") < e.index("preserve") < e.index(("stop", 2))
    assert result["host_try"] == 2 and result["pod_id"] == "p2" and result["session_id"] == "s2"
    assert [row["pod_id"] for row in result["host_rejections"]] == ["p1"]
    assert result["host_rejections"][0]["release"]["status"] == "DELETED"
    assert "error" not in result and result["provider_stop_verified"]


def test_v5_host_rejections_stop_after_the_bounded_number_of_tries(cached_driver, monkeypatch):
    from ec_clean30 import runpod_controller
    case = cached_driver
    first, callbacks = _v5_host_case(case, monkeypatch, rejections=runpod_controller.HOST_TRY_LIMIT)
    with pytest.raises(ws.HostRejected):
        ws._run_runpod_g2(case.config, case.path, controller=first, transport=case.transport, **callbacks)
    e = case.events
    limit = runpod_controller.HOST_TRY_LIMIT
    assert [x for x in e if isinstance(x, tuple) and x[0] == "start"] == [
        ("start", 1, None), *[("start", n, f"p{n - 1}") for n in range(2, limit + 1)]]
    assert [x for x in e if isinstance(x, tuple) and x[0] == "release"] == [("release", n) for n in range(1, limit)]
    assert [x for x in e if isinstance(x, tuple) and x[0] == "stop"] == [("stop", limit)]
    assert not {"queue", "stage"}.intersection(x for x in e if isinstance(x, str))
    state = json.loads((Path(case.config["state_root"]) / "driver_state.json").read_text())
    assert state["closed"] is True and state["status"] == "interrupted" and state["host_try"] == limit


def test_prepared_input_proof_moves_to_a_new_pass_without_rehashing(tmp_path, monkeypatch):
    config = {"provider": "runpod", "artifact_cache": {"fixture": True}, "state_root": str(tmp_path / "w1"),
              "durable_root": str(tmp_path / "g2a"), "mirror_root": str(tmp_path / "g2b"),
              "execution_manifest": str(tmp_path / "manifest.json"), "stage": [{"sha256": "a" * 64}],
              "runpod": {"state_root": str(tmp_path / "w1"), "pod_name": "w1-pod", "max_runtime_seconds": 7020,
                         "quote": {"observed_at": "before", "source": "fixture", "hourly_compute_usd": .89}}}
    path = tmp_path / "config.json"
    write(path, config)
    calls = []
    monkeypatch.setattr(ws, "_validate_runpod_structure", lambda c: None)
    monkeypatch.setattr(ws, "_local_storage", lambda c: calls.append(("local_storage", c["state_root"])))
    monkeypatch.setattr(runpod_controller, "validate_config", lambda c, *, fresh: {"status": "ready"})
    monkeypatch.setattr(phase3, "verify_execution_source", lambda *a, **kw: {"published": True})
    monkeypatch.setattr(ws, "preflight", lambda *a, **kw: calls.append("full_input_check") or
                        {"status": "ready", "source": {"published": True}, "staging_peak_bytes": 1})
    ws._prepared_preflight(config, path, published=True)
    for name in ("prepared_inputs.json", "prepared_config.json"):
        (tmp_path / "w2").mkdir(exist_ok=True)
        (tmp_path / "w2" / name).write_bytes((tmp_path / "w1" / name).read_bytes())
    config.update(state_root=str(tmp_path / "w2"), durable_root=str(tmp_path / "g2c"), mirror_root=str(tmp_path / "g2d"))
    config["runpod"].update(state_root=str(tmp_path / "w2"), pod_name="w2-pod")
    write(path, config)
    result = ws._prepared_preflight(config, path, published=True)
    assert result["input_proof_reused"] is True and calls.count("full_input_check") == 1
    assert ("local_storage", str(tmp_path / "w2")) in calls
    config["runpod"]["max_runtime_seconds"] = 7000
    write(path, config)
    assert ws._prepared_preflight(config, path, published=True)["input_proof_reused"] is False
