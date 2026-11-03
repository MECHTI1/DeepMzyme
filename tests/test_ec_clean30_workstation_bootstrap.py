"""CPU checks for ordered preparation and cache-backed Runpod admission."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
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
    monkeypatch.setattr(runpod_controller, "validate_config", lambda c: {"status": "ready"})
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

    class Controller:
        def start(self):
            events.append("start")
            return self.session()

        def session(self, previous=None):
            assert previous in (None, "owned-session")
            return {"session_id": "owned-session", "pod_id": "owned-pod",
                    "deadline_unix": time.time() + 7020}

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

    def preserve(*args):
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
                           controller=Controller(), transport=Transport())


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
