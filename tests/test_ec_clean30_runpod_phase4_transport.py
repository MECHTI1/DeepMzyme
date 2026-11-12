"""Inert same-session transport checks; no provider process or model execution."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from ec_clean30 import phase4, runpod_controller as controller, workstation


@pytest.fixture
def owned(tmp_path):
    session = {"pod_id": "pod", "session_id": "session", "allocation_started_unix": 1000,
               "deadline_unix": 20000, "max_total_seconds": 20000}
    calls = []
    owner = SimpleNamespace(config_sha="a" * 64, session=lambda previous: dict(session),
                            _cleanup_state=lambda: dict(session),
                            admit_output_root=lambda root: calls.append("admit"),
                            stop=lambda: calls.append("stop") or {"runtime_status": "stopped"})
    transport = workstation.RunpodTransport({"state_root": str(tmp_path), "bootstrap_python": "/usr/bin/python3",
                                             "runpod": {"ssh_alias": "owned"}}, owner)
    request = {**session, "provider": "runpod", "runpod_config_sha256": owner.config_sha}
    return request, transport, calls


@pytest.mark.parametrize("field", [None, "pod_id", "session_id", "allocation_started_unix", "deadline_unix",
                                   "max_total_seconds", "runpod_config_sha256"])
def test_runpod_attach_checks_original_allocation_without_gcp(owned, field):
    request, transport, calls = owned
    if field:
        request[field] = "changed"
        with pytest.raises(phase4.Phase4Error, match="configuration|identity|deadline"):
            phase4._attach_session(request, transport)
    else:
        assert phase4._attach_session(request, transport)["pod_id"] == "pod"
        assert phase4._close_session(request, transport)["provider_stop_verified"]
        assert calls == ["stop"]


@pytest.mark.parametrize("fail", [False, True])
def test_borrowed_queue_preserves_session_and_defers_stop_to_coordinator(owned, tmp_path, monkeypatch, fail):
    request, transport, lifecycle = owned
    request.update(execution_manifest="manifest.json", durable_root=str(tmp_path / "primary"),
                   mirror_root=str(tmp_path / "mirror"), host_state_root=str(tmp_path / "state"),
                   max_download_bytes=100, remote_runner="/repo/scripts/run_ec_clean30_phase4.py")
    request["remote_request_template"] = {**request, "execution_root": "/campaign/runs/fold0/phase4"}
    units = [{"unit_id": "first", "unit_timeout_seconds": 10}, {"unit_id": "second", "unit_timeout_seconds": 10}]
    transport.deadline_unix, transport.reserve_seconds = 20000, 30
    events = []
    monkeypatch.setattr(phase4.phase3, "verify_execution_source", lambda *a: {})
    monkeypatch.setattr(phase4, "verify_authority", lambda *a, **kw: None)
    monkeypatch.setattr(workstation, "HostPull", lambda *a, **kw: SimpleNamespace(
        sync_pending=lambda root: events.append("sync")))
    def complete(transport_arg, host, current, **kwargs):
        assert transport_arg is transport
        events.append(current["unit_id"])
        if fail:
            raise RuntimeError("terminal failure")
        return {"status": "passed", "unit_id": current["unit_id"]}
    monkeypatch.setattr(workstation, "complete_remote_unit", complete)
    run = lambda: phase4._run_remote_queue(request, units, action="s1-unit", next_gate="gate",
        transport=transport, check_session=lambda: transport.stop_controller.session("session"), close_session=False)
    if fail:
        with pytest.raises(RuntimeError, match="terminal failure"):
            run()
    else:
        assert run()["completed"] == ["first", "second"]
    assert events[-1] == "sync" and "stop" not in lifecycle
    assert (transport.deadline_unix, transport.reserve_seconds) == (20000, 30)


def test_remote_clean_queue_uses_exact_inner_units_and_gpu_requests(monkeypatch):
    captured = {}
    def queue(request, units, **kwargs):
        captured.update(units=units, options=kwargs)
        return {"status": "passed"}
    monkeypatch.setattr(phase4, "_run_remote_queue", queue)
    names = [unit["unit_id"] for unit in phase4.clean_units()]
    request = {"provider": "runpod", "remote_request_template": {"device": "cuda"},
               "clean_forecasts": dict.fromkeys(names, 20), "clean_timeouts": dict.fromkeys(names, 10)}
    phase4.run_clean_remote_queue(request, close_session=False, transport="owned", check_session="check")
    assert [row["unit_id"] for row in captured["units"]] == names
    assert captured["options"]["action"] == "clean-unit"
    request["remote_request_template"]["device"] = "cpu"
    with pytest.raises(phase4.Phase4Error, match="GPU route"):
        phase4.run_clean_remote_queue(request)


@pytest.mark.parametrize("problem", [None, "missing", "foreign", "hash"])
def test_preservation_set_checks_every_registered_root(tmp_path, problem):
    identity = {"pod_id": "pod", "session_id": "session", "config_sha256": "a" * 64}
    roots, children = [], []
    for name in ("g2", "phase4"):
        root = {"execution_root": f"/campaign/runs/fold0/{name}", "primary_root": str(tmp_path / "primary" / name),
                "mirror_root": str(tmp_path / "mirror" / name)}
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(root))
        roots.append(root)
        children.append({"path": str(path), "sha256": controller._sha(path)})
    proof = {"format": "deepmzyme_ec_clean30_runpod_preservation_set_v1", **identity, "receipts": children}
    if problem == "missing": proof["receipts"] = children[:1]
    if problem == "foreign": proof["session_id"] = "foreign"
    if problem == "hash": children[1]["sha256"] = "b" * 64
    path = tmp_path / "set.json"
    path.write_text(json.dumps(proof))
    checked = []
    def verify_child(path, *, _allow_set):
        assert _allow_set is False
        checked.append(path)
        return controller._sha(path)
    owner = SimpleNamespace(config={}, config_sha=identity["config_sha256"], _owned_id=lambda: "pod",
                            _state=lambda: {"session_id": "session", "required_preservation_roots": roots},
                            _verify_preservation=verify_child)
    if problem:
        with pytest.raises(controller.RunpodError):
            controller.RunpodController._verify_preservation(owner, path)
    else:
        assert controller.RunpodController._verify_preservation(owner, path) == controller._sha(path)
        assert checked == [Path(child["path"]) for child in children]


@pytest.mark.parametrize("overlap", [False, True])
def test_output_root_registration_rejects_alias_or_overlap(tmp_path, overlap):
    roots = [{"execution_root": "/campaign/g2", "primary_root": str(tmp_path / "primary"),
              "mirror_root": str(tmp_path / "mirror")}]
    roots.append({"execution_root": "/campaign/phase4", "primary_root": str(tmp_path / "primary" / "nested")
                  if overlap else str(tmp_path / "primary"), "mirror_root": str(tmp_path / "mirror2")})
    owner = SimpleNamespace(_unchanged=lambda: None)
    with pytest.raises(controller.RunpodError, match="alias or overlap"):
        controller.RunpodController.register_output_roots(owner, roots)


@pytest.mark.parametrize("override", ["remote_runner", "remote_request_template", "python", "cap_prefix",
                                      "freeze1c", "g3_authority", "g3_sha256", "execution_manifest", "durable_root"])
def test_queue_overrides_cannot_change_execution_or_authority(owned, tmp_path, override):
    request, transport, _ = owned
    config = {"execution_manifest": str(tmp_path / "manifest.json")}
    state = {"completed": [name for _, name in workstation.QUEUE]}
    def prepare(config, live_transport, *args):
        assert live_transport is None
        return {"execution_manifest": config["execution_manifest"], "queue_actions": ["s1"],
                "queue_requests": {"s1": {override: "changed"}}}
    with pytest.raises(workstation.WorkstationError, match="runtime and authority are fixed"):
        workstation._continue_runpod_phase4(config, transport, state, tmp_path / "state.json", request,
            {}, "remote_manifest", "remote_probe", lambda: request, prepare)


@pytest.mark.parametrize("admitted", [False, True])
def test_empty_registered_phase4_tree_is_allowed_only_before_its_own_admission(tmp_path, monkeypatch, admitted):
    identity = {"pod_id": "pod", "session_id": "session", "config_sha256": "a" * 64}
    primary, mirror = tmp_path / "primary", tmp_path / "mirror"
    for root in (primary, mirror):
        (root / "closeout").mkdir(parents=True)
    inventory = {"format": "deepmzyme_ec_clean30_runpod_remote_inventory_v1", **identity,
                 "files": [], "pending_transfer": None, "pending_unit": None}
    relative = "closeout/remote_inventory.json"
    for root in (primary, mirror):
        (root / relative).write_text(json.dumps(inventory))
    item = primary / relative
    proof = {"format": "deepmzyme_ec_clean30_runpod_preservation_v1", **identity,
             "execution_root": "/campaign/phase4", "primary_root": str(primary), "mirror_root": str(mirror),
             "remote_inventory_path": relative, "host_pull_manifests": [],
             "files": [{"path": relative, "size_bytes": item.stat().st_size, "sha256": controller._sha(item)}]}
    path = tmp_path / "proof.json"
    path.write_text(json.dumps(proof))
    original_stat = Path.stat
    def device_stat(path, *args, **kwargs):
        result = original_stat(path, *args, **kwargs)
        if path == mirror:
            values = list(result)
            values[2] += 1
            return os.stat_result(values)
        return result
    monkeypatch.setattr(Path, "stat", device_stat)
    state = {"session_id": "session", "workload_admitted": True,
             "required_preservation_roots": [{"execution_root": "/campaign/g2"}, {"execution_root": "/campaign/phase4"}],
             "admitted_output_roots": ["/campaign/g2"] + (["/campaign/phase4"] if admitted else [])}
    owner = SimpleNamespace(config={}, config_sha=identity["config_sha256"], _owned_id=lambda: "pod", _state=lambda: state)
    if admitted:
        with pytest.raises(controller.RunpodError, match="admitted workload"):
            controller.RunpodController._verify_preservation(owner, path, _allow_set=False)
    else:
        assert controller.RunpodController._verify_preservation(owner, path, _allow_set=False) == controller._sha(path)
        proof["execution_root"] = "/campaign/g2"
        path.write_text(json.dumps(proof))
        with pytest.raises(controller.RunpodError, match="admitted workload"):
            controller.RunpodController._verify_preservation(owner, path, _allow_set=False)


def test_controller_refuses_pending_hot_marker_even_when_execution_state_looks_terminal(tmp_path, monkeypatch):
    identity = {"pod_id": "pod", "session_id": "session", "config_sha256": "a" * 64}
    roots = [tmp_path / "primary", tmp_path / "mirror"]
    name = "runs/clean_unit/attempt1/hot_storage.json"
    for root in roots:
        target = root / name
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps({"format": "deepmzyme_ec_clean30_hot_storage_v1", "status": "pending"}))
    target = roots[0] / name
    proof = {"format": "deepmzyme_ec_clean30_runpod_preservation_v1", **identity,
             "execution_root": "/campaign/phase4", "primary_root": str(roots[0]), "mirror_root": str(roots[1]),
             "files": [{"path": name, "size_bytes": target.stat().st_size, "sha256": controller._sha(target)}]}
    path = tmp_path / "proof.json"
    path.write_text(json.dumps(proof))
    original_stat = Path.stat

    def device_stat(path, *args, **kwargs):
        result = original_stat(path, *args, **kwargs)
        if path == roots[1]:
            values = list(result)
            values[2] += 1
            return os.stat_result(values)
        return result

    monkeypatch.setattr(Path, "stat", device_stat)
    owner = SimpleNamespace(config={}, config_sha=identity["config_sha256"], _owned_id=lambda: "pod",
                            _state=lambda: {"session_id": "session", "workload_admitted": False})
    with pytest.raises(controller.RunpodError, match="Hot attempt is not completely preserved"):
        controller.RunpodController._verify_preservation(owner, path)
