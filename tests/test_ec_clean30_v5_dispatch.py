"""Synthetic CPU gates; fake transports cannot allocate, train or score."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ec_clean30 import execution_v5 as v5, phase3, phase4, workstation


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def approved(tmp_path, monkeypatch):
    identity = {"scientific_identity_sha256": "a" * 64, "fit_chain": ["b" * 64, "c" * 64]}
    source = {"format": phase3.EXECUTION_V5_FORMAT, "published": True, **identity,
              "amendment_chain": identity["fit_chain"], "execution_manifest_sha256": "d" * 64,
              "git_head": "e" * 40}
    request = write(tmp_path / "reviewed-request.json", identity)
    preapproval = write(tmp_path / "preapproval.json", {"format": v5.PREAPPROVAL_FORMAT,
        "status": "approved", "campaign_id": phase3.CAMPAIGN_ID, **identity,
        "scope": "amendment2_fold_execution_v5", "ceilings": {"gpu_seconds": 3600, "cpu_seconds": 3600, "cost_usd": 8}})
    manifest = write(tmp_path / "manifest.json", source)
    write(request, {**identity, "execution_manifest": str(manifest)})
    config = {"provider": "runpod", "execution_scope": "g2_then_fold0_v5", "state_root": str(tmp_path / "pass"),
              "execution_manifest": str(manifest), "runpod": {"gross_ceiling_usd": 8, "max_runtime_seconds": 3600},
              "phase4": {"request_path": str(request), "request_sha256": sha(request),
                         "preapproval_path": str(preapproval), "preapproval_sha256": sha(preapproval)}}
    config_path = write(tmp_path / "config.json", config)
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o700)
    authorization = {"format": v5.AUTHORIZATION_FORMAT, "launch_authorized": True, "provider": "runpod",
        "scope": "g2_then_fold0_v5", "approved_by": "maintainer-fixture", **identity,
        "config_sha256": sha(config_path), "execution_manifest_sha256": source["execution_manifest_sha256"],
        "fold0_cost_ceiling_usd": 8, "failed_window_limit": 2, "failure_ledger": str(shared / "ledger.json"),
        "failure_ledger_authorization_sha256": "f" * 64}
    auth_path = write(tmp_path / "authorization.json", authorization)
    ledger = write(Path(authorization["failure_ledger"]), v5.empty_failure_ledger(
        authorization["failure_ledger"], authorization["failure_ledger_authorization_sha256"]))
    ledger.chmod(0o600)
    monkeypatch.setattr(phase3, "authenticated_chain", lambda source: source)
    monkeypatch.setattr(phase3, "verify_execution_source", lambda *args, **kwargs: source)
    monkeypatch.setattr(phase3, "require_record_precedence", lambda record, source: phase3.require_same_scientific_identity(record, source))
    monkeypatch.setattr(workstation, "prepare_session_chain_receipt", lambda config: {})
    monkeypatch.setattr(workstation, "validate_phase4_plan", lambda *args: None, raising=False)
    return config, config_path, auth_path, authorization, source


def test_authorization_binds_exact_config_science_and_ceiling(approved):
    config, path, auth_path, authorization, source = approved
    assert v5.validate_launch_authorization(config, path, auth_path, source) == authorization


@pytest.mark.parametrize("field,value", [("launch_authorized", False), ("fold0_cost_ceiling_usd", 9),
    ("fit_chain", ["b" * 64]), ("failed_window_limit", 3), ("config_sha256", "f" * 64), ("approved_by", "")])
def test_refusal_precedes_provider_access(approved, monkeypatch, field, value):
    config, path, auth_path, authorization, source = approved
    authorization[field] = value
    write(auth_path, authorization)
    monkeypatch.setattr(workstation, "_run_runpod_g2", lambda *a, **k: pytest.fail("Provider dispatch reached"))
    with pytest.raises(ValueError if field == "fit_chain" else RuntimeError):
        v5.run_v5(config, path, authorization_path=auth_path, window_id="window")


def test_in_memory_config_cannot_bypass_reviewed_bytes(approved):
    config, path, auth_path, _, source = approved
    with pytest.raises(v5.LaunchWindowError, match="in-memory"):
        v5.validate_launch_authorization({**config, "extra": "unreviewed"}, path, auth_path, source)


@pytest.mark.parametrize("terminal", [True, False])
def test_launch_failure_counts_only_after_verified_terminal_closeout(approved, monkeypatch, terminal):
    config, path, auth_path, authorization, source = approved

    def fail(config, path, *, pre_allocation, **callbacks):
        pre_allocation(config, path, {"source": source})
        write(Path(config["state_root"]) / "driver_state.json", {
            "start_attempted": True, "provider_stop_verified": terminal, "status": "interrupted", "pod_id": "owned",
            "stop_status": {"status": "EXITED", "runtime_status": "stopped", "pod_id": "owned"} if terminal else {}})
        raise RuntimeError("synthetic failed window")

    monkeypatch.setattr(workstation, "_run_runpod_g2", fail)
    with pytest.raises(RuntimeError, match="synthetic failed window"):
        v5.run_v5(config, path, authorization_path=auth_path, window_id="window")
    status = v5.FailedWindowLedger(authorization["failure_ledger"], authorization["failure_ledger_authorization_sha256"]).status()
    assert status["failed_windows"] == int(terminal)
    assert (status["unresolved_window"] is None) == terminal


def test_g3_materialization_cannot_increase_preapproved_envelope(approved, tmp_path):
    config, _, _, _, source = approved
    preapproval = json.loads(Path(config["phase4"]["preapproval_path"]).read_text())
    freeze = write(tmp_path / "freeze1c.json", {**phase3.source_identity(source), "format": "deepmzyme_ec_clean30_freeze1c_v2"})
    session = {"session_id": "owned", "allocation_started_unix": 100, "deadline_unix": 3700, "max_total_seconds": 3600}
    out = tmp_path / "g3.json"
    value = v5.materialize_g3(preapproval, preapproval_sha256=config["phase4"]["preapproval_sha256"],
                             freeze1c=freeze, out=out, source=source, session=session, gross_ceiling_usd=8)
    assert value["allocations"]["owned"]["max_total_seconds"] == 3600
    with pytest.raises(v5.LaunchWindowError, match="exceeds"):
        v5.materialize_g3(preapproval, preapproval_sha256=config["phase4"]["preapproval_sha256"],
                         freeze1c=freeze, out=tmp_path / "other.json", source=source,
                         session={**session, "max_total_seconds": 3601}, gross_ceiling_usd=8)


def test_new_approved_configs_do_not_reset_two_failed_windows(approved, tmp_path, monkeypatch):
    config, path, auth_path, authorization, source = approved
    starts = []

    def fail(config, path, *, pre_allocation, **callbacks):
        pre_allocation(config, path, {"source": source})
        starts.append(config["state_root"])
        write(Path(config["state_root"]) / "driver_state.json", {"start_attempted": True,
              "stop_status": {"status": "CREATE_REJECTED_NO_POD", "provider_absence_verified": True},
              "status": "interrupted"})
        raise RuntimeError("synthetic capacity rejection")

    monkeypatch.setattr(workstation, "_run_runpod_g2", fail)
    for number in range(3):
        config["state_root"] = str(tmp_path / f"pass{number}")
        write(path, config)
        authorization["config_sha256"] = sha(path)
        write(auth_path, authorization)
        with pytest.raises(RuntimeError, match="capacity rejection" if number < 2 else "Two additional"):
            v5.run_v5(config, path, authorization_path=auth_path, window_id=f"window{number}")
    assert len(starts) == 2
    ledger = v5.FailedWindowLedger(authorization["failure_ledger"], authorization["failure_ledger_authorization_sha256"])
    assert ledger.status()["failed_windows"] == 2


@pytest.mark.parametrize("decision", ["continue", "stop_futility"])
def test_pipeline_order_and_binding_stop(approved, monkeypatch, decision):
    _, _, _, _, source = approved
    events = []
    request = {"execution_manifest": "manifest", "amendment_chain_receipt": "receipt",
               "expected_chain": source["amendment_chain"], **phase3.source_identity(source),
               "provider": "runpod", "remote_campaign": "/workspace/campaign", "run_final_clean": True,
               "remote_request_template": {"session_id": "owned", "clean_inputs": "inner"},
               "final_clean_request": {"clean_scope": "final", "clean_inputs": "final",
                                       "clean_forecasts": {}, "clean_timeouts": {}}}

    def queue(name):
        def call(request, **options):
            assert options["close_session"] is False
            if request.get("clean_scope") == "final":
                assert request["remote_request_template"] == {"session_id": "owned", "clean_inputs": "final", "clean_scope": "final"}
            events.append("clean_final" if request.get("clean_scope") == "final" else name)
            return {"status": "passed"}
        return call

    def step(request, *, step, **options):
        events.append(step)
        return {"status": "passed", "decision": decision, **phase3.source_identity(source)}

    for name, function in (("clean_embeddings", "run_embedding_queue"), ("clean_inner", "run_clean_remote_queue"),
                           ("s1", "run_s1_queue"), ("s4", "run_refit_queue")):
        monkeypatch.setattr(phase4, function, queue(name), raising=False)
    monkeypatch.setattr(phase4, "run_scientific_step", step, raising=False)
    result = v5.run_phase4_pipeline(request, transport=object(), check_session=lambda: None,
                                    record_progress=lambda name, result: None)
    expected = ["clean_embeddings", "clean_inner", "held_out_sanity", "verify_sanity", "s1", "select_s1", "select_s2", "futility"]
    if decision == "continue":
        expected += ["select_s4", "s4", "clean_final"]
    assert events == expected
    assert result["status"] == ("passed" if decision == "continue" else "stop_futility")


@pytest.mark.parametrize("state,expected", [
    ({"start_attempted": True, "stop_status": {"status": "NO_CREATE_ATTEMPT"}}, "not_allocated"),
    ({"start_attempted": True, "stop_status": {"status": "CREATE_CANCELLED_BEFORE_POST"}}, "not_allocated"),
    ({"start_attempted": True, "stop_status": {"status": "CREATE_REJECTED_NO_POD", "provider_absence_verified": True}}, "verified_absent"),
    ({"start_attempted": True, "provider_stop_verified": True}, None),
    ({"start_attempted": True, "pod_id": "owned", "stop_status": {"status": "EXITED", "runtime_status": "stopped", "pod_id": "foreign"}}, None),
    ({"start_attempted": True, "pod_id": "owned", "closeout": {"status": "DELETED", "provider_absence_verified": True, "pod_id": "owned"}}, "verified_absent"),
])
def test_ledger_terminal_classification_uses_actual_controller_receipt(state, expected):
    assert v5.verified_terminal_state(state) == expected
