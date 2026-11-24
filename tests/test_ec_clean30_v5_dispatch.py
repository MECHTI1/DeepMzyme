"""Synthetic CPU gates; fake transports cannot allocate, train or score."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ec_clean30 import execution_v5 as v5, phase3, phase4, workstation
from ec_clean30_fold0_fixtures import approved, sha  # noqa: F401


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return path


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
    # The controller's shape: the stop-request deadline lies one stop reserve before the hard stop.
    session = {"session_id": "owned", "allocation_started_unix": 100, "hard_stop_unix": 3700, "deadline_unix": 3580,
               "max_total_seconds": 3600}
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


def test_later_fold_pipeline_order_is_unchanged(approved, monkeypatch):
    # Fold 0 ends at its futility record (test_ec_clean30_v5_futility_boundary.py); a later fold has no
    # futility step and keeps the complete S1, S2, S4 and final CLEAN order.
    _, _, _, _, source = approved
    events = []
    request = {"execution_manifest": "manifest", "amendment_chain_receipt": "receipt",
               "expected_chain": source["amendment_chain"], **phase3.source_identity(source),
               "provider": "runpod", "fold": 1, "remote_campaign": "/workspace/campaign", "run_final_clean": True,
               "remote_request_template": {"session_id": "owned", "clean_inputs": "inner", "execution_root": "/workspace/campaign/runs/fold1/phase4"},
               "final_clean_request": {"clean_scope": "final", "clean_inputs": "final",
                                       "clean_forecasts": {}, "clean_timeouts": {}}}

    request["final_clean_request"]["farms_by_unit"] = phase4.final_farms_by_unit({**request, "clean_scope": "final"})

    def queue(name):
        def call(request, **options):
            assert options["close_session"] is False
            if request.get("clean_scope") == "final":
                assert request["remote_request_template"]["clean_inputs"] == "final"
                assert request["remote_request_template"]["farms_by_unit"] == request["farms_by_unit"]
            events.append(("final_embeddings" if name == "clean_embeddings" else "clean_final")
                          if request.get("clean_scope") == "final" else name)
            return {"status": "passed"}
        return call

    def step(request, *, step, **options):
        events.append(step)
        return {"status": "passed", **phase3.source_identity(source)}

    for name, function in (("clean_embeddings", "run_embedding_queue"), ("clean_inner", "run_clean_remote_queue"),
                           ("s1", "run_s1_queue"), ("s4", "run_refit_queue"), ("attach_fold", "run_fold_attachment")):
        monkeypatch.setattr(phase4, function, queue(name), raising=False)
    monkeypatch.setattr(phase4, "run_scientific_step", step, raising=False)
    result = v5.run_phase4_pipeline(request, transport=object(), check_session=lambda: None,
                                    record_progress=lambda name, result: None)
    assert events == ["attach_fold", "s1", "select_s1", "select_s2", "select_s4", "s4", "final_embeddings", "clean_final"]
    assert result["status"] == "passed"


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
