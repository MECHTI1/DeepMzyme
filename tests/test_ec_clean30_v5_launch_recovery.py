"""Real Runpod controller sessions through the v5 launch, workstation and phase-3 gates.

A local fake provider CLI and a fake retained-VM status script stand in for the
cloud; no network, provider account, background allocation or GPU is used.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

import pytest

from ec_clean30 import campaign, execution_v5 as v5, phase3, runpod_controller as rc, workstation as ws
from test_ec_clean30_runpod_controller import FAKE

GIB = 1024 ** 3
POD_NAME = "g2-v5-recovery-fixture"
TIMING = ("allocation_started_unix", "deadline_unix", "hard_stop_unix", "max_total_seconds")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class UnreachablePod:
    """Every pod transport call fails; only the controller's provider reads succeed."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        def unreachable(*args, **kwargs):
            raise LookupError(f"synthetic unreachable pod: {name}")
        return unreachable


@pytest.fixture
def launch(tmp_path, monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    cli = tmp_path / "fake_runpodctl"
    cli.write_text(FAKE.replace("#!/usr/bin/env python3", f"#!{sys.executable}"))
    cli.chmod(0o700)
    api = cli.with_suffix(".json")
    credential = write(tmp_path / "credentials.json", {"api_key": "rpa_fake_secret"})
    credential.chmod(0o600)
    key = tmp_path / "ssh_key"
    key.write_text("synthetic key, never used for SSH")
    key.chmod(0o600)
    vm_status = tmp_path / "gcp" / "bin" / "vm-status"
    vm_status.parent.mkdir(parents=True)
    identity = {"scientific_identity_sha256": "a" * 64, "fit_chain": ["b" * 64, "c" * 64]}
    source = {"format": phase3.EXECUTION_V5_FORMAT, "published": True, **identity,
              "amendment_chain": identity["fit_chain"], "execution_manifest_sha256": "d" * 64,
              "git_head": "e" * 40, "hardware_profiles": ["runpod_rtx4090"]}
    manifest = write(tmp_path / "manifest.json", source)
    request = write(tmp_path / "reviewed-request.json", {**identity, "execution_manifest": str(manifest)})
    preapproval = write(tmp_path / "preapproval.json", {"format": v5.PREAPPROVAL_FORMAT, "status": "approved",
        "campaign_id": phase3.CAMPAIGN_ID, **identity, "scope": "amendment2_fold_execution_v5",
        "ceilings": {"gpu_seconds": 7200, "cpu_seconds": 7200, "cost_usd": 8}})
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o700)
    ledger_path = shared / "ledger.json"
    write(ledger_path, v5.empty_failure_ledger(ledger_path, "f" * 64)).chmod(0o600)
    state_root = tmp_path / "pass"
    config = {"provider": "runpod", "execution_scope": "g2_then_fold0_v5", "hardware_profile": "runpod_rtx4090",
              "state_root": str(state_root), "execution_manifest": str(manifest),
              "bootstrap_python": "/usr/bin/python3", "remote_python": "/runtime-fit/bin/python",
              "remote_repo": "/workspace/repo", "remote_campaign": "/workspace/campaign",
              "durable_root": str(tmp_path / "durable"), "mirror_root": str(tmp_path / "mirror"),
              "g2_max_download_bytes": GIB, "timing_inputs": "/workspace/campaign/inputs/timing",
              "gcp_controller_root": str(vm_status.parents[1]),
              "gcp_stopped_identity": {"project": "p", "zone": "z", "vm_name": "retained"},
              "profiles": {"fit": {"cap_prefix": ["cap", "--"], "memory_limit_bytes": 8 * GIB},
                           "esm1b": {"cap_prefix": ["cap", "--"], "memory_limit_bytes": 24 * GIB}},
              "embedding": {"esmc": {}, "esm1b": {}},
              "lane_limits": {"cpu_percent": 600, "ram_bytes": 24 * GIB, "vram_bytes": 21 * GIB},
              "runpod": {"cli_path": str(cli), "credential_file": str(credential), "pod_name": POD_NAME,
                         "gpu_id": "NVIDIA GeForce RTX 4090", "data_center_id": "EUR-IS-2",
                         "image": "example/image@sha256:" + "a" * 64, "min_cuda_version": "13.0",
                         "container_disk_gb": 50, "volume_gb": 20, "volume_mount_path": "/workspace",
                         "ssh_alias": "runpod-v5-fixture", "ssh_private_key": str(key), "requested_hours": 2,
                         "max_runtime_seconds": 7020, "gross_ceiling_usd": 8, "stop_reserve_seconds": 120,
                         "keep_alive_for_operator": False, "stock_search_seconds": 0,
                         "runtime_environment": {"NVIDIA_TF32_OVERRIDE": "0"},
                         "retention_policy": {"retention_seconds": 86400, "cleanup_reviewed": True,
                                              "action": "verified_backup_then_delete", "owner": "fixture owner",
                                              "evidence": "synthetic approved cleanup policy"},
                         "quote": {"source": "synthetic quote", "hourly_compute_usd": .89,
                                   "running_storage_hourly_usd": .01, "stopped_storage_reserve_usd": .25,
                                   "stopped_storage_hourly_usd": .01, "egress_reserve_usd": .25,
                                   "prior_charge_usd": .6, "margin_fraction": .15}},
              "phase4": {"request_path": str(request), "request_sha256": sha(request),
                         "preapproval_path": str(preapproval), "preapproval_sha256": sha(preapproval)}}
    case = SimpleNamespace(config=config, config_path=tmp_path / "config.json", api=api, source=source,
                           authorization_path=tmp_path / "authorization.json", state_root=state_root,
                           driver_state=state_root / "driver_state.json", session_path=state_root / "session.json",
                           ledger=v5.FailedWindowLedger(ledger_path, "f" * 64), preflights=[],
                           real_prepared_preflight=ws._prepared_preflight)

    def configure(*, quote_age=0.0, max_age=900, gcp_delay=0, pods=()):
        vm_status.write_text(f"#!/bin/sh\nsleep {gcp_delay}\n"
                             "echo 'VM: retained (project=p zone=z)'\necho 'State: TERMINATED'\n")
        vm_status.chmod(0o700)
        api.write_text(json.dumps({"pods": list(pods)}))
        observed = datetime.fromtimestamp(time.time() - quote_age, timezone.utc).isoformat()
        config["runpod"]["quote"].update(observed_at=observed, max_age_seconds=max_age)
        write(case.config_path, config)
        write(case.authorization_path, {"format": v5.AUTHORIZATION_FORMAT, "launch_authorized": True,
            "provider": "runpod", "scope": "g2_then_fold0_v5", "approved_by": "maintainer-fixture", **identity,
            "config_sha256": sha(case.config_path), "execution_manifest_sha256": source["execution_manifest_sha256"],
            "fold0_cost_ceiling_usd": 8, "failed_window_limit": 2, "failure_ledger": str(ledger_path),
            "failure_ledger_authorization_sha256": "f" * 64})

    def binding(window_id):
        """The launch binding run_v5 hands the workstation for this window."""
        return {"config_sha256": sha(case.config_path), "execution_manifest_sha256": source["execution_manifest_sha256"],
                "authorization_sha256": sha(case.authorization_path), "window_id": window_id,
                "failure_ledger": str(ledger_path)}

    def run(window_id, **kwargs):
        return v5.run_v5(config, case.config_path, authorization_path=case.authorization_path,
                         window_id=window_id, **kwargs)

    def prepared(current, path, *, published=False, fresh=True):
        case.preflights.append(fresh)
        # The real preflight starts with this controller validation; input hashing is outside these contracts.
        return {"status": "ready", "source": source, "cost": rc.validate_config(current, fresh=fresh),
                "staging_peak_bytes": 0}

    case.configure, case.binding, case.run = configure, binding, run
    monkeypatch.setattr(phase3, "authenticated_chain", lambda current: current)
    monkeypatch.setattr(phase3, "verify_execution_source", lambda *args, **kwargs: source)
    monkeypatch.setattr(phase3, "require_record_precedence",
                        lambda record, current: phase3.require_same_scientific_identity(record, current))
    monkeypatch.setattr(ws, "prepare_session_chain_receipt", lambda current: {})
    monkeypatch.setattr(ws, "validate_phase4_plan", lambda *args: None)
    monkeypatch.setattr(ws, "_prepared_preflight", prepared)
    yield case
    if case.session_path.exists():
        state = json.loads(case.session_path.read_text())
        if state.get("pod_id") and state.get("phase") not in {"stopped", "deleted"}:
            rc.RunpodController(json.loads(case.config_path.read_text()), case.config_path).stop()
        if state.get("watchdog_pid"):
            try:
                os.kill(state["watchdog_pid"], signal.SIGTERM)
            except ProcessLookupError:
                pass


def provider_calls(case):
    return json.loads(case.api.read_text()).get("calls", [])


def ledger_rows(case):
    return json.loads(case.ledger.path.read_text())["windows"]


def test_real_controller_session_passes_the_v5_runtime_gate_for_every_g2_unit(launch, monkeypatch):
    case = launch
    case.configure()
    controller = rc.RunpodController(case.config, case.config_path)
    session = controller.start()
    # The controller keeps one stop reserve between the stop-request deadline and its hard stop.
    assert session["hard_stop_unix"] - session["deadline_unix"] == 120
    assert session["max_total_seconds"] == session["hard_stop_unix"] - session["allocation_started_unix"] == 7020
    resumed = rc.RunpodController(case.config, case.config_path).session(session["session_id"])
    assert {key: resumed[key] for key in TIMING} == {key: session[key] for key in TIMING}

    recipes = tuple((f"science_recipe_{index}", object()) for index in range(7))
    monkeypatch.setattr(campaign, "timing_recipes_v2", lambda: recipes, raising=False)
    config = case.config
    config["bounds"] = {name: {"forecast_seconds": 100, "unit_timeout_seconds": 90}
                        for _, name in ws.measurement_queue(case.source)}
    for name in ("main_model_probe.json", "esm1b_model_probe.json"):
        (case.state_root / name).write_text("{}")
    # Only the published artifacts checked before the session binding are synthetic.
    view = SimpleNamespace(fold=0, smoke=False, manifest_sha256="v" * 64,
                           default_runs_dir=Path(config["remote_campaign"]) / "runs" / "fold0")
    caps = {"synthetic": True}
    probe_path = f"{config['remote_campaign']}/g2_requests/vm_probe.json"
    probe = {"status": "passed", "view_manifest_sha256": view.manifest_sha256,
             "source": phase3.source_identity(case.source), "hardware_profile": "runpod_rtx4090",
             "container_caps_probe": caps, "container_caps_probe_sha256": "d" * 64,
             "container_caps_evidence_sha256": hashlib.sha256(phase3._bytes(caps)).hexdigest()}
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    monkeypatch.setattr(phase3.views.View, "load", lambda *args, **kwargs: view)
    for name in ("_bind_view", "verify_pretrain_evidence", "_verify_container_caps",
                 "_verify_esm1b_model_probe", "_verify_main_model_probe"):
        monkeypatch.setattr(phase3, name, lambda *args, **kwargs: None)
    original_json = phase3._json
    monkeypatch.setattr(phase3, "_json", lambda path: probe if str(path) == probe_path else original_json(path))
    admitted, requests = [], {}

    def complete(transport, host, request, *, action, completed_relative, **kwargs):
        _, _, policy = phase3._runtime(request, pretraining=action == "pretrain-prepare")
        admitted.append(action)
        requests.setdefault(action, dict(request))
        assert (policy.allocation_started_unix, policy.deadline_unix, policy.max_total_seconds) == (
            session["allocation_started_unix"], session["deadline_unix"], session["max_total_seconds"])
        record = {"status": "passed", **phase3.source_identity(case.source)}
        write(Path(config["durable_root"]) / completed_relative, record)
        return record

    monkeypatch.setattr(ws, "complete_remote_unit", complete)
    transport = SimpleNamespace(stop_controller=controller, read_json=lambda path: {"wall_seconds": 10})
    state, state_path, ready = {"completed": []}, case.state_root / "g2_state.json", {"source": case.source}
    options = {name: f"/workspace/campaign/inputs/{name}" for name in ("union_fasta", "corpus_table",
               "holdout_receipt", "official_split_dir", "identity_work", "hf_home", "esmc_weights")}
    options.update(cpu_cap_prefix=["cpu-cap", "--"], forecast_seconds=100, unit_timeout_seconds=90)

    def prepare(current, transport, *_args):
        return {"pretrain_request": options, "max_download_bytes": current["g2_max_download_bytes"]}

    check = lambda: controller.session(session["session_id"])  # noqa: E731
    manifest = f"{config['remote_campaign']}/execution_manifest.json"
    ws._prepare_g2_v5(config, transport, state, state_path, session, ready, manifest, probe_path, check, prepare)
    ws._execute_queue(config, transport, state, state_path, session, ready, manifest, probe_path, check)
    queue = ws.measurement_queue(case.source)
    assert admitted == ["pretrain-prepare"] + [action for action, _ in queue]
    assert state["completed"] == [name for _, name in queue]
    for action in ("pretrain-prepare", "timing-unit"):
        assert {key: requests[action][key] for key in TIMING} == {key: session[key] for key in TIMING}

    # The binding still refuses a session whose hard stop, deadline or duration disagree.
    pretrain = requests["pretrain-prepare"]
    for change in ({"hard_stop_unix": None}, {"hard_stop_unix": session["deadline_unix"]},
                   {"deadline_unix": session["hard_stop_unix"]},
                   {"max_total_seconds": session["max_total_seconds"] + 600}):
        changed = {key: value for key, value in {**pretrain, **change}.items() if value is not None}
        with pytest.raises(phase3.Phase3Error, match="hard stop"):
            phase3._runtime(changed, pretraining=True)
    controller.stop()


@pytest.mark.parametrize("problem", ["stale_quote_at_create", "other_active_gpu", "pod_name_collision"])
def test_refusal_before_any_launch_intent_closes_the_window_as_failed(launch, problem):
    case = launch
    if problem == "stale_quote_at_create":
        # Fresh at the preflight, stale once the retained-VM status check returns.
        case.configure(max_age=2, gcp_delay=3)
        match = "stale"
    elif problem == "other_active_gpu":
        case.configure(pods=[{"id": "foreign", "name": "foreign", "gpuCount": 1,
                              "desiredStatus": "RUNNING", "runtimeStatus": "running"}])
        match = "Another Runpod GPU"
    else:
        case.configure(pods=[{"id": "oldpod1", "name": POD_NAME, "gpuCount": 1,
                              "desiredStatus": "EXITED", "runtimeStatus": "stopped"}])
        match = "Pod name already exists"
    with pytest.raises(rc.RunpodError, match=match):
        case.run("w1")
    assert case.preflights == [True]
    state = json.loads(case.driver_state.read_text())
    assert state["start_attempted"] and state["closed"] and state["provider_stop_verified"]
    assert state["stop_status"]["status"] == "NO_CREATE_ATTEMPT" and state["stop_status"]["launch_intent_absent"]
    assert v5.verified_terminal_state(state) == "not_allocated"
    assert not case.session_path.exists() and not (case.state_root / "runpod_config.snapshot.json").exists()
    assert not {"create", "stop", "delete"} & set(provider_calls(case))
    [row] = ledger_rows(case)
    assert (row["window_id"], row["status"], row["provider_terminal"]) == ("w1", "failed", "not_allocated")
    assert row["closeout_sha256"] == sha(case.driver_state)
    assert case.ledger.status()["unresolved_window"] is None
    with pytest.raises(ws.WorkstationError, match="already closed"):
        case.run("w1")
    assert [row["status"] for row in ledger_rows(case)] == ["failed"]


def test_open_window_with_aged_quote_recovers_verifies_and_stops_its_owned_pod(launch):
    case = launch
    case.configure(max_age=4)
    # The first coordinator reserved the window, created its pod and recorded the session, then died.
    case.ledger.begin_window("w1", config_sha256=sha(case.config_path),
                             execution_manifest_sha256=case.source["execution_manifest_sha256"],
                             pass_root=case.state_root)
    session = rc.RunpodController(case.config, case.config_path).start()
    write(case.driver_state, {"config_sha256": sha(case.config_path), "completed": [], "g2_max_download_bytes": GIB,
                              "launch_binding": case.binding("w1"), "start_attempted": True, **session})
    observed = rc._epoch(case.config["runpod"]["quote"]["observed_at"])
    while time.time() - observed <= 4.5:
        time.sleep(.1)
    with pytest.raises(rc.RunpodError, match="stale"):
        rc.validate_config(case.config)
    with pytest.raises(LookupError, match="synthetic unreachable pod"):
        case.run("w1", transport=UnreachablePod())
    assert case.preflights == [False]
    # Recovery verified the owned provider session before reaching the pod, then stopped it.
    assert (case.state_root / "ssh_config").is_file()
    state = json.loads(case.driver_state.read_text())
    assert state["closed"] and state["stop_status"]["status"] == "EXITED"
    assert state["stop_status"]["pod_id"] == session["pod_id"] == state["pod_id"]
    assert json.loads(case.session_path.read_text())["phase"] == "stopped"
    api = json.loads(case.api.read_text())
    assert api["create_calls"] == 1 and set(api["stopped_ids"]) == {session["pod_id"]}
    [row] = ledger_rows(case)
    assert (row["status"], row["provider_terminal"]) == ("failed", "verified_stopped")


def test_new_window_with_stale_quote_is_refused_before_reservation(launch):
    case = launch
    case.configure(quote_age=1000)
    with pytest.raises(rc.RunpodError, match="stale"):
        case.run("w1")
    assert case.preflights == [True]
    assert ledger_rows(case) == [] and not provider_calls(case)
    assert not case.driver_state.exists() and not case.session_path.exists()


def test_open_window_with_another_binding_keeps_the_fresh_quote_requirement(launch):
    case = launch
    case.configure(quote_age=1000)
    case.ledger.begin_window("w1", config_sha256="0" * 64,
                             execution_manifest_sha256=case.source["execution_manifest_sha256"],
                             pass_root=case.state_root)
    with pytest.raises(rc.RunpodError, match="stale"):
        case.run("w1")
    assert case.preflights == [True] and not provider_calls(case)
    assert [row["status"] for row in ledger_rows(case)] == ["open"]


@pytest.mark.parametrize("search_seconds", [0, 60])
def test_recovery_needing_a_create_refuses_the_stale_quote_and_closes_failed(launch, search_seconds, monkeypatch):
    case = launch
    if search_seconds:
        # A stock search's create admission checks live price and billing, never quote age.
        case.config["runpod"].update(stock_search_seconds=search_seconds, stock_poll_interval_seconds=30,
                                     data_center_id="EU-RO-1")
        monkeypatch.setattr(rc.RunpodController, "_catalog_call",
                            lambda *args, **kwargs: pytest.fail("A stale recovered window reached a stock-search read."))
    case.configure(quote_age=1000)
    # The first coordinator reserved the window and recorded its binding, then died before start.
    case.ledger.begin_window("w1", config_sha256=sha(case.config_path),
                             execution_manifest_sha256=case.source["execution_manifest_sha256"],
                             pass_root=case.state_root)
    write(case.driver_state, {"config_sha256": sha(case.config_path), "completed": [], "g2_max_download_bytes": GIB,
                              "launch_binding": case.binding("w1")})
    with pytest.raises(rc.RunpodError, match="stale"):
        case.run("w1")
    assert case.preflights == [False]
    # Refused before the start intent: no provider call, no stock search, no controller session.
    state = json.loads(case.driver_state.read_text())
    assert not state.get("start_attempted") and v5.verified_terminal_state(state) == "not_allocated"
    assert not provider_calls(case) and not json.loads(case.api.read_text()).get("create_calls")
    assert not case.session_path.exists() and not (case.state_root / "stock_search.json").exists()
    [row] = ledger_rows(case)
    assert (row["status"], row["provider_terminal"]) == ("failed", "not_allocated")
    # The window is no longer open, so a re-run is a new launch and needs a fresh quote first.
    with pytest.raises(rc.RunpodError, match="stale"):
        case.run("w1")
    assert case.preflights == [False, True] and len(ledger_rows(case)) == 1 and not provider_calls(case)


def test_closed_driver_record_of_an_open_window_is_recorded_without_provider_access(launch):
    case = launch
    case.configure(quote_age=1000)
    # The coordinator wrote its closed driver record, then died before recording the window outcome.
    case.ledger.begin_window("w1", config_sha256=sha(case.config_path),
                             execution_manifest_sha256=case.source["execution_manifest_sha256"],
                             pass_root=case.state_root)
    closed = {"config_sha256": sha(case.config_path), "completed": [], "launch_binding": case.binding("w1"),
              "start_attempted": True, "closed": True, "provider_stop_verified": True,
              "stop_status": {"status": "NO_CREATE_ATTEMPT", "launch_intent_absent": True}}
    write(case.driver_state, closed)
    assert case.run("w1")["closed"] is True
    assert case.preflights == [] and not provider_calls(case)
    [row] = ledger_rows(case)
    assert (row["status"], row["provider_terminal"]) == ("failed", "not_allocated")
    assert row["closeout_sha256"] == sha(case.driver_state)
    # A closed record of another window, or one without verified terminal evidence, is never used.
    for record in ({**closed, "launch_binding": case.binding("w0")}, {**closed, "stop_status": {}}):
        write(case.ledger.path, v5.empty_failure_ledger(case.ledger.path, "f" * 64))
        case.ledger.begin_window("w1", config_sha256=sha(case.config_path),
                                 execution_manifest_sha256=case.source["execution_manifest_sha256"],
                                 pass_root=case.state_root)
        write(case.driver_state, record)
        with pytest.raises(ws.WorkstationError, match="already closed"):
            case.run("w1")
        assert [row["status"] for row in ledger_rows(case)] == ["open"] and not provider_calls(case)


def test_prepared_preflight_relaxes_only_quote_age_and_only_for_runpod(launch, monkeypatch):
    case = launch
    case.configure(quote_age=1000)
    config = {**case.config, "artifact_cache": {"synthetic": True}}
    path = write(case.state_root / "cached-config.json", config)
    write(case.state_root / "prepared_config.json", config)
    write(case.state_root / "prepared_inputs.json", {"status": "ready", "source": case.source,
                                                      "config_sha256": sha(case.state_root / "prepared_config.json")})
    for name in ("_validate_runpod_structure", "_validate_published_runtime_locks", "_local_storage"):
        monkeypatch.setattr(ws, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(ws, "_execution_source", lambda *args, **kwargs: case.source)
    with pytest.raises(rc.RunpodError, match="stale"):
        case.real_prepared_preflight(config, path, published=True)
    result = case.real_prepared_preflight(config, path, published=True, fresh=False)
    assert result["input_proof_reused"] and result["cost"]["quote_observed_at"] == config["runpod"]["quote"]["observed_at"]
    with pytest.raises(ws.WorkstationError, match="recovering Runpod launch window"):
        ws.preflight({"provider": "gcp"}, path, fresh=False)
