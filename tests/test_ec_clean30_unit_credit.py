"""Synthetic credit, persistence and idempotent dispatch tests. Does not use account credentials."""
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from ec_clean30 import runpod_controller as rc, workstation as ws
from ec_clean30_fold0_fixtures import cached_driver  # noqa: F401
from test_ec_clean30_workstation import LocalTransport, cycle  # noqa: F401


NOW = 1_800_000_000


@pytest.fixture
def money():
    utc = lambda n: datetime.fromtimestamp(n, timezone.utc).isoformat()
    settings = {
        "absolute_hard_stop_unix": NOW + 200_000, "gross_ceiling_usd": 60,
        "quote": {"hourly_compute_usd": .89, "running_storage_hourly_usd": .014247,
                  "margin_fraction": .15, "egress_reserve_usd": 2, "stopped_storage_reserve_usd": .3,
                  "prior_charge_usd": 5},
        "unit_credit": {"approved_from_unix": NOW-3600, "approved_until_unix": NOW+200_000,
                        "override_scope": "synthetic EC windows", "override_evidence_reference": "fixture",
                        "project_ceiling_usd": 240, "project_incurred_before_window_usd": 18, "reported_prior_spend_usd": 18,
                        "other_project_reserved_usd": 2, "opening_balance_floor_usd": 50,
                        "other_account_hourly_upper_usd": .003, "other_account_fixed_upper_usd": .01,
                        "unposted_before_window_upper_usd": .1, "rolling_before_window_upper_usd": .5}}
    policy = {"currency": "USD", "window": {"seconds": 86400}, "accounting": {"billing_max_age_seconds": 300},
              "default_limit_usd": 10, "warning_fraction": .8, "overrides": {"approved_overrides": [
                  {"evidence_reference": "fixture", "scope": "synthetic EC windows", "approved_limit_usd": 40,
                   "approved_at": utc(NOW-3600), "expires_at": utc(NOW+200_000),
                   "absolute_stop_boundary": utc(NOW+200_000)}]}}
    account = {"pods": [{"id": "p"}], "balance_usd": 80, "billing_25h_usd": .4, "spend_per_hour_usd": .89}
    baseline = {"observed_unix": NOW-300, "balance_usd": 80}
    return settings, policy, account, baseline


def decide(money, **args):
    return rc._window_credit(*money, **dict({"observed": NOW-1, "now": NOW, "pod_id": "p",
                                              "reserve_seconds": 10800}, **args))


def test_projection_and_prior_reservations(money):
    ans = decide(money)
    assert ans["approved"] and 25 < ans["project_gross_upper_usd"] < 27
    assert ans["unit_and_closeout_usd"] > 5
    # A successor first reconciles both lanes and carries all prior costs forward.
    money[0]["unit_credit"]["project_incurred_before_window_usd"] += 52
    assert decide(money)["project_gross_upper_usd"] == pytest.approx(ans["project_gross_upper_usd"] + 52)


@pytest.mark.parametrize("group,key,value,reason", [
    ("g","project_incurred_before_window_usd",235,"Project ceiling"),
    ("g","project_incurred_before_window_usd",0,"cannot be reset"),
    ("g","other_project_reserved_usd",220,"Project ceiling"),
    ("g","approved_until_unix",NOW+100,"funding window"),
    ("g","override_evidence_reference","wrong","recorded rolling override"),
    ("a","balance_usd",1,"debit exceeds"),
    ("a","balance_usd",float("nan"),"balance must be finite"),
    ("a","billing_25h_usd",12,"bill exceeds"),
    ("a","spend_per_hour_usd",10,"Unexpected account spend"),
    ("a","pods",[{"id":"p"},{"id":"foreign"}],"unexpected allocation"),
    ("p","warning_fraction",.05,"Rolling warning"),
])
def test_closes(group, key, value, reason, money):
    {"g":money[0]["unit_credit"],"a":money[2],"p":money[1]}[group][key] = value
    with pytest.raises(rc.RunpodError, match=reason):
        decide(money)


def test_unfunded_or_expired_rolling_limit(money):
    money[0]["unit_credit"].pop("override_evidence_reference")
    with pytest.raises(rc.RunpodError, match="Rolling warning"):
        decide(money, reserve_seconds=22*3600)
    money[0]["unit_credit"]["override_evidence_reference"] = "fixture"
    money[1]["overrides"]["approved_overrides"][0]["expires_at"]="2020-01-01T00:00:00Z"
    with pytest.raises(rc.RunpodError, match="does not cover closeout"):
        decide(money)


def test_stale_and_insufficient_charge_including_shutdown(money):
    with pytest.raises(rc.RunpodError, match="stale"):
        decide(money, observed=NOW-301)
    money[2]["balance_usd"] = money[3]["balance_usd"] = 4
    with pytest.raises(rc.RunpodError, match="Insufficient credit"):
        decide(money)


def test_create_envelope_and_balance_floor(money):
    money[2].update(pods=[], spend_per_hour_usd=0)
    receipt = decide(money, pod_id=None, reserve_seconds=172800)
    assert receipt["project_gross_upper_usd"] < 90
    money[2]["balance_usd"] = money[3]["balance_usd"] = 49
    with pytest.raises(rc.RunpodError, match="opening floor"):
        decide(money, pod_id=None)


@pytest.mark.parametrize("with_chain", [True, False])
def test_new_v5_dispatch_cannot_omit_credit_or_chain(cycle, with_chain):
    request, durable, state, options = cycle
    request.update(provider="runpod", unit_id="first", pod_id="p")
    if with_chain:
        request["expected_chain"] = ["synthetic"]
    transport = LocalTransport()
    owner = SimpleNamespace(config={"execution_scope": "g2_then_fold0_v5"}, cfg={},
                            cost={"stop_reserve_seconds": 1800}, _unchanged=lambda: None)
    owner.check_window_funds = rc.RunpodController.check_window_funds.__get__(owner)
    transport.stop_controller = owner
    transport.restored_completions = {}
    host = ws.HostPull(transport, durable, state, max_download_bytes=100000, deadline_unix=time.time() + 600)
    if with_chain:
        with pytest.raises(rc.RunpodError, match="financial authorization"):
            ws.complete_remote_unit(transport, host, request, **options)
    else:
        with pytest.raises(ws.WorkstationError, match="amendment chain"):
            ws.complete_remote_unit(transport, host, request, **options)
    assert transport.started == 0
    assert not Path(options["remote_request"] + ".job/command.json").exists()


def test_disappearing_v5_job_is_never_relaunched_on_reattachment(cycle):
    request, durable, state, options = cycle
    request.update(provider="runpod", expected_chain=["synthetic"], unit_id="first", pod_id="p")
    job = Path(options["remote_request"] + ".job")
    marker = job / "command.json"
    job.mkdir(parents=True)
    marker.write_text(json.dumps({
        "argv": [request["python"], options["runner"], options["action"], "--request", options["remote_request"]],
        "cwd": str(Path(options["runner"]).parent.parent),
    }))

    class DisappearingJobTransport(LocalTransport):
        def read_json(self, path, *, optional=False):
            record = super().read_json(path, optional=optional)
            if optional and Path(path) == marker and record is not None:
                marker.unlink()
                job.rmdir()
            return record

    transport = DisappearingJobTransport()
    transport.stop_controller = SimpleNamespace(config={"execution_scope": "g2_then_fold0_v5"},
                                               cfg={"unit_credit": {}},
                                               check_window_funds=lambda **kw: pytest.fail("Recovery re-admitted a job"))
    transport.restored_completions = {}
    host = ws.HostPull(transport, durable, state, max_download_bytes=100000, deadline_unix=time.time() + 600)
    with pytest.raises(ws.WorkstationError, match="existing remote job"):
        ws.complete_remote_unit(transport, host, request, **options)
    assert transport.started == 0 and not job.exists()


def test_credit_refusal_after_first_fit_preserves_and_recovers(cycle, tmp_path):
    request, durable, state, options = cycle
    request.update(provider="runpod", expected_chain="synthetic", unit_id="first", pod_id="p")
    transport = LocalTransport(lost_launch=True)
    checks = []
    def guard(**kw):
        checks.append(kw["unit"])
        if kw["unit"] != "first":
            raise rc.RunpodError("Insufficient credit")
    transport.stop_controller = SimpleNamespace(config={"execution_scope": "g2_then_fold0_v5"},
                                               cfg={"unit_credit": {}}, check_window_funds=guard,
                                               cost={"stop_reserve_seconds": 1800})
    transport.restored_completions = {}
    mirror = tmp_path/"mirror"
    host = ws.HostPull(transport, durable, state, max_download_bytes=100000, deadline_unix=time.time()+600,
                       mirror_root=mirror)
    assert ws.complete_remote_unit(transport, host, request, **options)["status"] == "passed"
    preserved = "runs/fixture/complete.json"
    original = (durable / preserved).read_bytes()
    next_options = dict(options, remote_request=str(tmp_path/"vm/queue/next.json"))
    with pytest.raises(rc.RunpodError, match="Insufficient credit"):
        ws.complete_remote_unit(transport, host, dict(request, unit_id="second"), **next_options)
    assert transport.started == 1
    assert not Path(next_options["remote_request"]+".job/command.json").exists()
    assert (durable/preserved).read_bytes() == (mirror/preserved).read_bytes() == original
    # Retry of accepted request after a lost response reads its existing command and seal; no new fit or credit.
    assert ws.complete_remote_unit(transport, host, request, **options)["status"] == "passed"
    assert transport.started == 1 and checks == ["first","second"]


def test_missing_reader_and_changed_policy_stop_before_execution(money, tmp_path):
    cfg, policy, account, base = money
    reader = tmp_path/"read.py"; reader.write_text("raise SystemExit(1)")
    policy_file = tmp_path/"policy.json"; policy_file.write_text(json.dumps(policy))
    cfg["unit_credit"].update(policy_path=str(policy_file), policy_sha256=rc._sha(policy_file),
                              account_reader=str(reader), account_reader_sha256=rc._sha(reader))
    obj = SimpleNamespace(cfg=cfg, root=tmp_path, config_sha="pin", host_try=1, _unchanged=lambda:None)
    check = lambda: rc.RunpodController.check_window_funds(obj, reserve_seconds=7200)
    with pytest.raises(rc.RunpodError, match="Cannot verify"):
        check()
    policy_file.write_text("{}")
    with pytest.raises(rc.RunpodError, match="policy changed"):
        check()
    assert not (tmp_path/"unit_credit_opening.json").exists()



def test_closeout_after_credit_refusal_records_prior_units(cached_driver, monkeypatch, tmp_path):
    d = cached_driver
    monkeypatch.setattr(ws, "_stage_runpod_runtime", lambda *args: None)
    def queue(config, transport, state, *args):
        state["completed"] = ["accepted-first-unit"]
        raise rc.RunpodError("Insufficient credit for the next unit")
    monkeypatch.setattr(ws, "_execute_queue", queue)
    proof = tmp_path / "preservation.json"
    proof.write_text('{"synthetic":true}')
    monkeypatch.setattr(ws, "_preserve_runpod_session_outputs", lambda *args: d.events.append("preserve") or proof)
    def close(path):
        assert ws._sha(path) == ws._sha(proof)
        d.events.append("delete")
        return {"status":"DELETED", "provider_absence_verified":True, "preservation_sha256":ws._sha(proof)}
    d.controller.closeout = close
    with pytest.raises(rc.RunpodError, match="Insufficient credit"):
        ws._run_runpod_g2(d.config, d.path, transport=d.transport, controller=d.controller)
    state = json.loads((Path(d.config["state_root"])/"driver_state.json").read_text())
    assert state["status"] == "interrupted" and state["provider_stop_verified"] and state["closed"]
    assert state["completed"] == ["accepted-first-unit"] and state["closeout"]["provider_absence_verified"]
    assert d.events.index("preserve") < d.events.index("stop") < d.events.index("delete")


def test_bound_reader_is_called_on_every_new_admission(money, tmp_path, monkeypatch):
    cfg, policy, account, base = money
    cfg["unit_credit"]["initial_unit_and_closeout_seconds"] = 7200
    monkeypatch.setattr(rc.time, "time", lambda: NOW)
    reader = tmp_path/"read.py"
    account_file = tmp_path/"account.json"
    reader.write_text(f"from pathlib import Path\nprint(Path({str(account_file)!r}).read_text())\n")
    policy_path = tmp_path/"policy.json"
    policy_path.write_text(json.dumps(policy))
    cfg["unit_credit"].update(account_reader=str(reader), account_reader_sha256=rc._sha(reader),
                              policy_path=str(policy_path), policy_sha256=rc._sha(policy_path))
    cfg["pod_name"] = "synthetic-window"
    ledger_path = tmp_path / "prior-ledger.json"
    ledger_path.write_text(json.dumps({
        "schema": "deepmzyme-project-credit-ledger-v1", "next_pod_name": cfg["pod_name"],
        "charges": {"metal": [], "ec": [{"resource": "fixture", "from_unix": 1, "until_unix": 2,
                                        "gross_usd": cfg["unit_credit"]["project_incurred_before_window_usd"]}]}}))
    cfg["unit_credit"].update(prior_ledger_path=str(ledger_path), prior_ledger_sha256=rc._sha(ledger_path))
    owner = rc.RunpodController.__new__(rc.RunpodController)
    owner.cfg, owner.root, owner.config_sha, owner.host_try = cfg, tmp_path, "pin", 1
    owner.state_path = tmp_path / "session.json"
    owner._cached_state = None
    owner._unchanged = lambda: None
    owner.state_path.write_text(json.dumps({"session_id": "s", "config_sha256": "pin", "phase": "creating"}))

    check = lambda **kwargs: rc.RunpodController.check_window_funds(owner, **kwargs)
    account_file.write_text(json.dumps(dict(account, pods=[], spend_per_hour_usd=0)))
    assert check(reserve_seconds=172800)["approved"]
    state = json.loads(owner.state_path.read_text())
    state.update(pod_id="p", phase="running")
    owner.state_path.write_text(json.dumps(state))
    account_file.write_text(json.dumps(account))
    assert check(reserve_seconds=10800, pod_id="p", unit="first")["unit_id"]=="first"
    account_file.write_text(json.dumps(dict(account,balance_usd=2)))
    with pytest.raises(rc.RunpodError, match="debit exceeds"):
        check(reserve_seconds=10800, pod_id="p", unit="second")
    assert json.loads((tmp_path/"unit_credit_last.json").read_text())["unit_id"]=="first"
