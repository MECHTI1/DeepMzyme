"""Project charges and opening balances remain bound across admissions and restarts."""
import json

import pytest

from ec_clean30 import runpod_controller as rc
from test_ec_clean30_unit_credit import NOW, money


def charge(resource, start, end, cost):
    return {"resource": resource, "from_unix": start, "until_unix": end, "gross_usd": cost}


def reviewed_ledger(tmp_path, cfg, charges):
    cfg["pod_name"] = "fixture-next"
    path = tmp_path / "reconciled.json"
    path.write_text(json.dumps({"schema": "deepmzyme-project-credit-ledger-v1",
                                "next_pod_name": cfg["pod_name"], "charges": charges}))
    cfg["unit_credit"].update(prior_ledger_path=str(path), prior_ledger_sha256=rc._sha(path))
    return path


def test_prior_charges_carry_both_lanes_to_the_named_successor(tmp_path):
    cfg = {"unit_credit": {"project_incurred_before_window_usd": 1.3}}
    path = reviewed_ledger(tmp_path, cfg, {"ec": [charge("machine-one", 10, 20, 1.1)],
                                          "metal": [charge("machine-one", 20, 30, 0.2)]})
    assert rc._verify_prior_credit(cfg, NOW) == rc._sha(path)
    cfg["unit_credit"]["project_incurred_before_window_usd"] = 1.1
    with pytest.raises(rc.RunpodError, match="below the reconciled"):
        rc._verify_prior_credit(cfg, NOW)
    cfg["pod_name"] = "fixture-later"
    with pytest.raises(rc.RunpodError, match="another window"):
        rc._verify_prior_credit(cfg, NOW)
    path.write_text("{}")
    with pytest.raises(rc.RunpodError, match="ledger changed"):
        rc._verify_prior_credit(cfg, NOW)


@pytest.mark.parametrize(("charges", "reason"), [
    ({"ec": []}, "both EC and metal"),
    ({"ec": [charge("disk", 5, 9, 1)], "metal": [charge("disk", 8, 10, 1)]}, "overlap"),
    ({"ec": [charge("disk", 5, NOW + 1, 1)], "metal": []}, "charge interval"),
])
def test_ledger_rejects_incomplete_or_duplicated_charges(tmp_path, charges, reason):
    cfg = {"unit_credit": {"project_incurred_before_window_usd": 3}}
    reviewed_ledger(tmp_path, cfg, charges)
    with pytest.raises(rc.RunpodError, match=reason):
        rc._verify_prior_credit(cfg, NOW)


def test_opening_bound_once_to_lifecycle(tmp_path, money, monkeypatch):
    cfg, policy, account, _ = money
    reviewed_ledger(tmp_path, cfg, {"ec": [], "metal": []})
    monkeypatch.setattr(rc.time, "time", lambda: NOW)
    reader, policy_file = tmp_path / "reader.py", tmp_path / "policy.json"
    live = tmp_path / "live.json"
    reader.write_text(f"from pathlib import Path\nprint(Path({str(live)!r}).read_text())\n")
    policy_file.write_text(json.dumps(policy))
    cfg["unit_credit"].update(initial_unit_and_closeout_seconds=1200, account_reader=str(reader),
                              account_reader_sha256=rc._sha(reader), policy_path=str(policy_file),
                              policy_sha256=rc._sha(policy_file))
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg))
    state_path = tmp_path / "session.json"

    def owner():
        controller = rc.RunpodController.__new__(rc.RunpodController)
        controller.cfg = controller.config = cfg
        controller.config_path, controller.source_sha = cfg_path, rc._sha(cfg_path)
        controller.root, controller.state_path = tmp_path, state_path
        controller.config_sha, controller.host_try = rc._digest(cfg), 1
        return controller

    rc._write(state_path, {"phase": "creating", "session_id": "session-one", "config_sha256": rc._digest(cfg)})
    live.write_text(json.dumps(dict(account, pods=[], spend_per_hour_usd=0)))
    first = owner().check_window_funds(reserve_seconds=1200)
    opening = tmp_path / "unit_credit_opening.json"
    original = opening.read_bytes()
    assert first["opening_sha256"] == rc._sha(opening) == rc._read(state_path)["unit_credit_opening_sha256"]
    live.write_text(json.dumps(dict(account, pods=[], spend_per_hour_usd=0, balance_usd=85)))
    owner().check_window_funds(reserve_seconds=1200)
    assert opening.read_bytes() == original
    owner()._update(pod_id="p", phase="allocated")
    live.write_text(json.dumps(account))
    assert owner().check_window_funds(reserve_seconds=1200, pod_id="p")["opening_sha256"] == first["opening_sha256"]
    opening.write_text(json.dumps(dict(json.loads(original), balance_usd=99)))
    with pytest.raises(rc.RunpodError, match="lifecycle pin"):
        owner().check_window_funds(reserve_seconds=1200, pod_id="p")


def test_opening_cannot_use_future_fees_to_cover_prior_billing(money):
    cfg, policy, account, baseline = money
    cfg["unit_credit"].pop("override_evidence_reference")
    baseline["observed_unix"] = NOW
    account.update(pods=[], spend_per_hour_usd=0, billing_25h_usd=3)
    with pytest.raises(rc.RunpodError, match="bill exceeds"):
        rc._window_credit(cfg, policy, account, baseline, observed=NOW, now=NOW, pod_id=None,
                          reserve_seconds=4.25 * 3600)


@pytest.mark.parametrize(("posted", "upper"), [(.4, .5), (.2, .3)])
def test_opening_reserves_unposted_prior_charges_separately(money, posted, upper):
    cfg, policy, account, baseline = money
    # The reviewed prior bound covers both the posted bill and the $0.1 still outstanding.
    cfg["unit_credit"]["rolling_before_window_upper_usd"] = upper
    baseline["observed_unix"] = NOW
    account.update(pods=[], spend_per_hour_usd=0, billing_25h_usd=posted)
    args = dict(observed=NOW, now=NOW, pod_id=None, reserve_seconds=3600)
    assert rc._window_credit(cfg, policy, account, baseline, **args)["approved"]
    account["billing_25h_usd"] = posted + .01
    with pytest.raises(rc.RunpodError, match="bill exceeds"):
        rc._window_credit(cfg, policy, account, baseline, **args)
    # Once running, that same outstanding debit can post without counting it a second time.
    account.update(pods=[{"id": "p"}], billing_25h_usd=upper)
    args["pod_id"] = "p"
    assert rc._window_credit(cfg, policy, account, baseline, **args)["approved"]


def test_in_progress_bill_cannot_consume_future_closeout_reserve(money):
    cfg, policy, account, baseline = money
    baseline["observed_unix"] = NOW - 3600
    # $2 fits only if the as-yet unspent egress/retention reserve can cover the posted bill.
    account["billing_25h_usd"] = 2
    cfg["unit_credit"]["other_account_fixed_upper_usd"] = 1
    with pytest.raises(rc.RunpodError, match="bill exceeds"):
        rc._window_credit(cfg, policy, account, baseline, observed=NOW, now=NOW, pod_id="p",
                          reserve_seconds=3600)


def test_aged_out_prior_bound_cannot_mask_new_bill(money):
    cfg, policy, account, baseline = money
    baseline["observed_unix"] = NOW - 25 * 3600
    cfg["unit_credit"]["rolling_before_window_upper_usd"] = 3
    # Actual recent billing can contain 25 hours of metered charges. It cannot contain pre-window use.
    quote = cfg["quote"]
    rate = (quote["hourly_compute_usd"] + quote["running_storage_hourly_usd"]
            + cfg["unit_credit"]["other_account_hourly_upper_usd"])
    account["billing_25h_usd"] = rate * 25
    args = dict(observed=NOW, now=NOW, pod_id="p", reserve_seconds=3600)
    receipt = rc._window_credit(cfg, policy, account, baseline, **args)
    assert receipt["approved"]
    account["billing_25h_usd"] = 28
    with pytest.raises(rc.RunpodError, match="bill exceeds"):
        rc._window_credit(cfg, policy, account, baseline, **args)
