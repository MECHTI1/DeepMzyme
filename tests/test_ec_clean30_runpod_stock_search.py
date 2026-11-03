"""Bounded stock-search checks; all provider calls use the local fake CLI."""
import json
from types import SimpleNamespace

import pytest

from test_ec_clean30_runpod_controller import fixture, CAPACITY_REJECTION, rc


@pytest.mark.parametrize("seconds", [-1, 1801, True, "1800", float("inf")])
def test_stock_search_window_cannot_exceed_authority(fixture, seconds):
    config, _, _, _, _ = fixture
    config["runpod"]["stock_search_seconds"] = seconds
    config["runpod"]["stock_poll_interval_seconds"] = 45
    with pytest.raises(rc.RunpodError):
        rc.validate_config(config)


@pytest.mark.parametrize("interval", [0, 29, 46, True, "45", float("nan")])
def test_stock_search_poll_interval_is_bounded(fixture, interval):
    config, _, _, _, _ = fixture
    config["runpod"]["stock_search_seconds"] = 1800
    config["runpod"]["stock_poll_interval_seconds"] = interval
    with pytest.raises(rc.RunpodError):
        rc.validate_config(config)


@pytest.mark.parametrize("interval", [30, 45])
def test_requested_stock_search_configuration_is_accepted(fixture, interval):
    config, _, _, _, _ = fixture
    config["runpod"]["data_center_id"] = "EU-RO-1"
    config["runpod"]["stock_search_seconds"] = 1800
    config["runpod"]["stock_poll_interval_seconds"] = interval
    assert rc.validate_config(config)["status"] == "ready"


def stock_record():
    return {"availability": "LOW", "id": "NVIDIA GeForce RTX 4090", "secure": True,
            "dataCenters": [{"id": "EU-RO-1", "availability": "LOW"}],
            "cudaVersions": [{"version": "13.2", "available": True}],
            "price": {"secure": .89}}


def search_controller(fixture):
    config, _, _, _, make = fixture
    config["runpod"].update(data_center_id="EU-RO-1", stock_search_seconds=1800,
                           stock_poll_interval_seconds=45)
    return make()


def test_stock_read_uses_secure_exact_region_cuda_and_price(fixture, monkeypatch):
    controller = search_controller(fixture)
    calls = []

    def catalog(path, query):
        calls.append((path, query))
        return stock_record()

    monkeypatch.setattr(controller, "_catalog_call", catalog)
    assert controller._stock_available() is True
    path, query = calls[0]
    assert path.endswith("NVIDIA%20GeForce%20RTX%204090")
    assert query["cloud"] == "SECURE" and query["product"] == "POD"
    assert query["countryCodes"] == "RO" and query["minCudaVersion"] == "13.0"
    assert (controller.root / "stock_observation.json").is_file()


@pytest.mark.parametrize("change", ["other_dc", "dc_empty", "unavailable", "cuda_low", "cuda_false"])
def test_stock_elsewhere_or_incompatible_cuda_never_admits(fixture, monkeypatch, change):
    controller = search_controller(fixture)
    record = stock_record()
    if change == "other_dc":
        record["dataCenters"][0]["id"] = "US-CA-1"
    elif change == "dc_empty":
        record["dataCenters"] = []
    elif change == "unavailable":
        record["dataCenters"][0]["availability"] = "NONE"
    elif change == "cuda_low":
        record["cudaVersions"][0]["version"] = "12.8"
    else:
        record["cudaVersions"][0]["available"] = False
    monkeypatch.setattr(controller, "_catalog_call", lambda *args: record)
    assert controller._stock_available() is False


def test_live_price_increase_blocks_search_admission(fixture, monkeypatch):
    controller = search_controller(fixture)
    record = stock_record()
    record["price"]["secure"] = .90
    monkeypatch.setattr(controller, "_catalog_call", lambda *args: record)
    with pytest.raises(rc.RunpodError, match="price|quote|rate"):
        controller._stock_available()


def mock_catalog(controller, monkeypatch):
    monkeypatch.setattr(controller, "_catalog_call", lambda path, query:
        {"metadata": {"totals": {"totalAmount": .1}}} if path == "/v2/billing" else stock_record())


def test_capacity_rejection_retries_without_rebasing_allocation(fixture, monkeypatch):
    _, _, api, _, _ = fixture
    controller = search_controller(fixture)
    mock_catalog(controller, monkeypatch)
    original = controller._call
    attempts = []
    deadlines = []

    def call(*args, **kwargs):
        if args[0] == "create":
            attempts.append(args)
            state = controller._state()
            deadlines.append(tuple(state[k] for k in ("allocation_started_unix", "deadline_unix", "hard_stop_unix")))
            if len(attempts) == 1:
                raise rc.RunpodError(CAPACITY_REJECTION)
        return original(*args, **kwargs)

    real_sleep = rc.time.sleep
    monkeypatch.setattr(controller, "_call", call)
    monkeypatch.setattr(rc.time, "sleep", lambda seconds: real_sleep(seconds) if seconds < 1 else None)
    session = controller.start()
    assert session["pod_id"] == "ownedpod123" and len(attempts) == 2
    assert deadlines[0] == deadlines[1]
    assert json.loads(api.read_text())["create_calls"] == 1
    assert (controller.root / "create_rejections.jsonl").is_file()
    assert controller._state()["create_attempt_count"] == 2


def test_uncertain_second_post_never_retries_or_reuses_stale_capacity_error(fixture, monkeypatch):
    controller = search_controller(fixture)
    mock_catalog(controller, monkeypatch)
    original = controller._call
    attempts = []

    def call(*args, **kwargs):
        if args[0] == "create":
            attempts.append(args)
            if len(attempts) == 1:
                raise rc.RunpodError(CAPACITY_REJECTION)
            raise rc.RunpodError("Runpod create result is uncertain (TimeoutExpired).")
        return original(*args, **kwargs)

    real_sleep = rc.time.sleep
    monkeypatch.setattr(controller, "_call", call)
    monkeypatch.setattr(rc.time, "sleep", lambda seconds: real_sleep(seconds) if seconds < 1 else None)
    with pytest.raises(rc.RunpodError, match="uncertain"):
        controller.start()
    state = controller._state()
    assert len(attempts) == 2 and state["stop_requested"]
    assert state["phase"] == "closing_after_failure"
    assert "uncertain" in state["create_error"]
    assert not state.get("stop_receipt", {}).get("provider_absence_verified")


def test_billing_over_reserved_prior_charge_blocks_before_any_intent(fixture, monkeypatch):
    controller = search_controller(fixture)
    monkeypatch.setattr(controller, "_catalog_call", lambda path, query:
        {"metadata": {"totals": {"totalAmount": .61}}} if path == "/v2/billing" else stock_record())
    with pytest.raises(rc.RunpodError, match="billing|reserve"):
        controller.start()
    assert not controller.state_path.exists()


def test_search_deadline_survives_reentry_without_reset(fixture, monkeypatch):
    controller = search_controller(fixture)
    now = [1000.0]
    observations = []
    monkeypatch.setattr(rc, "time", SimpleNamespace(time=lambda: now[0],
        sleep=lambda seconds: now.__setitem__(0, now[0] + seconds)))
    monkeypatch.setattr(controller, "_stock_available", lambda: observations.append(now[0]) or False)
    with pytest.raises(rc.RunpodError, match="exhausted"):
        controller._await_stock()
    first = json.loads((controller.root / "stock_search.json").read_text())
    assert first["started_unix"] == 1000 and first["deadline_unix"] == 2800
    count = len(observations)
    assert count == 40 and not controller.state_path.exists()
    with pytest.raises(rc.RunpodError, match="exhausted"):
        controller._await_stock()
    assert json.loads((controller.root / "stock_search.json").read_text()) == first
    assert len(observations) == count


def test_capacity_error_with_reconciled_pod_never_creates_again(fixture, monkeypatch):
    controller = search_controller(fixture)
    mock_catalog(controller, monkeypatch)
    original = controller._call
    attempts = []

    def call(*args, **kwargs):
        if args[0] == "create":
            attempts.append(args)
            original(*args, **kwargs)
            raise rc.RunpodError(CAPACITY_REJECTION)
        return original(*args, **kwargs)

    monkeypatch.setattr(controller, "_call", call)
    assert controller.start()["pod_id"] == "ownedpod123"
    assert len(attempts) == 1


def test_other_active_gpu_after_rejection_blocks_retry(fixture, monkeypatch):
    controller = search_controller(fixture)
    mock_catalog(controller, monkeypatch)
    original = controller._call
    attempts = []

    def call(*args, **kwargs):
        if args[0] == "create":
            attempts.append(args)
            raise rc.RunpodError(CAPACITY_REJECTION)
        if args[0] == "list" and attempts:
            return [{"id": "foreignpod", "name": "unrelated", "gpuCount": 1,
                     "desiredStatus": "RUNNING", "runtimeStatus": "running"}]
        return original(*args, **kwargs)

    monkeypatch.setattr(controller, "_call", call)
    with pytest.raises(rc.RunpodError, match="Another Runpod GPU"):
        controller.start()
    assert len(attempts) == 1


def test_positive_poll_reaches_provider_without_redundant_stock_read(fixture, monkeypatch):
    _, _, api, _, _ = fixture
    controller = search_controller(fixture)
    reads = []
    def catalog(path, query):
        if path == '/v2/billing':
            return {'metadata': {'totals': {'totalAmount': .1}}}
        reads.append(path)
        assert len(reads) == 1, 'Positive stock must be consumed without another catalog request'
        return stock_record()
    monkeypatch.setattr(controller, '_catalog_call', catalog)
    assert controller.start()['pod_id'] == 'ownedpod123'
    assert json.loads(api.read_text())['create_calls'] == 1
    controller.stop()


def test_missing_positive_poll_cannot_admit_create(fixture):
    controller = search_controller(fixture)
    with pytest.raises((rc.RunpodError, OSError)):
        controller._search_admission()
