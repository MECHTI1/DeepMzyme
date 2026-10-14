"""Synthetic authorization, gross ceiling and session-reset regression checks."""
import json
from copy import deepcopy
from pathlib import Path

import pytest

from metal_completion_campaign import CAMPAIGN_ID, sha256_file
from metal_completion_safety import check_authorization, validate_worker_authorization


def packet(tmp_path):
    manifest = {"campaign_id": CAMPAIGN_ID, "authorized_grid_units": ["u"],
                "successor_root": str(tmp_path / "out")}
    policy = {"schema_version": 1, "campaign_id": CAMPAIGN_ID, "execution_authorized": True,
              "manifest_sha256": "hash", "user_decision": "Run the bounded comparison.", "units": ["u"],
              "expires_unix": 30000, "max_sessions": 2, "lanes": 3, "persistence_mode": "host_pull",
              "durable_root": str(tmp_path / "durable"), "max_session_seconds": 10000,
              "snapshot_max_age_seconds": 300, "reserve_usd": 1, "gross_ceiling_usd": 8,
              "estimated_fit_seconds": 1800, "controller_owner": "sole-coordinator",
              "max_total_gpu_seconds": 30000}
    session = {"session_id": "session-1", "instance_id": "instance-1", "state": "RUNNING",
               "allocation_started_unix": 1000, "deadline_unix": 10000, "max_total_seconds": 9000,
               "captured_unix": 2000, "controller_owner": "sole-coordinator",
               "provider_hard_stop_verified": True, "provider_hard_stop_unix": 10000}
    spending = {"session_id": "session-1", "captured_unix": 2000, "gross_running_usd_per_hour": 1,
                "spent_before_session_usd": 2, "storage_and_snapshot_forecast_usd": 1,
                "allocated_seconds_before_session": 9000}
    return manifest, policy, session, spending


def test_full_session_is_reserved(tmp_path):
    m, p, s, cost = packet(tmp_path)
    result = check_authorization(manifest=m, manifest_sha256="hash", policy=p, session=s,
                                 spending=cost, unit_name="u", now=2000)
    assert result["gross_committed_usd"] == 6.5


@pytest.mark.parametrize("part,key,value", [
    (1, "execution_authorized", False), (1, "manifest_sha256", "other"),
    (1, "units", ["v"]), (1, "gross_ceiling_usd", 6), (1, "expires_unix", 1900),
    (1, "max_session_seconds", 8000), (1, "snapshot_max_age_seconds", 901),
    (2, "deadline_unix", 12000), (2, "captured_unix", 1000), (2, "state", "STOPPED"),
    (2, "provider_hard_stop_verified", False), (2, "provider_hard_stop_unix", 11000),
    (2, "provider_hard_stop_unix", 2100),
    (2, "controller_owner", "other"), (3, "gross_running_usd_per_hour", float("nan")),
    (3, "spent_before_session_usd", -1), (3, "session_id", "other"),
    (1, "max_total_gpu_seconds", 17000),
])
def test_invalid_packet_fails(tmp_path, part, key, value):
    values = list(packet(tmp_path)); values[part][key] = value
    with pytest.raises(ValueError):
        check_authorization(manifest=values[0], manifest_sha256="hash", policy=values[1],
                            session=values[2], spending=values[3], unit_name="u", now=2000)


@pytest.mark.parametrize("part", [2, 3])
def test_snapshot_captured_before_allocation_start_is_refused(tmp_path, part):
    values = list(packet(tmp_path))
    values[2]["allocation_started_unix"] = 1900
    values[part]["captured_unix"] = 1850  # fresh (150 s old) but before the session started
    with pytest.raises(ValueError, match="predates the allocation start"):
        check_authorization(manifest=values[0], manifest_sha256="hash", policy=values[1],
                            session=values[2], spending=values[3], unit_name="u", now=2000)
    values[part]["captured_unix"] = 1900
    check_authorization(manifest=values[0], manifest_sha256="hash", policy=values[1],
                        session=values[2], spending=values[3], unit_name="u", now=2000)


def write_packet(tmp_path, values):
    m, p, s, cost = deepcopy(values)
    paths = [tmp_path / f"{name}.json" for name in ("manifest", "policy", "session", "spending")]
    paths[0].write_text(json.dumps(m)); p["manifest_sha256"] = sha256_file(paths[0])
    decision = tmp_path / "decision.txt"; decision.write_text(p["user_decision"])
    p.update(user_decision_path=str(decision), user_decision_sha256=sha256_file(decision))
    paths[1].write_text(json.dumps(p)); paths[2].write_text(json.dumps(s))
    cost["session_sha256"] = sha256_file(paths[2]); paths[3].write_text(json.dumps(cost))
    return dict(zip(("manifest_path", "policy_path", "session_path", "spending_path"), paths))


def test_admission_is_idempotent_and_cannot_reset_clock(tmp_path):
    values = packet(tmp_path); kwargs = write_packet(tmp_path, values)
    kwargs.update(unit_name="u", records_dir=tmp_path / "out", now=2000)
    validate_worker_authorization(**kwargs); validate_worker_authorization(**kwargs)
    assert len(list((tmp_path / "out/authorization_sessions").glob("session_*.json"))) == 1
    values[2]["allocation_started_unix"] += 100
    write_packet(tmp_path, values)
    with pytest.raises(ValueError, match="changed after admission"):
        validate_worker_authorization(**kwargs)


def test_superseding_approval_preserves_session_count(tmp_path):
    values = packet(tmp_path); values[1]["max_sessions"] = 1
    kwargs = write_packet(tmp_path, values)
    kwargs.update(unit_name="u", records_dir=tmp_path / "out", now=2000)
    validate_worker_authorization(**kwargs)
    old_hash = sha256_file(kwargs["policy_path"])
    values[1]["gross_ceiling_usd"] = 10
    values[1]["superseded_policy_sha256"] = [old_hash]
    values[2]["session_id"] = values[3]["session_id"] = "session-2"
    write_packet(tmp_path, values)
    with pytest.raises(ValueError, match="count exhausted"):
        validate_worker_authorization(**kwargs)
