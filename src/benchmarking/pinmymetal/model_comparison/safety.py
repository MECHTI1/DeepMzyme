"""Fail-closed authorization for the additive completion worker; no cloud operations.

The coordinator records an explicit maintainer decision and refreshes controller/session
and gross-spending snapshots. A scientific manifest never authorizes execution.
"""

from __future__ import annotations


import sys as _sys
from pathlib import Path as _Path
for _p in (str(_Path(__file__).resolve().parents[4]), str(_Path(__file__).resolve().parents[4] / "src")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
from project_paths import RELOCATED_FILES, install_relocated_module_aliases, resolve_existing_path, resolve_repo_file
install_relocated_module_aliases()
import fcntl
import json
import math
import time
from pathlib import Path

from metal_completion_campaign import CAMPAIGN_ID, require, sha256_file


def read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text())
    require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def number(value, name: str, *, positive: bool = False) -> float:
    require(type(value) in (int, float) and math.isfinite(value), f"Invalid {name}")
    require(value > 0 if positive else value >= 0, f"Invalid {name}")
    return float(value)


def check_authorization(*, manifest: dict, manifest_sha256: str, policy: dict,
                        session: dict, spending: dict, unit_name: str, now: float) -> dict:
    """Pure validation, also used by the host before transferring a launch packet."""
    require(manifest.get("campaign_id") == CAMPAIGN_ID, "Wrong campaign manifest")
    require(policy.get("schema_version") == 1 and policy.get("campaign_id") == CAMPAIGN_ID,
            "Wrong authorization schema/campaign")
    require(policy.get("execution_authorized") is True, "Execution has not been approved")
    require(policy.get("manifest_sha256") == manifest_sha256, "Authorization binds a different manifest")
    require(isinstance(policy.get("user_decision"), str) and policy["user_decision"].strip(),
            "Missing explicit maintainer decision")
    units = policy.get("units")
    require(isinstance(units, list) and units and len(units) == len(set(units))
            and set(units) <= set(manifest["authorized_grid_units"]), "Invalid authorized unit subset")
    require(unit_name in units, "Unit not authorized")
    expires = number(policy["expires_unix"], "policy expiry", positive=True)
    require(now < expires, "Authorization expired")
    require(type(policy["max_sessions"]) is int and policy["max_sessions"] > 0, "Invalid session count")
    require(type(policy["lanes"]) is int and 1 <= policy["lanes"] <= 3, "Invalid lane allowance")
    require(policy["persistence_mode"] in ("host_pull", "mounted"), "Unknown persistence mode")
    require(Path(policy["durable_root"]).is_absolute(), "Durable root must be absolute")
    for name in ("session_id", "instance_id"):
        require(isinstance(session.get(name), str) and session[name].strip(), f"Missing controller {name}")
    require(session.get("state") == "RUNNING", "Controller session is not RUNNING")
    require(isinstance(policy.get("controller_owner"), str) and policy["controller_owner"].strip()
            and session.get("controller_owner") == policy["controller_owner"], "Controller ownership mismatch")
    require(session.get("provider_hard_stop_verified") is True, "Provider automatic stop is not verified")
    start = number(session["allocation_started_unix"], "allocation start", positive=True)
    deadline = number(session["deadline_unix"], "controller deadline", positive=True)
    maximum = number(session["max_total_seconds"], "controller duration", positive=True)
    require(start <= now < deadline <= expires, "Controller session outside live approval window")
    stop = number(session["provider_hard_stop_unix"], "provider stop", positive=True)
    require(now < stop <= deadline, "Provider automatic stop exceeds the controller deadline")
    require(deadline - start <= maximum + 1, "Controller deadline exceeds its original duration")
    require(maximum <= number(policy["max_session_seconds"], "approved session duration", positive=True),
            "Session exceeds approved duration")
    require(spending.get("session_id") == session["session_id"], "Spending belongs to another session")
    fresh = number(policy["snapshot_max_age_seconds"], "snapshot freshness", positive=True)
    require(fresh <= 900, "Spending snapshot freshness may not exceed fifteen minutes")
    for item in (session, spending):
        captured = number(item["captured_unix"], "snapshot capture", positive=True)
        require(0 <= now - captured <= fresh, "Stale or future controller/spending snapshot")
        # A RUNNING session cannot be observed, nor its spending accounted, before it started.
        require(captured >= start, "Controller/spending snapshot predates the allocation start")
    rate = number(spending["gross_running_usd_per_hour"], "gross running rate", positive=True)
    prior = number(spending["spent_before_session_usd"], "previous gross spending")
    storage = number(spending["storage_and_snapshot_forecast_usd"], "storage/snapshot allowance")
    reserve = number(policy["reserve_usd"], "budget reserve")
    ceiling = number(policy["gross_ceiling_usd"], "gross ceiling", positive=True)
    # Reserve the entire current session, not only its remainder, so startup and
    # simultaneous lanes cannot undercount the active allocation.
    committed = prior + maximum / 3600 * rate + storage + reserve
    require(committed <= ceiling, "Full-session gross cost plus storage/reserve exceeds approved ceiling")
    allocated = number(spending["allocated_seconds_before_session"], "previous allocated seconds")
    gpu_ceiling = number(policy["max_total_gpu_seconds"], "campaign GPU time ceiling", positive=True)
    require(allocated + maximum <= gpu_ceiling, "Full session exceeds campaign GPU time ceiling")
    estimate = number(policy["estimated_fit_seconds"], "fit estimate", positive=True)
    effective_deadline = min(deadline, stop)
    require(now + 1.25 * estimate + 900 <= effective_deadline,
            "Insufficient time for a fit and shutdown reserve")
    return {"session_id": session["session_id"], "allocation_started_unix": start,
            "deadline_unix": effective_deadline, "max_total_seconds": maximum,
            "estimated_fit_seconds": estimate, "durable_root": policy["durable_root"],
            "persistence_mode": policy["persistence_mode"], "lanes": policy["lanes"],
            "gross_committed_usd": committed}


def validate_worker_authorization(*, manifest_path: Path, policy_path: Path, session_path: Path,
                                  spending_path: Path, unit_name: str, records_dir: Path,
                                  now: float | None = None) -> dict:
    manifest, policy = read_json(manifest_path), read_json(policy_path)
    session, spending = read_json(session_path), read_json(spending_path)
    runtime = check_authorization(manifest=manifest, manifest_sha256=sha256_file(manifest_path),
                                  policy=policy, session=session, spending=spending,
                                  unit_name=unit_name, now=time.time() if now is None else now)
    log_path = Path(policy["user_decision_path"])
    require(sha256_file(log_path) == policy["user_decision_sha256"], "Maintainer decision record changed")
    require(policy["user_decision"] in log_path.read_text(), "Decision absent from bound maintainer record")
    require(spending.get("session_sha256") == sha256_file(session_path), "Session/spending binding changed")
    require(Path(records_dir).resolve() == Path(manifest["successor_root"]).resolve(),
            "Session records must live in the campaign output root")
    bindings = Path(records_dir) / "authorization_sessions"
    require(not bindings.is_symlink(), "Session binding directory must not be a symlink")
    bindings.mkdir(parents=True, exist_ok=True)
    lock = bindings / ".lock"
    require(not lock.is_symlink(), "Unsafe session lock")
    # All lanes share this lock. Session id, original start/deadline and allowance
    # are immutable once admitted; reconnecting cannot reset the allocation clock.
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        records = sorted(bindings.glob("session_*.json"))
        require({p.name for p in records} == {f"session_{i}.json" for i in range(1, len(records) + 1)},
                "Non-contiguous session binding history")
        require(all(not p.is_symlink() for p in records), "Unsafe session record")
        existing = [read_json(p) for p in records]
        policy_hash = sha256_file(policy_path)
        superseded = policy.get("superseded_policy_sha256", [])
        require(isinstance(superseded, list) and all(isinstance(x, str) and len(x) == 64 for x in superseded),
                "Invalid explicitly superseded policies")
        require(all(r["policy_sha256"] in [policy_hash, *superseded] for r in existing),
                "Session policy changed without an explicitly bound superseding approval")
        require(len({r["session_id"] for r in existing}) == len(existing), "Duplicate session binding")
        binding = {"policy_sha256": policy_hash,
                   **{k: session[k] for k in ("session_id", "instance_id", "allocation_started_unix",
                                             "deadline_unix", "max_total_seconds")}}
        match = [r for r in existing if r["session_id"] == session["session_id"]]
        if match:
            require(all(match[0][key] == value for key, value in binding.items() if key != "policy_sha256"),
                    "Controller session changed after admission")
        else:
            require(len(existing) < policy["max_sessions"], "Approved session count exhausted")
            with (bindings / f"session_{len(existing) + 1}.json").open("x") as out:
                json.dump(binding, out, sort_keys=True, indent=2)
                out.write("\n")
    return runtime
