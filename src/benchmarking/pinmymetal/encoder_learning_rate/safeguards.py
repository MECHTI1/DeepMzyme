#!/usr/bin/env python3
"""Additive safeguards (v2) of the trunk-learning-rate comparison (campaign ``pmm_trunk_lr_2026-10``).

Written at the campaign's closeout maintenance (2026-10-13), after its five fits had completed. The frozen launcher,
VM runner, assessor, specification and every record they wrote stay unchanged; these checks were NOT active during
the original session. pmm_trunklr_v2.py (launches) and pmm_trunklr_assessment_v2.py (decisions) use them.

1. Session authorization: pure validation plus an atomic, write-once binding of each controller session used.
   A v2 launch policy binds the launch manifest, ceiling and recorded authorization (SHA-256), the safety code
   (SHA-256 of these modules and the frozen launcher/assessor/shared modules), the authorized units, an explicit
   authorization reference (the dated campaign log entry), the maximum number of controller sessions, the maximum
   session duration and an expiry. A session used for launches is bound once: session and instance identity,
   allocation start, deadline and recorded duration. A second session, or a session allowed to run longer than the
   policy, is refused even when the compute ceiling still has room. The historical v1 authorization record has none
   of these bindings and is refused as a v2 policy.
2. Final worker completion: for each candidate fit, the final attempt's launch record, exit-code file and claim, and
   the matching lane status in an evidence copy, must be listed in its evidence manifest with their actual SHA-256
   (any file type), stay inside the evidence root, agree with one another (unit, run, lane, session, allocation,
   argv, scientific identity) and end with exit code 0. Archived earlier attempts never stand in for the final one.
   Any failure blocks a scientific decision.

Standard library only (no torch); nothing here starts, stops or contacts a VM.
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
import os
import re
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pmm_trunklr as tl

b, e2 = tl.b, tl.e2
LauncherError = tl.LauncherError
require = tl.require
sha256_file = tl.sha256_file
ROOT = tl.ROOT

SCHEMA_VERSION = 2
POLICY_KIND = "trunklr_v2_launch_policy"
BINDING_KIND = "trunklr_v2_session_binding"
POLICY_NAME = "tlr_v2_launch_policy.json"
SESSIONS_DIR = "tlr_v2_sessions"
V2_FILES = ("pmm_trunklr_safeguards_v2.py", "pmm_trunklr_v2.py", "pmm_trunklr_assessment_v2.py")
SAFETY_CODE_FILES = V2_FILES + tl.TOOL_FILES
POLICY_KEYS = ("schema_version", "kind", "campaign_id", "comparison_id", "authorization_ref", "manifest_sha256",
               "ceiling_sha256", "authorization_sha256", "safety_code_sha256", "authorized_units", "max_sessions",
               "max_session_seconds", "expires_at", "recorded_at")
BINDING_KEYS = ("schema_version", "kind", "policy_sha256", "ordinal", "session_id", "instance_id",
                "allocation_started", "deadline", "recorded_duration_seconds", "bound_at")
SESSION_KEYS = ("session_id", "instance_id", "allocation_started", "deadline", "max_seconds")
# The controller writes whole-second timestamps; the bound allocation must match the session exactly.
DURATION_TOLERANCE_SECONDS = 1.0
LOCK_WAIT_SECONDS = 30.0
FINAL_EXIT = re.compile(r"(\d+)\n?")


def utc_now() -> str:
    return tl.utc_now()


# ---------------------------------------------------------------------------
# Session authorization: the policy record
# ---------------------------------------------------------------------------

def safety_code_sha256(root: Path = ROOT) -> dict[str, str]:
    return {name: sha256_file(root / name) for name in SAFETY_CODE_FILES}


def policy_paths(records: Path = tl.RECORDS) -> dict[str, Path]:
    return {**tl.record_paths(records), "policy": records / POLICY_NAME, "sessions": records / SESSIONS_DIR}


def policy_record(*, records: Path, log_entry: str, decision: str, units: list[str], max_sessions: int,
                  max_session_seconds: float, expires_at: str, root: Path = ROOT) -> dict[str, Any]:
    """A v2 launch policy over the recorded manifest, ceiling and (v1) authorization; refused unless they agree."""
    tl.require_log_entry(log_entry)
    paths = policy_paths(records)
    manifest, authorization = tl.d.read_record(paths["manifest"]), tl.d.read_record(paths["authorization"])
    require(manifest is not None and paths["ceiling"].is_file() and authorization is not None,
            "Record the manifest, the ceiling and the maintainer's authorization first")
    require(authorization.get("manifest_sha256") == sha256_file(paths["manifest"])
            and authorization.get("ceiling_sha256") == sha256_file(paths["ceiling"]),
            "The recorded authorization does not name this manifest and ceiling")
    require(bool(units) and set(units) <= set(manifest["units"]), "The policy may only authorize manifest units")
    require(isinstance(max_sessions, int) and max_sessions >= 1 and float(max_session_seconds) > 0,
            "Implausible session limits")
    e2.stamp_seconds(expires_at)
    require(bool(decision.strip()), "Name the maintainer's decision this policy records")
    return {"schema_version": SCHEMA_VERSION, "kind": POLICY_KIND, "campaign_id": tl.CAMPAIGN_ID,
            "comparison_id": tl.COMPARISON_ID, "authorization_ref": {"log_entry": log_entry, "decision": decision},
            "manifest_sha256": sha256_file(paths["manifest"]), "ceiling_sha256": sha256_file(paths["ceiling"]),
            "authorization_sha256": sha256_file(paths["authorization"]), "safety_code_sha256": safety_code_sha256(root),
            "authorized_units": list(units), "max_sessions": max_sessions,
            "max_session_seconds": float(max_session_seconds), "expires_at": expires_at, "recorded_at": utc_now()}


def policy_problems(policy: Any, *, manifest_sha256: str | None, ceiling_sha256: str | None,
                    authorization_sha256: str | None, code_sha256: dict[str, str], now: float) -> list[str]:
    """Why a record is not a valid, current v2 launch policy (empty list: valid)."""
    if not isinstance(policy, dict):
        return ["the launch policy is not a JSON object"]
    missing = [key for key in POLICY_KEYS if key not in policy]
    if missing:
        return [f"the launch policy lacks the required v2 bindings {missing} (a v1 authorization never authorizes a "
                "v2 launch)"]
    problems = []
    if policy["schema_version"] != SCHEMA_VERSION or policy["kind"] != POLICY_KIND:
        problems.append("not a v2 launch policy record")
    if (policy["campaign_id"], policy["comparison_id"]) != (tl.CAMPAIGN_ID, tl.COMPARISON_ID):
        problems.append("the launch policy belongs to another campaign")
    ref = policy["authorization_ref"]
    if not (isinstance(ref, dict) and str(ref.get("log_entry", "")).startswith(tl.LOG_PREFIX)
            and str(ref.get("decision", "")).strip()):
        problems.append("the launch policy names no dated authorization (log entry and decision)")
    units = policy["authorized_units"]
    if not (isinstance(units, list) and units and all(isinstance(u, str) for u in units)):
        problems.append("the launch policy's authorized units are malformed")
    if not (isinstance(policy["max_sessions"], int) and not isinstance(policy["max_sessions"], bool)
            and policy["max_sessions"] >= 1):
        problems.append("the launch policy's maximum session count is malformed")
    if not (isinstance(policy["max_session_seconds"], (int, float)) and policy["max_session_seconds"] > 0):
        problems.append("the launch policy's maximum session duration is malformed")
    for key, current in (("manifest_sha256", manifest_sha256), ("ceiling_sha256", ceiling_sha256),
                         ("authorization_sha256", authorization_sha256)):
        if current is None or policy[key] != current:
            problems.append(f"the launch policy's {key} is not the recorded file's")
    if policy["safety_code_sha256"] != code_sha256:
        problems.append("the safety code changed since the launch policy was recorded (record a new policy)")
    try:
        if now >= e2.stamp_seconds(str(policy["expires_at"])):
            problems.append(f"the launch authorization expired at {policy['expires_at']}")
    except (ValueError, TypeError):
        problems.append("the launch policy's expiry is malformed")
    return problems


# ---------------------------------------------------------------------------
# Session authorization: bound sessions
# ---------------------------------------------------------------------------

def session_problems_of(session: Any) -> list[str]:
    if not isinstance(session, dict) or any(key not in session for key in SESSION_KEYS):
        return ["the controller session record is missing or malformed"]
    problems = []
    if not (isinstance(session["session_id"], str) and session["session_id"]
            and isinstance(session["instance_id"], str) and session["instance_id"]):
        problems.append("the controller session has no session or instance identity")
    try:
        start, deadline, recorded = (float(session[k]) for k in ("allocation_started", "deadline", "max_seconds"))
        if not (0 < start < deadline and recorded > 0):
            problems.append("the controller session's allocation start, deadline or duration is inconsistent")
    except (TypeError, ValueError):
        problems.append("the controller session's allocation fields are malformed")
    return problems


def binding_record(policy_sha256: str, ordinal: int, session: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "kind": BINDING_KIND, "policy_sha256": policy_sha256,
            "ordinal": ordinal, "session_id": session["session_id"], "instance_id": session["instance_id"],
            "allocation_started": float(session["allocation_started"]), "deadline": float(session["deadline"]),
            "recorded_duration_seconds": float(session["max_seconds"]), "bound_at": utc_now()}


def read_bindings(sessions_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """The bound sessions in order (session_1.json, session_2.json, ...), or the reasons the history is unusable."""
    if not sessions_dir.exists():
        return [], []
    if not sessions_dir.is_dir() or sessions_dir.is_symlink():
        return [], [f"{sessions_dir} is not a plain directory"]
    found = sorted(sessions_dir.iterdir())
    names = [p.name for p in found if not p.name.startswith(".")]
    expected = [f"session_{k}.json" for k in range(1, len(names) + 1)]
    if sorted(names) != sorted(expected):
        return [], [f"the bound-session history is malformed: {sorted(names)} (expected {expected})"]
    bindings = []
    for name in expected:
        path = sessions_dir / name
        try:
            record = e2.read_json(path)
        except (OSError, ValueError):
            return [], [f"{name}: unreadable bound-session record"]
        if path.is_symlink() or not isinstance(record, dict) or any(key not in record for key in BINDING_KEYS):
            return [], [f"{name}: malformed bound-session record"]
        bindings.append(record)
    return bindings, []


def session_gate(policy: dict[str, Any], policy_sha256: str, bindings: list[dict[str, Any]], session: Any,
                 units: list[str]) -> tuple[list[str], dict[str, Any] | None]:
    """Pure check of one launch request against the policy and the bound sessions: (problems, binding to record if
    this is a new session the policy admits). Dollars are never consulted: the ceiling is the frozen launcher's."""
    problems = session_problems_of(session)
    outside = sorted(set(units) - set(policy["authorized_units"]))
    if outside:
        problems.append(f"{outside} are not authorized by the launch policy")
    if problems:
        return problems, None
    limit = float(policy["max_session_seconds"])
    allocated = float(session["deadline"]) - float(session["allocation_started"])
    if allocated > limit + DURATION_TOLERANCE_SECONDS or float(session["max_seconds"]) > limit + DURATION_TOLERANCE_SECONDS:
        problems.append(f"the controller session may run {max(allocated, float(session['max_seconds'])):.0f} s, above "
                        f"the authorized {limit:.0f} s per session (refused even if the compute ceiling has room)")
    for ordinal, record in enumerate(bindings, start=1):
        if (record.get("kind"), record.get("schema_version"), record.get("ordinal")) != (BINDING_KIND, SCHEMA_VERSION,
                                                                                         ordinal):
            problems.append(f"bound session {ordinal} is inconsistent with its position in the history")
        if record.get("policy_sha256") != policy_sha256:
            problems.append(f"bound session {ordinal} was bound under another launch policy")
    ids = [record.get("session_id") for record in bindings]
    if len(set(ids)) != len(ids):
        problems.append("the bound-session history names one session twice")
    if problems:
        return problems, None
    matching = [record for record in bindings if record["session_id"] == session["session_id"]]
    if matching:
        bound = matching[0]
        for key, current in (("instance_id", session["instance_id"]),
                             ("allocation_started", float(session["allocation_started"])),
                             ("deadline", float(session["deadline"])),
                             ("recorded_duration_seconds", float(session["max_seconds"]))):
            if bound[key] != current:
                problems.append(f"session {session['session_id']}: {key} changed since it was bound "
                                f"({bound[key]!r} -> {current!r})")
        return problems, None
    if len(bindings) >= int(policy["max_sessions"]):
        return [f"a new controller session ({session['session_id']}) would be session {len(bindings) + 1} of at most "
                f"{policy['max_sessions']}: refused even if the compute ceiling has room"], None
    return [], binding_record(policy_sha256, len(bindings) + 1, session)


def launch_gate(records: Path, *, units: list[str], session: Any, now: float,
                root: Path = ROOT) -> tuple[list[str], dict[str, Any] | None]:
    """The policy and the bound sessions read from the records folder, checked for one launch request."""
    paths = policy_paths(records)
    if not paths["policy"].is_file():
        return ["no v2 launch policy is recorded ('record-policy' after the maintainer's decision); the v1 "
                "authorization alone never authorizes a v2 launch"], None
    try:
        policy = e2.read_json(paths["policy"])
    except (OSError, ValueError):
        return ["the v2 launch policy is unreadable"], None
    current = {key: sha256_file(paths[key]) if paths[key].is_file() else None
               for key in ("manifest", "ceiling", "authorization")}
    problems = policy_problems(policy, manifest_sha256=current["manifest"], ceiling_sha256=current["ceiling"],
                               authorization_sha256=current["authorization"], code_sha256=safety_code_sha256(root),
                               now=now)
    if problems:
        return problems, None
    bindings, history = read_bindings(paths["sessions"])
    if history:
        return history, None
    return session_gate(policy, sha256_file(paths["policy"]), bindings, session, units)


def bind_session_once(records: Path, record: dict[str, Any]) -> dict[str, Any]:
    """Atomic write-once binding: the full record is written to a temporary file and hard-linked into place, which
    fails if any binding with that ordinal exists (no partial file is ever visible)."""
    sessions = policy_paths(records)["sessions"]
    sessions.mkdir(parents=True, exist_ok=True)
    target = sessions / f"session_{record['ordinal']}.json"
    descriptor, temporary = tempfile.mkstemp(prefix=".binding-", dir=sessions)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            raise LauncherError(f"{target} exists: session {record['ordinal']} is already bound; nothing was "
                                "launched") from None
    finally:
        os.unlink(temporary)
    return e2.read_json(target)


@contextmanager
def controller_lock(controller: Path = b.CONTROLLER, *, wait_seconds: float = LOCK_WAIT_SECONDS) -> Iterator[None]:
    """A shared hold of the controller's own flock (state/controller.lock), read-only: no controller lifecycle action
    (start, stop, extend, fallback) can change the session while it is held. The owner record is never written."""
    path = controller / "state" / "controller.lock"
    require(path.is_file(), f"{path} is missing: the controller has never run here")
    handle = open(path, "r", encoding="utf-8")
    try:
        deadline = time.monotonic() + wait_seconds
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise LauncherError("A controller lifecycle action holds its lock; nothing was launched "
                                        "(check vm-status, then retry)") from None
                time.sleep(1)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        handle.close()


# ---------------------------------------------------------------------------
# Controller ledger (read-only): a session's recorded allocation
# ---------------------------------------------------------------------------

def ledger_session(session_id: str, ledger: Path = e2.LEDGER) -> dict[str, Any] | None:
    """The allocation the controller recorded for a session (RUNNING event) and its stop, as runner fields."""
    running, stopped = None, None
    for line in Path(ledger).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("session_id") != session_id:
            continue
        if event.get("event") == "RUNNING":
            require(running is None, f"{session_id}: two RUNNING records in the controller ledger")
            running = event
        elif event.get("event") == "STOPPED":
            stopped = event
    if running is None:
        return None
    return {"session_id": session_id, "instance_id": str(running["instance_id"]),
            "allocation_started": e2.stamp_seconds(running["session_start"]),
            "deadline": e2.stamp_seconds(running["termination_ts"]),
            "max_seconds": float(running["max_run_duration_s"]),
            "stopped_at": None if stopped is None else stopped.get("session_stop"),
            "elapsed_running_seconds": None if stopped is None else float(stopped["elapsed_running_s"])}


# ---------------------------------------------------------------------------
# Final worker completion (evidence copy)
# ---------------------------------------------------------------------------

def _confined(root: Path, rel: str) -> tuple[Path | None, str | None]:
    """The path of a manifest-relative file if it stays inside the evidence root without links."""
    parts = Path(rel).parts
    if Path(rel).is_absolute() or ".." in parts or not parts:
        return None, f"{rel}: not a plain relative path inside the evidence root"
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            return None, f"{rel}: passes through a symbolic link"
    resolved = current.resolve()
    if root.resolve() not in resolved.parents:
        return None, f"{rel}: resolves outside the evidence root"
    return current, None


def _hashed(root: Path, listed: dict[str, str], rel: str, problems: list[str], *, required: bool = True) -> Path | None:
    """The file if it is listed in the evidence manifest with its actual SHA-256 (every file type)."""
    path, problem = _confined(root, rel)
    if problem:
        problems.append(problem)
        return None
    if rel not in listed:
        if required or path.exists():
            problems.append(f"{rel}: not listed in the evidence manifest")
        return None
    if not path.is_file():
        problems.append(f"{rel}: listed in the evidence manifest but missing")
        return None
    if sha256_file(path) != listed[rel]:
        problems.append(f"{rel}: SHA-256 differs from the evidence manifest")
        return None
    return path


def _json(path: Path | None, rel: str, problems: list[str]) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        problems.append(f"{rel}: not valid JSON")
        return None
    if not isinstance(value, dict):
        problems.append(f"{rel}: not a JSON object")
        return None
    return value


def fold_of(spec: dict[str, Any], unit: str) -> int | None:
    return spec["folds"][spec["units"].index(unit)] if unit in spec["units"] else None


def earlier_attempts(evidence_dir: Path, unit: str) -> dict[str, Any]:
    """Archived earlier attempts (renamed launch files <unit>.<suffix>.N and failed_attempts/<unit>/attempt*): they
    are reported, never used in place of the final attempt."""
    launch = evidence_dir / "campaign" / "step_tlr" / "launch"
    renamed = sorted(p.name for p in launch.glob(f"{unit}.*.*") if p.name.rsplit(".", 1)[-1].isdigit()) \
        if launch.is_dir() else []
    archived = evidence_dir / "campaign" / "failed_attempts" / unit
    attempts = sorted(p.name for p in archived.glob("attempt*")) if archived.is_dir() else []
    return {"renamed_launch_files": renamed, "failed_attempts": attempts}


def final_completion(evidence_dir: Path, unit: str, *, listed: dict[str, str], spec: dict[str, Any],
                     runner_sha256: str, expected_session: dict[str, Any] | None = None,
                     durable: Path | None = None) -> dict[str, Any]:
    """Verify one candidate's FINAL attempt from an evidence copy; returns {"verified": bool, "problems": [...], ...}."""
    problems: list[str] = []
    root = Path(evidence_dir)
    base = f"campaign/step_tlr/launch/{unit}"
    launch = _json(_hashed(root, listed, f"{base}.launch.json", problems), f"{base}.launch.json", problems)
    rc_path = _hashed(root, listed, f"{base}.rc", problems)
    boot_path = _hashed(root, listed, f"{base}.boot", problems)
    for suffix in ("out", "pid"):  # written by the same launch script; hash-checked when listed
        _hashed(root, listed, f"{base}.{suffix}", problems, required=False)
    claim = _json(_hashed(root, listed, f"campaign/claims/{unit}.json", problems), f"campaign/claims/{unit}.json",
                  problems)
    exit_code = None
    if rc_path is not None:
        text = rc_path.read_text(encoding="utf-8", errors="replace")
        match = FINAL_EXIT.fullmatch(text)
        if match is None:
            problems.append(f"{base}.rc: the final exit file is not an integer exit code ({text[:20]!r})")
        else:
            exit_code = int(match.group(1))
            if exit_code != 0:
                problems.append(f"{base}.rc: the final attempt exited with code {exit_code}")
    elif not any(p.startswith(f"{base}.rc") for p in problems):
        problems.append(f"{base}.rc: no final exit file")
    lane = launch.get("lane") if launch else None
    status_rel = f"campaign/lanes/lane{lane}/run_status_{unit}.json"
    status = None
    if isinstance(lane, int) and not isinstance(lane, bool):
        status = _json(_hashed(root, listed, status_rel, problems), status_rel, problems)
    elif launch is not None:
        problems.append(f"{base}.launch.json: malformed lane {lane!r}")
    if launch is not None:
        if launch.get("unit") != unit or launch.get("kind") != "unit":
            problems.append(f"{base}.launch.json: names another unit or kind")
        argv = launch.get("argv")
        session_id = launch.get("session_id")
        if not isinstance(argv, list) or not all(isinstance(a, str) for a in argv):
            problems.append(f"{base}.launch.json: malformed argv")
        else:
            flags = dict(zip(argv[3::2], argv[4::2])) if len(argv) > 3 else {}
            try:
                argv_session = {"session_id": flags["--session-id"],
                                "allocation_started": float(flags["--allocation-started"]),
                                "deadline": float(flags["--execution-deadline"]),
                                "max_seconds": float(flags["--execution-max-seconds"])}
                expected_argv = tl.unit_command(unit, lane, argv_session, fit_seconds=float(launch.get("fit_seconds")))
            except (KeyError, TypeError, ValueError):
                argv_session, expected_argv = None, None
            if expected_argv is None or argv != expected_argv:
                problems.append(f"{base}.launch.json: argv is not the comparison runner command for this unit, lane "
                                "and session")
            elif argv_session["session_id"] != session_id:
                problems.append(f"{base}.launch.json: argv names another session than the launch record")
            if expected_session is not None and argv_session is not None:
                for key in ("session_id", "allocation_started", "deadline", "max_seconds"):
                    if argv_session[key] != expected_session.get(key):
                        problems.append(f"{base}.launch.json: launched with {key} {argv_session[key]!r}, not the bound "
                                        f"session's {expected_session.get(key)!r}")
    if claim is not None and launch is not None:
        if claim.get("run_name") != unit or claim.get("lane") != lane or claim.get("session_id") != launch.get("session_id"):
            problems.append(f"campaign/claims/{unit}.json: unit, lane or session differs from the final launch record")
        boot = boot_path.read_text(encoding="utf-8").strip() if boot_path is not None else None
        if boot is None or (claim.get("worker") or {}).get("boot_id") != boot:
            problems.append(f"campaign/claims/{unit}.json: the worker did not run in the final launch's boot")
        try:
            claimed = e2.stamp_seconds(str(claim["claimed_at"]).replace("+0000", "Z"))
            if claimed < e2.stamp_seconds(str(launch["launched_at"])):
                problems.append(f"campaign/claims/{unit}.json: claimed before the final launch (an earlier attempt)")
        except (KeyError, ValueError):
            problems.append(f"campaign/claims/{unit}.json: missing or malformed claim/launch times")
    if status is not None:
        identity = status.get("identity") or {}
        if status.get("status") != "completed":
            problems.append(f"{status_rel}: status {status.get('status')!r}, not completed")
        if status.get("unit") != unit or status.get("run_name") != unit or status.get("lane") != lane:
            problems.append(f"{status_rel}: unit, run or lane differs from the final launch record")
        if claim is not None and identity != claim.get("identity"):
            problems.append(f"{status_rel}: scientific identity differs from the claim")
        extension = identity.get("lr_extension") or {}
        expected = {"family": spec["family"], "target_scheme": spec["target"], "recipe": spec["recipe"],
                    "model_seed": spec["seed"], "epochs": spec["epochs"], "fold": fold_of(spec, unit)}
        differing = sorted(k for k, v in expected.items() if identity.get(k) != v)
        if differing:
            problems.append(f"{status_rel}: scientific identity differs from the specified unit in {differing}")
        if (extension.get("campaign_id"), extension.get("gvp_learning_rate"), extension.get("runner_sha256")) != (
                tl.CAMPAIGN_ID, spec["candidate_gvp_learning_rate"], runner_sha256):
            problems.append(f"{status_rel}: the learning-rate extension is not this comparison's bound runner")
        if status.get("selected_epoch") != spec["epochs"]:
            problems.append(f"{status_rel}: the selected checkpoint is not the terminal epoch {spec['epochs']}")
        if durable is not None and isinstance(lane, int):
            copy = Path(durable) / f"lane{lane}" / f"run_status_{unit}.json"
            if not copy.is_file() or copy.read_bytes() != (root / status_rel).read_bytes():
                problems.append(f"{status_rel}: the durable copy differs (pull first)")
            else:
                checkpoint = Path(durable) / f"lane{lane}" / "runs" / unit / str(status.get("selected_checkpoint"))
                if not checkpoint.is_file() or sha256_file(checkpoint) != status.get("selected_checkpoint_sha256"):
                    problems.append(f"{unit}: the durable selected checkpoint differs from its status record")
    return {"unit": unit, "verified": not problems, "problems": problems, "lane": lane,
            "session_id": None if launch is None else launch.get("session_id"), "exit_code": exit_code,
            "earlier_attempts": earlier_attempts(root, unit)}


def completion_report(evidence_dir: Path, *, spec: dict[str, Any], runner_sha256: str,
                      sessions: dict[str, dict[str, Any]] | None = None, durable: Path | None = None) -> dict[str, Any]:
    """Every candidate's final attempt; 'verified' only if all of them verify. ``sessions`` maps a session id to its
    bound (or controller-recorded) allocation; a launch under an unknown session then fails."""
    problems: list[str] = []
    try:
        manifest = e2.read_json(Path(evidence_dir) / "evidence_manifest.json")
        listed = manifest["files"]
        require(isinstance(listed, dict), "malformed")
        if manifest.get("mismatched"):
            problems.append("the evidence copy recorded mismatches")
    except (OSError, ValueError, KeyError, LauncherError):
        return {"verified": False, "problems": ["the evidence manifest is missing or malformed"], "units": {}}
    units = {}
    for unit in spec["units"]:
        launch_rel = f"campaign/step_tlr/launch/{unit}.launch.json"
        expected = None
        if sessions is not None:
            try:
                session_id = json.loads((Path(evidence_dir) / launch_rel).read_text()).get("session_id")
            except (OSError, ValueError):
                session_id = None
            expected = sessions.get(session_id, {"session_id": None})
        units[unit] = final_completion(Path(evidence_dir), unit, listed=listed, spec=spec, runner_sha256=runner_sha256,
                                       expected_session=expected, durable=durable)
        problems += [f"{unit}: {p}" for p in units[unit]["problems"]]
    return {"verified": not problems, "problems": problems, "units": units,
            "sessions_used": sorted({u["session_id"] for u in units.values() if u["session_id"]})}
