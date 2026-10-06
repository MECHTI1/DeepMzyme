#!/usr/bin/env python3
"""Workstation launcher for v3 plan step C (standard library only).

Like pmm_v3_step_b.py, whose session, SSH, lane-readiness, host-pull and evidence helpers it reuses, it never
starts, stops, restores or extends a VM: the controller (~/deepmzyme-vm/bin) owns billing, and a GPU start needs
the user's typed authorization. Every VM command talks to an already RUNNING VM over the controller's IAP SSH
configuration and starts the VM-side runner (run_pmm_v3_campaign.py) detached under setsid/nohup.

Commands (step C order; see the metal playbook, "PMM ion metal v3 campaign"):
  units                   offline: the step C units, the reused cell, admission forecasts and the suggested order
  session                 print the controller session (refuses if inactive, expired or mismatched)
  status                  read-only: lanes, launches, claims, unit statuses and the regression gate on the VM
  regression [--fit-seconds S] [--no-wait]
                          run the regression fit alone in lane 0 (every lane idle, nothing else launched), then
                          wait, pull lane 0 and print its gate
  launch UNIT@LANE ... [--fit-seconds S] [--no-wait]
                          start 1-3 step C units together, one per lane, each detached; refused until the
                          regression run passed its gate; then waits, pulling each lane as its unit ends
  wait [UNIT ...] [--any] resume waiting after a dropped connection (never relaunches); pulls every lane whose unit
                          ended; --any returns after the first unit ends (to refill that lane)
  pull [--lanes K ...]    verified per-lane host pull (a failure leaves the lane closed and fails the command)
  recover-lane K          a lane whose runner died mid-unit (VM stop, reboot, kill): run the lane's own recovery,
                          which records the unit as interrupted, then pull it
  archive-failed UNIT     move a failed or interrupted step C unit aside for its one unchanged rerun (refused while
                          any lane awaits its pull or the unit's lane still needs recover-lane)
  assess                  pmm_v3_assessment.py --step C on the VM; checked copy on the workstation
  evidence [--allow-running]   copy the step C evidence with SHA-256 checked on both ends; run before every
                          vm-stop. It fails after copying while a unit runs or a lane awaits its pull or recovery

Protections: a launch is refused while any lane awaits its host pull, when a target lane holds a running or
interrupted unit, a live child or its lock, or has a launch without an exit code; for a unit that is not a
step C unit, is the reused step-B cell, completed, claimed, already launched, listed twice or placed outside
the recorded lanes; before the regression run passed its gate; and when the session has expired or has less
than 1.25 x forecast + 900 s (+ launch margin) left. A launch whose runner died without an exit code (VM
stopped or rebooted), or whose detached shell never started, is reported as lost, never awaited forever.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pmm_v3_step_b as b  # noqa: E402  (shared session, SSH, lane, pull and evidence helpers)

LauncherError = b.LauncherError
require = b.require

FAMILIES = ("only_esm", "only_gvp", "gvp_late_fusion")
TARGETS = ("four_class", "five_class", "six_class")
# pmm_v3_campaign.step_units("C") (checked by tests; importing the campaign module here would import torch).
STEP_C_UNITS = tuple(f"{family}__{target}__baseline__fold0__seed42" for family in FAMILIES for target in TARGETS) \
    + tuple(f"{family}__four_class__v2recipe__fold0__seed42" for family in FAMILIES)
REGRESSION_NAME = "regression__gvp_late_fusion__four_class__v2recipe__v2fold0__seed42"  # pmm_v3_campaign
USES_ESM = {"only_esm": True, "only_gvp": False, "gvp_late_fusion": True}  # pmm_ion_campaign.FAMILIES

# Admission forecasts (seconds, end to end) from step B (log v3-010): three concurrent late-fusion 50-epoch fits
# projected 2,336-2,344 s; Only-GVP and Only-ESMC scaled by the v2 family ratios (0.93, 0.64) and rounded up. A
# unit whose cache set (target scheme, with or without ESMC: both key the parse and graph caches) has no completed
# fit yet adds the measured cold-minus-warm gap (step B r1 vs r2: 2,155 - 647 = 1,508 s, i.e. parsing 6.1 vs 0.6
# min and about 1,200 s of single-threaded graph building) x 1.3 for contention when cold builds overlap. The
# regression run (serial, its cache possibly cold) gets the serial full fit plus that allowance. --fit-seconds
# overrides; the runner admits a unit only if 1.25 x forecast + 900 s remain before the hard stop.
FIT_SECONDS = {"gvp_late_fusion": 2400, "only_gvp": 2300, "only_esm": 1800}
COLD_CACHE_SECONDS = 2000
REGRESSION_FIT_SECONDS = 3800
# Longest and cold-cache units first, so short Only-ESMC fits fill the end of a session (not enforced).
SUGGESTED_ORDER = (
    "gvp_late_fusion__five_class__baseline__fold0__seed42", "gvp_late_fusion__six_class__baseline__fold0__seed42",
    "only_gvp__four_class__baseline__fold0__seed42", "only_gvp__five_class__baseline__fold0__seed42",
    "only_gvp__six_class__baseline__fold0__seed42", "gvp_late_fusion__four_class__v2recipe__fold0__seed42",
    "only_gvp__four_class__v2recipe__fold0__seed42", "only_esm__four_class__baseline__fold0__seed42",
    "only_esm__five_class__baseline__fold0__seed42", "only_esm__six_class__baseline__fold0__seed42",
    "only_esm__four_class__v2recipe__fold0__seed42")
EVIDENCE = Path("/media/mechti/Data1/DeepMzyme_Data/campaigns/pmm_ion_metal_v3/step_c_evidence")
EVIDENCE_PATTERNS = ("campaign_manifest.json", "execution_settings.json", "fold_class_weights.json", "claims/*.json",
                     "step_c/**/*", "assessments/**/*", "lanes/lane*/execution_state.json",
                     "lanes/lane*/execution_events.jsonl", "lanes/lane*/run_status_*.json",
                     "lanes/lane*/commands/*.json", "lanes/lane*/persistence_receipts/*.ack.json",
                     "failed_attempts/**/*")
RC_MEANING = {"0": "completed", "1": "failed (after its lane is pulled: archive-failed, then its one unchanged rerun)",
              "2": "refused before anything ran", "3": "blocked (lane, time or a pending host acknowledgment)",
              "4": "persistence failed"}
MAX_CONTACT_FAILURES = 10
MAX_PULL_ATTEMPTS = 3  # a pull is idempotent; transient IAP/rsync errors are retried, then the command fails
STARTING_GRACE_SECONDS = 120  # a launch record without a member PID after this long never started


def rc_meaning(rc: str | None, *, regression: bool = False) -> str:
    if rc is None:
        return "no exit code (VM stopped or rebooted)"
    if regression and rc == "1":
        return "regression failed or its gate failed: STOP for diagnosis"
    if rc in RC_MEANING:
        return RC_MEANING[rc]
    return f"killed by signal {int(rc) - 128}" if rc.isdigit() and int(rc) > 128 else f"unexpected exit code {rc}"


def launch_dir() -> str:
    return f"{b.REMOTE_V3}/step_c/launch"


def family_target(name: str) -> tuple[str, str]:
    parts = name.split("__")
    return parts[0], parts[1]


# ---------------------------------------------------------------------------
# VM state (one read-only snapshot per poll)
# ---------------------------------------------------------------------------

STATE_PROBE = r"""
import json, sys, time
from pathlib import Path
root, launch_dir, grace = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
settings = root / "execution_settings.json"
out = {"settings": json.loads(settings.read_text()) if settings.is_file() else None,
       "statuses": {}, "claims": [], "archived": {}, "launches": {}}
for path in sorted(root.glob("lanes/lane*/run_status_*.json")):
    record = json.loads(path.read_text())
    identity = record.get("identity") or {}
    out["statuses"][record["run_name"]] = {"status": record.get("status"), "lane": record.get("lane"),
                                           "gate_passed": (record.get("gate") or {}).get("passed"),
                                           "family": identity.get("family"), "target": identity.get("target_scheme")}
out["claims"] = sorted(path.name[:-5] for path in (root / "claims").glob("*.json"))
for directory in sorted((root / "failed_attempts").glob("*")):
    out["archived"][directory.name] = len(list(directory.glob("attempt*")))
for record_path in sorted(launch_dir.glob("*.launch.json")):
    name = record_path.name[:-len(".launch.json")]
    record = json.loads(record_path.read_text())
    rc, pid, boot_file = (launch_dir / f"{name}.{suffix}" for suffix in ("rc", "pid", "boot"))
    code = rc.read_text().strip() if rc.is_file() else None
    if code is not None:
        state = "exited"
    elif not boot_file.is_file() or boot_file.read_text().strip() != boot:
        state = "lost"  # the VM stopped or rebooted after the launch: no exit code will ever come
    elif not pid.is_file():  # the detached shell writes the PID within milliseconds of the launch
        state = "starting" if time.time() - record_path.stat().st_mtime < grace else "lost"
    else:
        try:
            alive = name.encode() in Path(f"/proc/{int(pid.read_text())}/cmdline").read_bytes()
        except (OSError, ValueError):
            alive = False
        if not alive and rc.is_file():  # it ended between the two reads
            code = rc.read_text().strip()
        state = "running" if alive else "exited" if code is not None else "lost"
    out["launches"][name] = {"unit": record.get("unit"), "lane": record.get("lane"), "kind": record.get("kind"),
                             "session_id": record.get("session_id"), "state": state, "rc": code}
print(json.dumps(out))
"""


def vm_state(remote: b.Remote, lanes: int = 3) -> dict[str, Any]:
    """Launch records, statuses, claims and settings, plus the step-B lane probe of lanes 0..lanes-1."""
    state = json.loads(remote.run(f"{b.REMOTE_PY} -c {shlex.quote(STATE_PROBE)} {b.REMOTE_V3} {launch_dir()} "
                                 f"{STARTING_GRACE_SECONDS}").stdout)
    width = int((state.get("settings") or {}).get("concurrent_lanes", lanes))
    state["lanes"] = json.loads(remote.run(f"{b.REMOTE_PY} -c {shlex.quote(b.LANE_PROBE)} {b.REMOTE_V3} "
                                           + " ".join(str(k) for k in range(width))).stdout)
    state["width"] = width
    for name, record in state["launches"].items():  # a busy lane overrides a launch record that reads 'lost'
        info = state["lanes"].get(f"lane{record.get('lane')}", {})
        if record["state"] == "lost" and lane_busy(info) and (info.get("pending_unit") or {}).get("run_name") == name:
            record["state"] = "running"
    return state


def lane_busy(info: dict[str, Any]) -> bool:
    """A worker holds the lane lock or a training child is alive: something runs there, whatever the records say."""
    return not info.get("lock_free", True) or bool(info.get("active_child_alive"))


def active_launches(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {name: record for name, record in state["launches"].items() if record["state"] in ("running", "starting")}


def awaiting_pull(state: dict[str, Any]) -> list[int]:
    """Lanes whose ended unit has not been acknowledged by the host yet."""
    out = []
    for lane, info in sorted(state["lanes"].items()):
        pending = info.get("pending_unit") or {}
        if info.get("pending_transfer") and not info.get("ack_uploaded") and pending.get("status") != "running":
            out.append(int(lane[4:]))
    return out


def interrupted_lanes(state: dict[str, Any]) -> dict[int, str]:
    """Lanes whose worker died mid-unit (VM stop, reboot, killed runner): the unit is still 'running' in the
    lane state, but no lock holder, live child or live launch remains. 'recover-lane' records and pulls it."""
    live = {record["lane"] for record in active_launches(state).values()}
    out = {}
    for lane, info in sorted(state["lanes"].items()):
        pending = info.get("pending_unit") or {}
        if (pending.get("status") == "running" and not pending.get("persisted") and not info.get("pending_transfer")
                and info.get("lock_free", True) and not info.get("active_child_alive") and int(lane[4:]) not in live):
            out[int(lane[4:])] = pending.get("run_name")
    return out


def cache_warm(name: str, statuses: dict[str, dict[str, Any]]) -> bool:
    """A completed fit (any lane, step-B probes included) already built this unit's cache set."""
    family, target = family_target(name)
    return any(record.get("status") == "completed" and record.get("target") == target
               and record.get("family") in USES_ESM and USES_ESM[record["family"]] == USES_ESM[family]
               for record in statuses.values())


def forecast_seconds(name: str, statuses: dict[str, dict[str, Any]]) -> int:
    if name == REGRESSION_NAME:
        return REGRESSION_FIT_SECONDS
    return FIT_SECONDS[family_target(name)[0]] + (0 if cache_warm(name, statuses) else COLD_CACHE_SECONDS)


# ---------------------------------------------------------------------------
# Launch planning (pure: every refusal is decided before anything starts)
# ---------------------------------------------------------------------------

def regression_blockers(state: dict[str, Any]) -> list[str]:
    """Plan step C: the regression run goes first and alone; any other fit waits for its passed gate."""
    launch = state["launches"].get(REGRESSION_NAME)
    if launch and launch["state"] in ("running", "starting"):
        return ["the regression run is still running; step C units start only after it passed its gate ('wait')"]
    record = state["statuses"].get(REGRESSION_NAME)
    if record is None:
        return ["the regression run has not completed; run 'regression' first (alone)"]
    if record.get("status") != "completed" or record.get("gate_passed") is not True:
        return [f"STOP: the regression run ended '{record.get('status')}' (gate passed: {record.get('gate_passed')}); "
                "diagnose before any other step C fit (plan step C)"]
    return []


def unit_blockers(name: str, state: dict[str, Any]) -> list[str]:
    settings = state.get("settings") or {}
    if name not in STEP_C_UNITS:
        return [f"{name}: not a step C unit (steps D-E need their own authorization)"]
    if name in settings.get("reuse", {}):
        return [f"{name}: reused from step B ({settings['reuse'][name]}); never launched"]
    record, archived = state["statuses"].get(name), state["archived"].get(name, 0)
    launch = state["launches"].get(name)
    if launch and launch["state"] in ("running", "starting"):
        return [f"{name}: an earlier launch has no exit code yet (use 'wait', never relaunch)"]
    if record is not None:
        if record.get("status") == "completed":
            return [f"{name}: completed; a completed unit is never rerun"]
        if archived:
            return [f"{name}: failed after its one rerun ('{record.get('status')}'); step C stops for diagnosis "
                    "and the user's decision"]
        return [f"{name}: ended '{record.get('status')}'; after its lane is pulled, run 'archive-failed {name}' "
                "(its one unchanged rerun)"]
    stranded = [k for k, run in interrupted_lanes(state).items() if run == name]
    if stranded:
        return [f"{name}: interrupted in lane{stranded[0]} (VM stop, reboot or killed runner); run 'recover-lane "
                f"{stranded[0]}', then 'archive-failed {name}' (its one unchanged rerun)"]
    if name in state["claims"]:
        return [f"{name}: claimed without a terminal status (running or interrupted); check 'status' and the lane"]
    return []


def lane_problems(state: dict[str, Any], lanes: list[int]) -> list[str]:
    problems = b.lane_blockers({f"lane{k}": state["lanes"].get(f"lane{k}", {"exists": False, "lock_free": True})
                                for k in lanes})
    busy = {record["lane"]: name for name, record in active_launches(state).items()}
    problems += [f"lane{k}: {busy[k]} was launched there and has no exit code yet (use 'wait')" for k in lanes if k in busy]
    stranded = interrupted_lanes(state)
    problems += [f"lane{k}: {stranded[k]} was interrupted; run 'recover-lane {k}' (the runner's own recovery, then "
                 "a verified pull)" for k in lanes if k in stranded]
    return problems


def pull_problems(state: dict[str, Any]) -> list[str]:
    return [f"lane{k}: an ended unit awaits its verified host pull (run 'pull --lanes {k}'); nothing new starts "
            "until every lane is acknowledged" for k in awaiting_pull(state)]


def plan_units(requests: list[tuple[str, int]], state: dict[str, Any], *,
               fit_seconds: float | None = None) -> list[dict[str, Any]]:
    require(state.get("settings") is not None, "No execution setting is recorded on the VM (set-execution)")
    width = int(state["settings"]["concurrent_lanes"])
    require(1 <= len(requests) <= width, f"Launch 1-{width} units at once (the recorded concurrent lanes)")
    names, lanes = [name for name, _ in requests], [lane for _, lane in requests]
    problems = []
    if len(set(names)) != len(names):
        problems.append("a unit is listed twice")
    if len(set(lanes)) != len(lanes):
        problems.append("two units share a lane")
    problems += [f"lane {k} is outside the recorded {width} lanes (0..{width - 1})" for k in lanes if not 0 <= k < width]
    problems += regression_blockers(state) + pull_problems(state)
    for name in dict.fromkeys(names):
        problems += unit_blockers(name, state)
    problems += lane_problems(state, sorted({k for k in lanes if 0 <= k < width}))
    require(not problems, "Nothing was launched:\n  " + "\n  ".join(problems))
    return [{"unit": name, "lane": lane, "fit_seconds": float(fit_seconds or forecast_seconds(name, state["statuses"]))}
            for name, lane in requests]


def plan_regression(state: dict[str, Any], *, fit_seconds: float | None = None) -> list[dict[str, Any]]:
    require(state.get("settings") is not None, "No execution setting is recorded on the VM (set-execution)")
    problems = []
    active = active_launches(state)
    if active:
        problems.append(f"the regression run goes alone; launches without an exit code: {sorted(active)} ('wait')")
    record = state["statuses"].get(REGRESSION_NAME)
    if record is not None:
        problems.append(f"the regression run already ended '{record.get('status')}'; it is never repeated "
                        "automatically (a failure stops v3 for diagnosis)")
    elif REGRESSION_NAME in state["claims"]:
        problems.append("the regression run is claimed without a terminal status (running or interrupted)")
    problems += pull_problems(state) + lane_problems(state, list(range(int(state["width"]))))
    require(not problems, "Nothing was launched:\n  " + "\n  ".join(problems))
    return [{"unit": REGRESSION_NAME, "lane": 0, "fit_seconds": float(fit_seconds or REGRESSION_FIT_SECONDS)}]


# ---------------------------------------------------------------------------
# Commands built for the VM
# ---------------------------------------------------------------------------

def unit_command(name: str, lane: int, session: dict[str, Any], *, fit_seconds: float) -> list[str]:
    head = [b.REMOTE_PY, "-u", "run_pmm_v3_campaign.py"]
    if name == REGRESSION_NAME:
        head += ["--action", "regression", "--campaign-dir", b.REMOTE_V3, "--v2-root", b.REMOTE_V2]
    else:
        head += ["--action", "run", "--campaign-dir", b.REMOTE_V3, "--unit", name]
    head += ["--train-dir", b.REMOTE_TRAIN, "--esm-dir", b.REMOTE_ESM, "--lane", str(lane)]
    return head + b.allocation_args(session, fit_seconds=fit_seconds)


def launch_script(batch: str, planned: list[dict[str, Any]], session: dict[str, Any]) -> str:
    """One detached shell that starts every planned unit at once; per unit it records the launch, the boot ID
    and the member PID (so a later 'wait' can tell running from lost) and finally the runner's exit code."""
    directory = launch_dir()
    q = shlex.quote
    lines = ["set -eu", f"cd {q(b.REMOTE_CODE)}", f"mkdir -p {q(directory)}"]
    for item in planned:  # keep earlier attempts' files (a rerun after archive-failed reuses the name)
        base = f"{directory}/{item['unit']}"
        files = " ".join(q(f"{base}.{suffix}") for suffix in ("out", "rc", "pid", "boot", "launch.json"))
        lines.append(f"for f in {files}; do if [ -e \"$f\" ]; then n=1; while [ -e \"$f.$n\" ]; do n=$((n+1)); done; "
                     "mv \"$f\" \"$f.$n\"; fi; done")
    lines.append("boot=$(cat /proc/sys/kernel/random/boot_id)")
    members = []
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for item in planned:
        base = f"{directory}/{item['unit']}"
        command = unit_command(item["unit"], item["lane"], session, fit_seconds=item["fit_seconds"])
        record = {"unit": item["unit"], "lane": item["lane"], "batch": batch, "session_id": session["session_id"],
                  "kind": "regression" if item["unit"] == REGRESSION_NAME else "unit",
                  "fit_seconds": item["fit_seconds"], "launched_at": stamp, "argv": command}
        lines.append(f"printf '%s\\n' {q(json.dumps(record, sort_keys=True))} > {q(base + '.launch.json')}")
        lines.append(f"printf '%s\\n' \"$boot\" > {q(base + '.boot')}")
        members.append(f"( {shlex.join(command)} > {q(base + '.out')} 2>&1; echo $? > {q(base + '.rc.tmp')}; "
                       f"mv {q(base + '.rc.tmp')} {q(base + '.rc')} ) & echo $! > {q(base + '.pid')}")
    inner = "\n".join(members + ["wait"])
    lines.append(f"setsid nohup bash -c {q(inner)} > {q(directory + '/' + batch + '.launcher.log')} 2>&1 < /dev/null &")
    lines.append("echo launched")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def start(remote: b.Remote, session: dict[str, Any], planned: list[dict[str, Any]], *, wait: bool,
          sleep=time.sleep) -> dict[str, Any]:
    b.admission_check(session, max(item["fit_seconds"] for item in planned))
    batch = "c-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    try:
        out = remote.run(launch_script(batch, planned, session)).stdout
    except (LauncherError, subprocess.SubprocessError, OSError) as exc:
        raise LauncherError(f"The launch command failed ({exc}). The runners may have started: check with "
                            "'status' and resume with 'wait'; never relaunch") from exc
    require("launched" in out, f"launch did not confirm: {out}; check with 'status' and 'wait' before anything else")
    result: dict[str, Any] = {"batch": batch, "launched": planned}
    if wait:
        result.update(cmd_wait(remote, session, names=[item["unit"] for item in planned], sleep=sleep))
    else:
        result["next"] = "wait (or wait --any to refill a lane as soon as its unit ends)"
    return result


def cmd_regression(remote: b.Remote, session: dict[str, Any], *, fit_seconds: float | None = None,
                   wait: bool = True, sleep=time.sleep) -> dict[str, Any]:
    return start(remote, session, plan_regression(vm_state(remote), fit_seconds=fit_seconds), wait=wait, sleep=sleep)


def cmd_launch(remote: b.Remote, session: dict[str, Any], requests: list[tuple[str, int]], *,
               fit_seconds: float | None = None, wait: bool = True, sleep=time.sleep) -> dict[str, Any]:
    return start(remote, session, plan_units(requests, vm_state(remote), fit_seconds=fit_seconds), wait=wait,
                 sleep=sleep)


def cmd_wait(remote: b.Remote, session: dict[str, Any], *, names: list[str] | None = None, any_unit: bool = False,
             poll: float = b.POLL_SECONDS, sleep=time.sleep) -> dict[str, Any]:
    """Wait for launched units and pull every lane whose unit ended. It never launches anything, so it is the
    recovery after a dropped connection. A pull is retried (it is idempotent); a lane whose pull still fails
    stays closed, which blocks every new launch, and the command fails."""
    failures, awaited, ended, pulls, pull_failures = 0, None, {}, {}, {}
    while True:
        try:
            state = vm_state(remote)
            failures = 0
        except (LauncherError, subprocess.SubprocessError, OSError, ValueError) as exc:
            failures += 1
            require(failures < MAX_CONTACT_FAILURES,
                    f"Lost contact with the VM ({exc}); the runners continue there. Resume with 'wait'.")
            require(time.time() < session["deadline"], "The session hard stop passed while waiting")
            sleep(poll)
            continue
        if awaited is None:
            awaited = list(names) if names else sorted(active_launches(state))
            missing = [name for name in awaited if name not in state["launches"]]
            require(not missing, f"No launch record for {missing}")
        # The runner persists (pending transfer) before it exits and the exit code is written, and the launch
        # records are read before the lanes, so every unit seen as ended here has its transfer visible already.
        unpulled = []
        for lane in awaiting_pull(state):  # the ended unit's lane, or one left over by a dropped connection
            try:
                pulls[f"lane{lane}"] = b.cmd_pull(remote, [lane])["lanes"][f"lane{lane}"]
            except LauncherError as exc:
                pull_failures[lane] = pull_failures.get(lane, 0) + 1
                require(pull_failures[lane] < MAX_PULL_ATTEMPTS,
                        f"{exc} ({pull_failures[lane]} attempts; lane{lane} stays closed and nothing new starts; "
                        f"the other runners continue. Fix the cause, then 'pull --lanes {lane}' and 'wait')")
                unpulled.append(lane)
        for name in awaited:
            record = state["launches"][name]
            if name not in ended and record["state"] in ("exited", "lost"):
                status = state["statuses"].get(name, {})
                ended[name] = {"lane": record["lane"], "launch": record["state"], "exit_code": record["rc"],
                               "meaning": rc_meaning(record["rc"], regression=name == REGRESSION_NAME),
                               "status": status.get("status"), "gate_passed": status.get("gate_passed")}
        if not unpulled and (len(ended) == len(awaited) or (any_unit and ended)):
            return {"ended": ended, "still_running": [n for n in awaited if n not in ended], "pulls": pulls,
                    **summary(state), "next": next_hint(state, ended)}
        require(time.time() < session["deadline"], "The session hard stop passed while waiting")
        sleep(poll)


def summary(state: dict[str, Any]) -> dict[str, Any]:
    """Unit outcomes from the VM statuses, so a 'wait' after a dropped connection still reports them."""
    reuse = (state.get("settings") or {}).get("reuse", {})
    outcomes = {name: state["statuses"].get(reuse.get(name, name), {}).get("status")
                for name in (REGRESSION_NAME, *STEP_C_UNITS)}
    return {"completed_units": sorted(n for n, s in outcomes.items() if s == "completed" and n != REGRESSION_NAME),
            "failed_units": sorted(n for n, s in outcomes.items() if s not in (None, "completed")),
            "not_started": sorted(n for n, s in outcomes.items() if s is None and n not in state["launches"]
                                  and n != REGRESSION_NAME),
            "regression": {"status": outcomes[REGRESSION_NAME],
                           "gate_passed": state["statuses"].get(REGRESSION_NAME, {}).get("gate_passed")},
            "interrupted_lanes": interrupted_lanes(state)}


def next_hint(state: dict[str, Any], ended: dict[str, dict[str, Any]]) -> str:
    regression = state["statuses"].get(REGRESSION_NAME)
    if regression is not None and regression_blockers(state):
        return regression_blockers(state)[0]  # STOP: failed run or gate
    stranded = interrupted_lanes(state)
    if stranded:
        return (f"interrupted lanes {stranded}: 'recover-lane K' for each, then 'archive-failed UNIT' "
                "(the regression instead stops for diagnosis)")
    if REGRESSION_NAME in ended and regression is None:
        return "the regression did not start (refused or blocked; see its .out); fix the cause and run 'regression' again"
    failed = [name for name in STEP_C_UNITS if state["statuses"].get(name, {}).get("status") not in (None, "completed")]
    if failed:
        return (f"failed units {failed}: 'archive-failed UNIT' (after its lane is pulled), then relaunch it once, "
                "unchanged; a second failure stops step C for the user's decision")
    if regression is None:
        return "run 'regression' first (alone)"
    remaining = summary(state)["not_started"]
    if remaining:
        return f"refill free lanes ('launch UNIT@LANE'); not started yet: {remaining}"
    return "every step C unit has run: 'assess', then 'evidence' and vm-stop"


def cmd_status(remote: b.Remote) -> dict[str, Any]:
    state = vm_state(remote)
    reuse = (state.get("settings") or {}).get("reuse", {})
    units = {}
    for name in (REGRESSION_NAME, *STEP_C_UNITS):
        record = state["statuses"].get(reuse.get(name, name), {})
        launch = state["launches"].get(name, {})
        units[name] = {"status": record.get("status"), "launch": launch.get("state"), "lane": launch.get("lane"),
                       "claimed": name in state["claims"], "archived_attempts": state["archived"].get(name, 0),
                       "reused_from": reuse.get(name)}
        if record.get("status") is None and name != REGRESSION_NAME:
            units[name]["forecast_seconds"] = forecast_seconds(name, state["statuses"])
    return {"units": units, "regression_blockers": regression_blockers(state), "awaiting_pull": awaiting_pull(state),
            "lane_blockers": lane_problems(state, list(range(int(state["width"])))),
            "active_launches": sorted(active_launches(state)), "next": next_hint(state, {}),
            "settings": {k: (state.get("settings") or {}).get(k) for k in ("amp", "concurrent_lanes")}}


def cmd_archive_failed(remote: b.Remote, name: str) -> dict[str, Any]:
    require(name != REGRESSION_NAME, "A regression failure stops v3 for diagnosis; it is rerun only after the "
                                     "user's decision (use the runner directly then)")
    require(name in STEP_C_UNITS, f"{name} is not a step C unit")
    state = vm_state(remote)
    require(name not in active_launches(state), f"{name} still has a running launch; wait for it first")
    # Archiving moves files that a pending transfer lists, so that pull could never verify again.
    require(not awaiting_pull(state), f"lanes {awaiting_pull(state)} await their host pull; run 'pull' first")
    stranded = [k for k, run in interrupted_lanes(state).items() if run == name]
    require(not stranded, f"{name} was interrupted in lane{stranded[0] if stranded else ''}; run 'recover-lane "
                          f"{stranded[0] if stranded else ''}' first")
    out = remote.run(f"cd {shlex.quote(b.REMOTE_CODE)} && {b.REMOTE_PY} run_pmm_v3_campaign.py --action archive-failed "
                     f"--campaign-dir {b.REMOTE_V3} --unit {shlex.quote(name)}").stdout
    return json.loads(out)


RECOVER_LANE = r"""
import importlib.util, json, sys
from pathlib import Path
code, lane_root, durable, session_id, deadline, started, max_seconds = sys.argv[1:8]
spec = importlib.util.spec_from_file_location("pmm_execution_recovery", Path(code) / "src/benchmarking/pmm_execution.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
policy = module.ExecutionPolicy(deadline_unix=float(deadline), max_total_seconds=float(max_seconds),
                                allocation_started_unix=float(started))
execution = module.CampaignExecution(Path(lane_root), policy=policy, session_id=session_id,
                                     durable_root=Path(durable), persistence_mode="host_pull")
try:
    with execution:
        print(json.dumps({"recovered": False, "message": "nothing to recover"}))
except module.ExecutionBlocked as exc:
    print(json.dumps({"recovered": True, "message": str(exc)}))
"""


def cmd_recover_lane(remote: b.Remote, session: dict[str, Any], lane: int) -> dict[str, Any]:
    """Run the lane's own recovery (benchmarking.pmm_execution, unchanged): an interrupted unit is recorded as
    'interrupted' and its artifacts are offered for a host pull, which is then verified and acknowledged."""
    state = vm_state(remote)
    stranded = interrupted_lanes(state)
    require(lane in stranded, f"lane{lane} holds no interrupted unit (lock free, child dead, no live launch); "
                              "nothing to recover")
    out = remote.run(f"{b.REMOTE_PY} -c {shlex.quote(RECOVER_LANE)} {shlex.quote(b.REMOTE_CODE)} "
                     f"{b.REMOTE_V3}/lanes/lane{lane} {shlex.quote(str(b.DURABLE / f'lane{lane}'))} "
                     f"{shlex.quote(session['session_id'])} {float(session['deadline'])!r} "
                     f"{float(session['allocation_started'])!r} {float(session['max_seconds'])!r}").stdout
    recovered = json.loads(out)
    require(recovered["recovered"], f"lane{lane} recovery did not record the interrupted unit: {recovered}")
    pulled = b.cmd_pull(remote, [lane])
    name = stranded[lane]
    after = ("the regression failed: STOP for diagnosis" if name == REGRESSION_NAME
             else f"'archive-failed {name}', then relaunch it once, unchanged")
    return {"lane": lane, "unit": name, "recovery": recovered, "pull": pulled, "next": after}


def cmd_assess(remote: b.Remote, *, evidence_root: Path | None = None) -> dict[str, Any]:
    """Run the frozen step C assessor on the VM (validation predictions only) and keep a checked copy."""
    out = remote.run(f"cd {shlex.quote(b.REMOTE_CODE)} && {b.REMOTE_PY} pmm_v3_assessment.py "
                     f"--campaign-dir {b.REMOTE_V3} --step C").stdout
    remote_dir = json.loads(out)["assessment"]
    require(remote_dir.startswith(f"{b.REMOTE_V3}/assessments/"), f"unexpected assessment path {remote_dir}")
    data = remote.transport.read_bytes(f"{remote_dir}/assessment.json")
    require(data is not None, "the assessor wrote no assessment.json")
    digest = remote.run(f"sha256sum {shlex.quote(remote_dir + '/assessment.json')}").stdout.split()[0]
    require(hashlib.sha256(data).hexdigest() == digest, "the copied assessment differs from the VM file")
    target = (evidence_root or EVIDENCE) / "assessments" / Path(remote_dir).name / "assessment.json"
    target.parent.mkdir(parents=True, exist_ok=False)
    target.write_bytes(data)
    return {"remote": remote_dir, "copy": str(target), "sha256": digest, "result": json.loads(data)["result"]}


def cmd_evidence(remote: b.Remote, *, destination: Path | None = None, allow_running: bool = False) -> dict[str, Any]:
    """The step-B evidence copy with step C patterns. The copy is always made; the command then fails while a
    unit still runs or a lane awaits its pull, because a vm-stop now would kill a paid fit or strand a transfer."""
    state = vm_state(remote)
    copied = b.cmd_evidence(remote, destination=destination, patterns=EVIDENCE_PATTERNS, evidence_root=EVIDENCE)
    unsafe = {"active_launches": sorted(active_launches(state)), "awaiting_pull": awaiting_pull(state),
              "interrupted_lanes": interrupted_lanes(state),
              "busy_lanes": sorted(int(lane[4:]) for lane, info in state["lanes"].items() if lane_busy(info))}
    require(allow_running or not any(unsafe.values()),
            f"Evidence copied to {copied['evidence_dir']}, but it is NOT safe to vm-stop yet: {unsafe} "
            "('wait' / 'pull' / 'recover-lane' first, or --allow-running for an emergency stop)")
    return {**copied, **unsafe, "safe_to_stop": not any(unsafe.values())}


def cmd_units() -> dict[str, Any]:
    return {"regression": {"unit": REGRESSION_NAME, "forecast_seconds": REGRESSION_FIT_SECONDS, "lane": 0,
                           "rule": "first and alone; a failed gate stops step C for diagnosis"},
            "units": list(STEP_C_UNITS), "suggested_order": list(SUGGESTED_ORDER),
            "forecast_seconds": {"warm": FIT_SECONDS, "cold_cache_extra": COLD_CACHE_SECONDS},
            "note": "the late-fusion four_class baseline is the reused step-B full-fp32 run (execution_settings.json)"}


def parse_request(item: str) -> tuple[str, int]:
    name, separator, lane = item.rpartition("@")
    require(bool(separator) and bool(name) and lane.isdigit(), f"Write each unit as UNIT@LANE, not {item!r}")
    return name, int(lane)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("units", "session", "status", "regression", "launch", "wait", "pull",
                                            "recover-lane", "archive-failed", "assess", "evidence"))
    parser.add_argument("items", nargs="*", help="launch: UNIT@LANE ...; wait: optional UNIT ...; archive-failed: "
                                                 "UNIT; recover-lane: K")
    parser.add_argument("--fit-seconds", type=float, help="launch / regression: admission forecast for every unit")
    parser.add_argument("--no-wait", action="store_true", help="launch / regression: return after the launch")
    parser.add_argument("--any", action="store_true", help="wait: return after the first unit ends")
    parser.add_argument("--lanes", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--allow-running", action="store_true", help="evidence: emergency copy while units run")
    args = parser.parse_args(argv)
    if args.command == "units":
        print(json.dumps(cmd_units(), indent=2))
        return 0
    session = b.read_session()
    if args.command == "session":
        print(json.dumps(session, indent=2, sort_keys=True))
        return 0
    remote = b.Remote(b.check_ssh_config())
    if args.command == "launch":
        require(bool(args.items), "launch needs UNIT@LANE ...")
    if args.command == "archive-failed":
        require(len(args.items) == 1, "archive-failed needs one unit")
    if args.command == "recover-lane":
        require(len(args.items) == 1 and args.items[0].isdigit(), "recover-lane needs one lane number")
    actions = {
        "status": lambda: cmd_status(remote),
        "regression": lambda: cmd_regression(remote, session, fit_seconds=args.fit_seconds, wait=not args.no_wait),
        "launch": lambda: cmd_launch(remote, session, [parse_request(item) for item in args.items],
                                     fit_seconds=args.fit_seconds, wait=not args.no_wait),
        "wait": lambda: cmd_wait(remote, session, names=args.items or None, any_unit=args.any),
        "pull": lambda: b.cmd_pull(remote, args.lanes),
        "recover-lane": lambda: cmd_recover_lane(remote, session, int(args.items[0])),
        "archive-failed": lambda: cmd_archive_failed(remote, args.items[0]),
        "assess": lambda: cmd_assess(remote),
        "evidence": lambda: cmd_evidence(remote, allow_running=args.allow_running),
    }
    print(json.dumps(actions[args.command](), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (LauncherError, b.host_pull.HostPullError) as exc:
        print(f"step-C launcher refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
