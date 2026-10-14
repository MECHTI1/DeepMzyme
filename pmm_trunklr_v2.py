#!/usr/bin/env python3
"""Launcher v2 of the trunk-learning-rate comparison: the frozen launcher (pmm_trunklr.py, unchanged) plus the
session authorization of pmm_trunklr_safeguards_v2 at submission time (standard library only).

Added 2026-10-13 at the campaign's closeout maintenance; it was not used for the original session. No live policy or
session binding exists for this campaign: its authorized session is used and its five fits are complete.

Commands (everything else is passed to pmm_trunklr.py unchanged):
  launch UNIT@LANE ... [--fit-seconds S] [--no-wait]
                          every frozen check (readiness, manifest, ceiling, authorization, spending, unit/lane/pull
                          state, admission) in planning; then, immediately before submission and under a shared hold
                          of the controller lock (~/deepmzyme-vm/state/controller.lock), the controller session is read
                          again, checked against the v2 launch policy and its bound sessions, bound once if it is new,
                          and admitted; only then is the launch script sent
  policy                  offline: the recorded policy, the bound sessions and the verdict for the current session
  record-policy --log-entry tlr-NNN --decision TEXT --max-sessions N --max-session-seconds S --expires-at UTC
                [--units UNIT ...]
                          once, only after the user's decision: the v2 launch policy (write-once)

Status, wait, pull, recover-lane, archive-failed, evidence, deploy, session, units, gates and spend go to the frozen
launcher and never need the policy, so evidence collection and recovery stay possible after launch authorization is
exhausted. A refusal sends nothing to the VM.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pmm_trunklr as tl  # noqa: E402
import pmm_trunklr_safeguards_v2 as sg  # noqa: E402

b, c = tl.b, tl.c
require = tl.require
V2_COMMANDS = ("launch", "policy", "record-policy")


def start_v2(remote: b.Remote, session: dict[str, Any], planned: list[dict[str, Any]], *, step: c.Step,
             records: Path = tl.RECORDS, wait: bool = True, root: Path = ROOT,
             read_session: Callable[[], dict[str, Any]] = b.read_session,
             lock: Callable[[], Any] = sg.controller_lock, now: Callable[[], float] = time.time,
             sleep=time.sleep) -> dict[str, Any]:
    """Re-read the session under the controller lock, apply the v2 policy, bind a new session once, then submit
    through the frozen start (its admission check and launch script unchanged)."""
    names = [item["unit"] for item in planned]
    with lock():
        fresh = read_session()
        require(fresh.get("session_id") == session.get("session_id")
                and fresh.get("instance_id") == session.get("instance_id"),
                "The controller session changed since this command started; nothing was launched (run it again)")
        problems, new_binding = sg.launch_gate(records, units=names, session=fresh, now=now(), root=root)
        require(not problems, "Nothing was launched (v2 session authorization):\n  " + "\n  ".join(problems))
        b.admission_check(fresh, max(item["fit_seconds"] for item in planned))
        bound = None if new_binding is None else sg.bind_session_once(records, new_binding)
        result = tl.start(remote, fresh, planned, wait=False, step=step, sleep=sleep)
    result["session_binding"] = bound or "already bound"
    if wait:
        result.pop("next", None)
        result.update(c.cmd_wait(remote, fresh, names=names, sleep=sleep, step=step))
    return result


def cmd_launch_v2(remote: b.Remote, session: dict[str, Any], requests: list[tuple[str, int]], *, step: c.Step,
                  records: Path = tl.RECORDS, fit_seconds: float | None = None, wait: bool = True,
                  **gates) -> dict[str, Any]:
    planned = c.plan_units(requests, c.vm_state(remote, step=step), fit_seconds=fit_seconds, step=step)
    return start_v2(remote, session, planned, step=step, records=records, wait=wait, **gates)


def cmd_policy(records: Path = tl.RECORDS, *, read_session: Callable[[], dict[str, Any]] = b.read_session,
               root: Path = ROOT) -> dict[str, Any]:
    paths = sg.policy_paths(records)
    bindings, history = sg.read_bindings(paths["sessions"])
    try:
        session = read_session()
    except (tl.LauncherError, OSError, ValueError, KeyError) as exc:
        session = None
        verdict = [f"no usable controller session ({exc})"]
    else:
        units = (tl.d.read_record(paths["policy"]) or {}).get("authorized_units") or []
        verdict, _ = sg.launch_gate(records, units=list(units), session=session, now=time.time(), root=root)
    return {"policy": tl.d.read_record(paths["policy"]), "bound_sessions": bindings, "history_problems": history,
            "current_session": None if session is None else session.get("session_id"),
            "current_session_problems": verdict}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in V2_COMMANDS:
        return tl.main(argv)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=V2_COMMANDS)
    parser.add_argument("items", nargs="*", help="launch: UNIT@LANE ...")
    parser.add_argument("--fit-seconds", type=float)
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--log-entry")
    parser.add_argument("--decision", default="")
    parser.add_argument("--max-sessions", type=int)
    parser.add_argument("--max-session-seconds", type=float)
    parser.add_argument("--expires-at")
    parser.add_argument("--units", nargs="+")
    args = parser.parse_args(argv)
    paths = sg.policy_paths()
    if args.command == "policy":
        print(json.dumps(cmd_policy(), indent=2, sort_keys=True, default=str))
        return 0
    if args.command == "record-policy":
        require(None not in (args.max_sessions, args.max_session_seconds, args.expires_at),
                "record-policy needs --max-sessions, --max-session-seconds and --expires-at")
        manifest = tl.d.read_record(paths["manifest"])
        require(manifest is not None and not tl.verify_manifest(manifest), "The recorded manifest does not verify")
        record = sg.policy_record(records=tl.RECORDS, log_entry=args.log_entry, decision=args.decision,
                                  units=args.units or list(manifest["units"]), max_sessions=args.max_sessions,
                                  max_session_seconds=args.max_session_seconds, expires_at=args.expires_at)
        print(json.dumps(tl.write_json_once(paths["policy"], record), indent=2, sort_keys=True))
        return 0
    require(bool(args.items), "launch needs UNIT@LANE ...")
    session = b.read_session()
    remote = b.Remote(b.check_ssh_config())
    step = tl.build_step(vm_runner=lambda: tl.remote_runner_sha256(remote))
    result = cmd_launch_v2(remote, session, [c.parse_request(item) for item in args.items], step=step,
                           fit_seconds=args.fit_seconds, wait=not args.no_wait)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (tl.LauncherError, b.host_pull.HostPullError) as exc:
        print(f"trunk-LR comparison launcher v2 refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
