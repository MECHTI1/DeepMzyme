#!/usr/bin/env python3
"""VM-side runner of the v3 10-epoch schedule comparison (extension 2, ``v3-ext2-sched10``).

Rules: docs/campaigns/pmm_ion_metal_v3/sched10_assessment_spec.json (handoff section 11, log v3-039). The comparison
runs the selected cell ``gvp_late_fusion__four_class__baseline`` on folds 1-4 with seed 42 and ``--epochs 10``:
a complete 10-epoch cosine schedule with its terminal checkpoint. Everything else is the 50-epoch control's.

This file is additive. The hash-frozen runner files (pmm_v3_campaign.py, run_pmm_v3_campaign.py, pmm_v3_probes.py,
pmm_v3_speed_report.py) are imported and never changed: the frozen ``run_unit`` admits a non-probe unit only under
its own unit name and 50 epochs, so this runner repeats its admission sequence with the frozen building blocks
(campaign verification, command builder, claim, admission, execution, verification, independent replay and
persistence) under the run names below. The workstation launcher (pmm_v3_sched10.py) copies this file next to the
frozen runner in the extension code directory; its SHA-256 is bound in the launch manifest and in every run's
identity (``schedule_extension``). Archive, replay and status use run_pmm_v3_campaign.py unchanged (by run name).

Actions:
  plan   CPU, read-only: the four run names, each training argv and its difference from the 50-epoch command.
  run    admit and run one comparison fit in one lane (the allocation fields of run_pmm_v3_campaign.py).

No action provisions or stops compute, reads held-out data, or retries a fit.
"""

from __future__ import annotations


import sys as _sys
from pathlib import Path as _Path
for _p in (str(_Path(__file__).resolve().parents[2]), str(_Path(__file__).resolve().parents[2] / "src")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
from project_paths import RELOCATED_FILES, install_relocated_module_aliases, resolve_existing_path, resolve_repo_file
install_relocated_module_aliases()
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import pmm_v3_campaign as v3

SCHEDULE_ID = "v3-ext2-sched10"
LABEL = "sched10"
EPOCHS = 10
FAMILY, TARGET, RECIPE, SEED = "gvp_late_fusion", "four_class", "baseline", 42
FOLDS = (1, 2, 3, 4)
# The only trainer settings in which a comparison fit may differ from its 50-epoch control.
ALLOWED_CONFIG_DIFFERENCES = frozenset({"epochs", "run_name", "campaign_run_identity"})
ALLOWED_IDENTITY_DIFFERENCES = frozenset({"epochs", "resolved_config_sha256", "schedule_extension"})
RUNTIME_FIELDS = ("session_id", "execution_deadline", "allocation_started", "execution_max_seconds",
                  "estimated_fit_seconds", "durable_root", "persistence_mode")


def run_name(fold: int) -> str:
    v3.require(fold in FOLDS, f"The comparison runs folds {FOLDS} only")
    return f"{FAMILY}__{TARGET}__{LABEL}__fold{fold}__seed{SEED}"


def fold_of(name: str) -> int:
    """The fold of one of the four comparison run names; anything else is refused."""
    for fold in FOLDS:
        if name == run_name(fold):
            return fold
    raise ValueError(f"{name!r} is not a run of the 10-epoch comparison ({', '.join(run_name(f) for f in FOLDS)})")


def control_unit(fold: int) -> v3.Unit:
    """The 50-epoch control unit of the same fold (its command is the comparison's except the epochs)."""
    return v3.Unit(FAMILY, TARGET, RECIPE, fold, SEED)


def runner_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def schedule_binding() -> dict[str, Any]:
    return {"id": SCHEDULE_ID, "runner_file": "run_pmm_v3_sched10.py", "runner_sha256": runner_sha256(),
            "epochs": EPOCHS, "control_epochs": int(v3.PROFILE["epochs"])}


def build(paths: v3.V3Paths, fold: int, *, python_bin: str, train_dir: Path, esm_dir: Path | None, device: str,
          lane: int, amp: bool, load_workers: int | None = None) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    """The frozen command builder at 10 epochs under the comparison's run name, with the extension bound into the
    run identity (the trainer stores the identity verbatim; it is excluded from the resolved-config hash)."""
    command, env, identity = v3.build_command(paths, control_unit(fold), python_bin=python_bin, train_dir=train_dir,
                                              esm_dir=esm_dir, device=device, lane=lane, load_workers=load_workers,
                                              epochs=EPOCHS, amp=amp, run_name=run_name(fold))
    v3.require(command[-2] == "--campaign-run-identity" and json.loads(command[-1]) == identity,
               "The frozen command builder no longer ends with the run identity")
    identity = json.loads(json.dumps({**identity, "schedule_extension": schedule_binding()}, sort_keys=True))
    return command[:-1] + [json.dumps(identity, sort_keys=True)], env, identity


def differences_from_control(paths: v3.V3Paths, fold: int, *, python_bin: str, train_dir: Path,
                             esm_dir: Path | None, device: str, lane: int, amp: bool,
                             load_workers: int | None = None) -> dict[str, Any]:
    """Resolved trainer configuration and identity of the comparison fit against its 50-epoch control."""
    from training.config import config_to_payload, parse_args

    common = dict(python_bin=python_bin, train_dir=train_dir, esm_dir=esm_dir, device=device, lane=lane,
                  load_workers=load_workers, amp=amp)
    mine, _, identity = build(paths, fold, **common)
    theirs, _, control_identity = v3.build_command(paths, control_unit(fold), **common)
    a, b = config_to_payload(parse_args(mine[3:])), config_to_payload(parse_args(theirs[3:]))
    config = sorted(key for key in set(a) | set(b) if a.get(key) != b.get(key))
    ident = sorted(key for key in set(identity) | set(control_identity) if identity.get(key) != control_identity.get(key))
    return {"run_name": run_name(fold), "control_unit": control_unit(fold).name,
            "config_differences": {key: [b.get(key) if key != "campaign_run_identity" else "<control identity>",
                                         a.get(key) if key != "campaign_run_identity" else "<comparison identity>"]
                                   for key in config},
            "identity_differences": ident,
            "unexpected": sorted(set(config) - ALLOWED_CONFIG_DIFFERENCES)
            + sorted(set(ident) - ALLOWED_IDENTITY_DIFFERENCES),
            "epochs": [b.get("epochs"), a.get("epochs")]}


def run(paths: v3.V3Paths, fold: int, *, lane: int, train_dir: Path, esm_dir: Path | None, python_bin: str,
        device: str, execution_policy: dict[str, Any], load_workers: int | None = None) -> dict[str, Any]:
    """v3.run_unit's admission sequence for one comparison fit (pmm_v3_campaign.run_unit and _run_admitted)."""
    from benchmarking.pmm_execution import CampaignExecution, ExecutionPolicy
    from training.access_guard import install_forbidden_read_guard

    preflight_started = time.time()
    install_forbidden_read_guard(v3.v2.forbidden_read_roots(train_dir))
    v3.require(Path(train_dir).name == "train", "Only the training-side directory named train is allowed")
    name, unit = run_name(fold), control_unit(fold)
    amp = v3.resolve_amp(paths, probe=None, amp=None)  # the recorded step-B setting
    existing = v3.unit_artifacts(paths, name)
    if existing:
        raise ValueError(f"Unit artifacts exist ({existing[0]}); runs are never retried automatically")
    v3.require(paths.extension.exists(), "The comparison runs on the extension-1 campaign root only")
    v3.verify_campaign(paths, esm_dir=esm_dir)  # runner, recipes, extension record, frozen inputs, source tree, ESMC
    v3.require_frozen_spec(paths)
    diff = differences_from_control(paths, fold, python_bin=python_bin, train_dir=train_dir, esm_dir=esm_dir,
                                    device=device, lane=lane, amp=amp, load_workers=load_workers)
    v3.require(not diff["unexpected"] and diff["epochs"] == [v3.PROFILE["epochs"], EPOCHS],
               f"{name} would differ from its 50-epoch control in {diff['unexpected']}")
    command, env, identity = build(paths, fold, python_bin=python_bin, train_dir=train_dir, esm_dir=esm_dir,
                                   device=device, lane=lane, amp=amp, load_workers=load_workers)
    v3.require_retry_identity(paths, name, identity)
    lane_root = paths.lane(lane)
    policy = ExecutionPolicy(deadline_unix=execution_policy["deadline_unix"],
                             max_total_seconds=execution_policy["max_total_seconds"],
                             allocation_started_unix=execution_policy.get("allocation_started_unix"))
    execution = CampaignExecution(lane_root, policy=policy, session_id=execution_policy["session_id"],
                                  durable_root=Path(execution_policy["durable_root"]) / lane_root.name,
                                  persistence_mode=execution_policy["persistence_mode"])
    claim = v3.claim_run(paths, name, lane=lane, identity=identity, session_id=execution_policy["session_id"])
    admitted = False
    try:
        with execution:
            execution.admit(name, float(execution_policy["estimated_fit_seconds"]))
            admitted = True
            preflight_seconds = time.time() - preflight_started  # as run_unit: checks before the fit count too
            result = v3.execute_unit(paths, lane, unit, command, env, identity, execution=execution,
                                     python_bin=python_bin, device=device, run_name=name)
            status_path = lane_root / f"run_status_{name}.json"
            v3.v2.write_json(status_path, {**result, "identity": identity, "lane": lane, "unit": name, "probe": None,
                                           "control_unit": unit.name, "schedule_extension": identity["schedule_extension"],
                                           "preflight_seconds": preflight_seconds})
            execution.record_result(name, result["status"], result["elapsed_seconds"])
            execution.persist([path for path in (lane_root / "runs" / name, lane_root / "runs" / f"{name}.log",
                                                 lane_root / "commands" / f"{name}.json", status_path)
                               if path.exists()])
    finally:
        if not admitted:  # nothing ran: the lane or the admission refused, so the name is released
            claim.unlink(missing_ok=True)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--action", choices=("plan", "run"), required=True)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--unit", help="run: one comparison run name, gvp_late_fusion__four_class__sched10__foldK__seed42")
    parser.add_argument("--train-dir", type=Path)
    parser.add_argument("--esm-dir", type=Path)
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load-workers", type=int)
    parser.add_argument("--session-id")
    for name in ("execution-deadline", "allocation-started", "execution-max-seconds", "estimated-fit-seconds"):
        parser.add_argument("--" + name, type=float)
    parser.add_argument("--durable-root", type=Path)
    parser.add_argument("--persistence-mode", choices=("mounted", "host_pull"))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    paths = v3.V3Paths(args.campaign_dir)
    if args.action == "plan":
        v3.require(args.train_dir is not None, "plan needs --train-dir (the command builder binds it)")
        amp = bool((v3.read_execution_settings(paths) or {}).get("amp", False))
        rows = [differences_from_control(paths, fold, python_bin=args.python_bin, train_dir=args.train_dir.resolve(),
                                         esm_dir=args.esm_dir.resolve() if args.esm_dir else None, device=args.device,
                                         lane=index % 3, amp=amp, load_workers=args.load_workers)
                for index, fold in enumerate(FOLDS)]
        print(json.dumps({"schedule": schedule_binding(), "fits": rows,
                          "note": "Read-only plan; every run needs explicit allocation fields and the user's OK."},
                         indent=2, sort_keys=True))
        return 0
    missing = ["--" + key.replace("_", "-") for key in RUNTIME_FIELDS if getattr(args, key) is None]
    v3.require(not missing, "run needs allocation/persistence fields: " + ", ".join(missing))
    v3.require(args.unit is not None and args.train_dir is not None and args.esm_dir is not None,
               "run needs --unit, --train-dir and --esm-dir")
    policy = {"session_id": args.session_id, "deadline_unix": args.execution_deadline,
              "allocation_started_unix": args.allocation_started, "max_total_seconds": args.execution_max_seconds,
              "estimated_fit_seconds": args.estimated_fit_seconds, "durable_root": str(args.durable_root.resolve()),
              "persistence_mode": args.persistence_mode}
    result = run(paths, fold_of(args.unit), lane=args.lane, train_dir=args.train_dir.resolve(),
                 esm_dir=args.esm_dir.resolve(), python_bin=args.python_bin, device=args.device,
                 execution_policy=policy, load_workers=args.load_workers)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    from benchmarking.pmm_execution import ExecutionBlocked, PersistenceError

    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"v3 10-epoch comparison refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except ExecutionBlocked as exc:  # as run_pmm_v3_campaign.py: an admitted fit stopped this way is 'interrupted'
        print(f"v3 10-epoch comparison blocked: {exc}", file=sys.stderr)
        raise SystemExit(3)
    except PersistenceError as exc:
        print(f"v3 10-epoch comparison persistence failed: {exc}", file=sys.stderr)
        raise SystemExit(4)
