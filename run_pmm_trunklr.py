#!/usr/bin/env python3
"""VM-side runner of the trunk-learning-rate comparison (campaign ``pmm_trunk_lr_2026-10``).

Rules: docs/campaigns/pmm_trunk_lr_2026-10/trunklr_assessment_spec.json. The comparison runs the v3-selected cell
``gvp_late_fusion__four_class__baseline`` on folds 0-4 with seed 42 and ``--gvp-learning-rate 1e-4`` instead of the
frozen ``3e-4``. Everything else (base learning rate 3e-5, optimizer groups, 50-epoch cosine schedule, terminal
checkpoint, inputs, class weights, precision, lanes) is the control's.

This file is additive, like run_pmm_v3_sched10.py: the hash-frozen v3 runner files are imported and never changed.
The frozen ``run_unit`` admits a non-probe unit only under its own unit name and recipe, so this runner repeats its
admission sequence with the frozen building blocks (campaign verification, command builder, claim, admission,
execution, verification, independent replay and persistence) under the run names below. The frozen command builder
has no learning-rate parameter, so ``build`` replaces the one ``--gvp-learning-rate 3e-4`` pair it emits and
recomputes the identity's ``resolved_config_sha256`` with the builder's own functions. The workstation launcher
(pmm_trunklr.py) copies this file next to the frozen runner in the extension code directory; its SHA-256 is bound in
the launch manifest and in every run's identity (``lr_extension``).

Actions:
  plan   CPU, read-only: the five run names, each training argv and its difference from the control command.
  run    admit and run one comparison fit in one lane (the allocation fields of run_pmm_v3_campaign.py).

No action provisions or stops compute, reads held-out data, or retries a fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import pmm_v3_campaign as v3

CAMPAIGN_ID = "pmm_trunk_lr_2026-10"
LABEL = "trunklr1e4"
FAMILY, TARGET, RECIPE, SEED = "gvp_late_fusion", "four_class", "baseline", 42
FOLDS = (0, 1, 2, 3, 4)
CANDIDATE_GVP_LR = "1e-4"
GVP_LR_FLAG = "--gvp-learning-rate"
# The only trainer settings in which a comparison fit may differ from its control.
ALLOWED_CONFIG_DIFFERENCES = frozenset({"gvp_learning_rate", "run_name", "campaign_run_identity"})
ALLOWED_IDENTITY_DIFFERENCES = frozenset({"resolved_config_sha256", "lr_extension"})
RUNTIME_FIELDS = ("session_id", "execution_deadline", "allocation_started", "execution_max_seconds",
                  "estimated_fit_seconds", "durable_root", "persistence_mode")


def control_gvp_lr() -> str:
    """The frozen family setting the control fits ran with ('3e-4')."""
    return v3.v2.FAMILIES[FAMILY]["gvp_lr"]


def run_name(fold: int) -> str:
    v3.require(fold in FOLDS, f"The comparison runs folds {FOLDS} only")
    return f"{FAMILY}__{TARGET}__{LABEL}__fold{fold}__seed{SEED}"


def fold_of(name: str) -> int:
    """The fold of one of the five comparison run names; anything else is refused."""
    for fold in FOLDS:
        if name == run_name(fold):
            return fold
    raise ValueError(f"{name!r} is not a run of the trunk-LR comparison ({', '.join(run_name(f) for f in FOLDS)})")


def control_unit(fold: int) -> v3.Unit:
    """The control unit of the same fold (fold 0 resolves to the reused step-B run through execution_settings)."""
    return v3.Unit(FAMILY, TARGET, RECIPE, fold, SEED)


def runner_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def lr_binding() -> dict[str, Any]:
    return {"campaign_id": CAMPAIGN_ID, "runner_file": Path(__file__).name, "runner_sha256": runner_sha256(),
            "gvp_learning_rate": CANDIDATE_GVP_LR, "control_gvp_learning_rate": control_gvp_lr()}


def resolved_config_sha256(command: list[str]) -> str:
    """pmm_v3_campaign.build_command's identity hash of a training argv (without the run identity)."""
    from training.config import config_to_payload, parse_args

    resolved = config_to_payload(parse_args(command[3:]))
    return v3.stable_hash({k: v for k, v in resolved.items() if k not in v3.v2.NON_IDENTITY_CONFIG_KEYS})


def build(paths: v3.V3Paths, fold: int, *, python_bin: str, train_dir: Path, esm_dir: Path | None, device: str,
          lane: int, amp: bool, load_workers: int | None = None) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    """The frozen command builder under the comparison's run name, with the one GVP learning-rate value replaced, the
    identity hash recomputed and the extension bound into the run identity (stored verbatim by the trainer)."""
    command, env, identity = v3.build_command(paths, control_unit(fold), python_bin=python_bin, train_dir=train_dir,
                                              esm_dir=esm_dir, device=device, lane=lane, load_workers=load_workers,
                                              amp=amp, run_name=run_name(fold))
    v3.require(command[-2] == "--campaign-run-identity" and json.loads(command[-1]) == identity,
               "The frozen command builder no longer ends with the run identity")
    command = command[:-2]
    v3.require(resolved_config_sha256(command) == identity["resolved_config_sha256"],
               "The identity hash recomputation no longer matches the frozen command builder")
    positions = [i for i, token in enumerate(command) if token == GVP_LR_FLAG]
    v3.require(len(positions) == 1 and command[positions[0] + 1] == control_gvp_lr(),
               f"Expected exactly one '{GVP_LR_FLAG} {control_gvp_lr()}' in the frozen command")
    command[positions[0] + 1] = CANDIDATE_GVP_LR
    identity = {**identity, "resolved_config_sha256": resolved_config_sha256(command), "lr_extension": lr_binding()}
    identity = json.loads(json.dumps(identity, sort_keys=True))
    return command + ["--campaign-run-identity", json.dumps(identity, sort_keys=True)], env, identity


def differences_from_control(paths: v3.V3Paths, fold: int, *, python_bin: str, train_dir: Path,
                             esm_dir: Path | None, device: str, lane: int, amp: bool,
                             load_workers: int | None = None) -> dict[str, Any]:
    """Resolved trainer configuration and identity of the comparison fit against its control."""
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
            "gvp_learning_rate": [b.get("gvp_learning_rate"), a.get("gvp_learning_rate")],
            "learning_rate": [b.get("learning_rate"), a.get("learning_rate")]}


def expected_lr_difference(diff: dict[str, Any]) -> bool:
    return (not diff["unexpected"] and diff["gvp_learning_rate"] == [float(control_gvp_lr()), float(CANDIDATE_GVP_LR)]
            and diff["learning_rate"][0] == diff["learning_rate"][1])


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
    v3.require(expected_lr_difference(diff), f"{name} would differ from its control in {diff['unexpected']} "
                                             f"(GVP learning rate {diff['gvp_learning_rate']})")
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
                                           "control_unit": unit.name, "lr_extension": identity["lr_extension"],
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
    parser.add_argument("--unit", help="run: one comparison run name, gvp_late_fusion__four_class__trunklr1e4__foldK__seed42")
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
        print(json.dumps({"lr_extension": lr_binding(), "fits": rows,
                          "all_expected": all(expected_lr_difference(row) for row in rows),
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
        print(f"trunk-LR comparison refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except ExecutionBlocked as exc:  # as run_pmm_v3_campaign.py: an admitted fit stopped this way is 'interrupted'
        print(f"trunk-LR comparison blocked: {exc}", file=sys.stderr)
        raise SystemExit(3)
    except PersistenceError as exc:
        print(f"trunk-LR comparison persistence failed: {exc}", file=sys.stderr)
        raise SystemExit(4)
