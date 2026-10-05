#!/usr/bin/env python3
"""Command line for the PMM ion metal v3 campaign (see pmm_v3_campaign.py).

Actions:
  prepare  CPU: freeze a new v3 campaign root from the v2 cohort and a frozen fold set.
  plan     CPU, read-only: list a step's units and their training argv.
  run      Admit and run one unit in one lane (needs explicit allocation fields).
  replay   Independent validation replay of one completed run directory.
  status   CPU, read-only: terminal status of every unit across lanes.

No action provisions or stops compute, reads held-out data, or retries a unit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pmm_v3_campaign as v3

RUNTIME_FIELDS = ("session_id", "execution_deadline", "allocation_started", "execution_max_seconds",
                  "estimated_fit_seconds", "durable_root", "persistence_mode")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--action", choices=("prepare", "plan", "run", "regression", "replay", "status"),
                        required=True)
    parser.add_argument("--campaign-dir", type=Path)
    parser.add_argument("--v2-root", type=Path, help="prepare: frozen v2 context campaign root")
    parser.add_argument("--fold-dir", type=Path, help="prepare: frozen v3 fold-set directory")
    parser.add_argument("--train-dir", type=Path)
    parser.add_argument("--esm-dir", type=Path, help="Directory holding the frozen ESMC-600M payloads")
    parser.add_argument("--step", choices=("C", "D-A", "D-B", "D-combo", "E-neutral", "E-improvement"))
    parser.add_argument("--unit", help="run: one unit name, family__target__recipe__foldK__seedS")
    parser.add_argument("--family", choices=v3.FAMILIES)
    parser.add_argument("--recipe", help="plan D-combo: combo-a+b")
    parser.add_argument("--final-recipe", action="append", default=[],
                        help="plan E-improvement: FAMILY=RECIPE, repeatable")
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load-workers", type=int)
    parser.add_argument("--run-dir", type=Path, help="replay: completed run directory")
    parser.add_argument("--session-id")
    for name in ("execution-deadline", "allocation-started", "execution-max-seconds", "estimated-fit-seconds"):
        parser.add_argument("--" + name, type=float)
    parser.add_argument("--durable-root", type=Path)
    parser.add_argument("--persistence-mode", choices=("mounted", "host_pull"))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.action == "replay":
        from training.campaign_runtime import replay_campaign_run

        v3.require(args.run_dir is not None, "replay needs --run-dir")
        receipt = replay_campaign_run(args.run_dir.resolve(), device=args.device)
        print(json.dumps({"selected_epoch": receipt["selected_epoch"],
                          "prediction_rows_verified": receipt["prediction_rows_verified"]}, sort_keys=True))
        return 0
    v3.require(args.campaign_dir is not None, "--campaign-dir is required")
    paths = v3.V3Paths(args.campaign_dir)
    if args.action == "prepare":
        v3.require(args.v2_root is not None and args.fold_dir is not None, "prepare needs --v2-root and --fold-dir")
        manifest = v3.prepare_campaign(paths.root, v2_root=args.v2_root, fold_dir=args.fold_dir)
        print(json.dumps({key: manifest[key] for key in ("campaign_id", "fold_set", "frozen_source_tree_sha256",
                                                          "git_commit")}, indent=2, sort_keys=True))
        return 0
    if args.action == "status":
        print(json.dumps({name: record["status"] for name, record in v3.completed_units(paths).items()},
                         indent=2, sort_keys=True))
        return 0
    if args.action == "plan":
        v3.require(args.step is not None, "plan needs --step")
        finals = dict(item.split("=", 1) for item in args.final_recipe)
        units = v3.step_units(args.step, final_recipes=finals, combo=args.recipe, family=args.family)
        done = v3.completed_units(paths) if paths.lanes.is_dir() else {}
        rows = [{"unit": unit.name, "lane_suggestion": index % 3,
                 "existing_status": done.get(unit.name, {}).get("status")} for index, unit in enumerate(units)]
        print(json.dumps({"step": args.step, "n_units": len(units), "units": rows,
                          "note": "Read-only plan; every run needs explicit allocation fields and the user's OK."},
                         indent=2, sort_keys=True))
        return 0
    # run / regression
    missing = ["--" + key.replace("_", "-") for key in RUNTIME_FIELDS if getattr(args, key) is None]
    v3.require(not missing, f"{args.action} needs allocation/persistence fields: " + ", ".join(missing))
    v3.require(args.train_dir is not None, f"{args.action} needs --train-dir")
    policy = {"session_id": args.session_id, "deadline_unix": args.execution_deadline,
              "allocation_started_unix": args.allocation_started, "max_total_seconds": args.execution_max_seconds,
              "estimated_fit_seconds": args.estimated_fit_seconds, "durable_root": str(args.durable_root.resolve()),
              "persistence_mode": args.persistence_mode}
    esm_dir = args.esm_dir.resolve() if args.esm_dir else None
    if args.action == "regression":
        v3.require(args.v2_root is not None and esm_dir is not None, "regression needs --v2-root and --esm-dir")
        result = v3.run_regression(paths, v2_root=args.v2_root, lane=args.lane, train_dir=args.train_dir.resolve(),
                                   esm_dir=esm_dir, python_bin=args.python_bin, device=args.device,
                                   execution_policy=policy, load_workers=args.load_workers)
    else:
        v3.require(args.unit is not None, "run needs --unit")
        result = v3.run_unit(paths, v3.Unit.parse(args.unit), lane=args.lane, train_dir=args.train_dir.resolve(),
                             esm_dir=esm_dir, python_bin=args.python_bin, device=args.device,
                             load_workers=args.load_workers, execution_policy=policy)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"v3 campaign refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
