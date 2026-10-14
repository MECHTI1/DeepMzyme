#!/usr/bin/env python3
"""Plan, or execute one explicitly authorized metal-completion unit.

No provisioning, downloads, held-out evaluation or automatic retries. ``run``
requires the independently bound host authorization, controller session and
spending snapshots; a prepared scientific manifest alone cannot run a fit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import metal_completion_campaign as campaign


def require_manifest_input_paths(manifest: dict, *, train_dir: Path, esm_dir: Path) -> None:
    for key, supplied in (("train_dir", train_dir), ("esm_dir", esm_dir)):
        bound = manifest.get(key)
        campaign.require(isinstance(bound, str) and bool(bound.strip()),
                         f"Manifest must bind {key}")
        campaign.require(Path(supplied).resolve() == Path(bound).resolve(),
                         f"Supplied {key} does not match the manifest")


def run(*, manifest_path: Path, manifest_sha256: str, policy_path: Path, session_path: Path,
        spending_path: Path, unit: campaign.Unit, lane: int, train_dir: Path, esm_dir: Path,
        python_bin: str, device: str, load_workers: int | None = None) -> dict:
    import pmm_v3_campaign as v3
    import metal_completion_safety as safety
    from benchmarking.pmm_execution import CampaignExecution, ExecutionPolicy
    from training.access_guard import install_forbidden_read_guard

    started = time.time()
    campaign.require(Path(train_dir).name == "train", "Only the training-side directory named train is allowed")
    install_forbidden_read_guard(v3.v2.forbidden_read_roots(train_dir))
    manifest = campaign.verify_manifest(manifest_path, manifest_sha256, esm_dir=esm_dir)
    require_manifest_input_paths(manifest, train_dir=train_dir, esm_dir=esm_dir)
    campaign.require(unit.name in manifest["authorized_grid_units"], "Unit outside the scientific grid")
    parent = v3.V3Paths(Path(manifest["parent_campaign_dir"]))
    paths = campaign.isolated_paths(parent.root, Path(manifest["successor_root"]))
    campaign.require(not v3.unit_artifacts(paths, unit.name), "Unit artifacts exist; no automatic retries")
    campaign.require(len(v3.archived_attempts(paths, unit.name)) <= v3.MAX_RERUNS,
                     "Unit has exceeded its one unchanged rerun")
    runtime = safety.validate_worker_authorization(
        manifest_path=manifest_path, policy_path=policy_path, session_path=session_path,
        spending_path=spending_path, unit_name=unit.name, records_dir=paths.root)
    campaign.require(campaign.sha256_file(manifest_path) == manifest_sha256,
                     "Manifest changed while authorization was checked")
    campaign.require(type(lane) is int and 0 <= lane < runtime["lanes"], "Lane not authorized")
    amp = v3.resolve_amp(parent, probe=None, amp=None)
    command, env, identity = campaign.build_command(
        parent, unit, successor_root=paths.root, manifest_sha256=manifest_sha256,
        code_sha256=manifest["code_sha256"], python_bin=python_bin, train_dir=train_dir,
        esm_dir=esm_dir, device=device, lane=lane, amp=amp, load_workers=load_workers)
    v3.require_retry_identity(paths, unit.name, identity)
    lane_root = paths.lane(lane)
    campaign.require(not lane_root.is_symlink() and lane_root.resolve().is_relative_to(paths.root),
                     "Unsafe lane path")
    execution = CampaignExecution(
        lane_root, policy=ExecutionPolicy(deadline_unix=runtime["deadline_unix"],
                                         max_total_seconds=runtime["max_total_seconds"],
                                         allocation_started_unix=runtime["allocation_started_unix"]),
        session_id=runtime["session_id"], durable_root=Path(runtime["durable_root"]) / lane_root.name,
        persistence_mode=runtime["persistence_mode"])
    claim = v3.claim_run(paths, unit.name, lane=lane, identity=identity, session_id=runtime["session_id"])
    admitted = False
    try:
        with execution:
            execution.admit(unit.name, float(runtime["estimated_fit_seconds"]))
            admitted = True
            # Baseline is the immutable runner's checkpoint-rule template only.
            # The command/identity carry the actual successor recipe and seed.
            template = v3.Unit(campaign.FAMILY, unit.target, "baseline", unit.fold, 42)
            result = v3.execute_unit(paths, lane, template, command, env, identity, execution=execution,
                                     python_bin=python_bin, device=device, run_name=unit.name)
            status = lane_root / f"run_status_{unit.name}.json"
            v3.v2.write_json(status, {**result, "identity": identity, "unit": unit.name, "lane": lane,
                                      "session_id": runtime["session_id"],
                                      "authorization_policy_sha256": campaign.sha256_file(policy_path),
                                      "preflight_seconds": time.time() - started - result["elapsed_seconds"]})
            execution.record_result(unit.name, result["status"], result["elapsed_seconds"])
            execution.persist([path for path in (lane_root / "runs" / unit.name,
                                                 lane_root / "runs" / f"{unit.name}.log",
                                                 lane_root / "commands" / f"{unit.name}.json", status)
                               if path.exists()])
    finally:
        if not admitted:
            claim.unlink(missing_ok=True)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--action", choices=("plan", "run"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--esm-dir", type=Path, required=True)
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load-workers", type=int)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--session", type=Path)
    parser.add_argument("--spending", type=Path)
    args = parser.parse_args(argv)
    if args.action == "run" and not all((args.policy, args.session, args.spending)):
        parser.error("run requires --policy, --session and --spending; preparation is not authorization")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    unit = campaign.Unit.parse(args.unit)
    if args.action == "run":
        result = run(manifest_path=args.manifest, manifest_sha256=args.manifest_sha256,
                     policy_path=args.policy, session_path=args.session, spending_path=args.spending,
                     unit=unit, lane=args.lane, train_dir=args.train_dir, esm_dir=args.esm_dir,
                     python_bin=args.python, device=args.device, load_workers=args.load_workers)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "completed" else 1
    import pmm_v3_campaign as v3
    manifest = campaign.verify_manifest(args.manifest, args.manifest_sha256, esm_dir=args.esm_dir)
    require_manifest_input_paths(manifest, train_dir=args.train_dir, esm_dir=args.esm_dir)
    campaign.require(unit.name in manifest["authorized_grid_units"], "Unit outside the scientific grid")
    parent = v3.V3Paths(Path(manifest["parent_campaign_dir"]))
    command, env, identity = campaign.build_command(
        parent, unit, successor_root=Path(manifest["successor_root"]),
        manifest_sha256=args.manifest_sha256, code_sha256=manifest["code_sha256"],
        python_bin=args.python, train_dir=args.train_dir, esm_dir=args.esm_dir, device=args.device,
        lane=args.lane, amp=v3.resolve_amp(parent, probe=None, amp=None), load_workers=args.load_workers)
    print(json.dumps({"unit": unit.name, "argv": command, "environment": env, "identity": identity,
                      "execution_authorized": False}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
