"""Manual, bounded single-GPU campaign. No command implicitly allocates a GPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from serial_metal_campaign import budget, evidence, profile, runtime, workflow


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="action", required=True)
    for name in ("plan", "export-runtime-plan", "preview", "prepare", "session-open", "session-close", "readiness", "forecast", "advance",
                 "admit", "execute", "report", "reuse", "verify-transfer", "export-state", "verify-state-transfer",
                 "reconcile", "authorize-budget", "ring-audit", "_ring-audit"):
        command = sub.add_parser(name)
        command.add_argument("--output-dir", type=Path, required=True)
        if name == "plan":
            command.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
            command.add_argument("--data-root", type=Path, required=True)
            command.add_argument("--external-features-root-dir", type=Path, required=True)
            command.add_argument("--feature-overlay-manifest", type=Path, required=True)
        if name == "export-runtime-plan":
            command.add_argument("--destination-dir", type=Path, required=True)
            command.add_argument("--worker-repo-root", required=True)
            command.add_argument("--worker-data-root", required=True)
            command.add_argument("--worker-output-dir", required=True)
            command.add_argument("--worker-external-features-root", required=True)
            command.add_argument("--worker-python", required=True)
        if name in ("session-open", "session-close"):
            command.add_argument("--session-id", required=True)
        if name == "session-open":
            command.add_argument("--allocation-started-epoch", type=float, required=True)
            command.add_argument("--host-receipt-json", type=Path, required=True)
        if name == "session-close":
            command.add_argument("--stopped-epoch", type=float, required=True)
            command.add_argument("--stop-evidence-json", type=Path, required=True)
        if name == "readiness":
            command.add_argument("--operations-plan-json", type=Path)
        if name in ("execute", "ring-audit"):
            command.add_argument("--host-receipt-json", type=Path, required=True)
        if name == "authorize-budget":
            command.add_argument("--authorization-json", type=Path, required=True)
        if name == "reuse":
            command.add_argument("--run-id", required=True)
            command.add_argument("--source-campaign", type=Path, required=True)
            command.add_argument("--source-run-id", required=True)
        if name in ("verify-transfer", "verify-state-transfer"):
            command.add_argument("--receipt-json", type=Path, required=True)
        if name == "verify-transfer":
            command.add_argument("--attempt-id", required=True)
        if name == "ring-audit":
            command.add_argument("--fold-index", type=int, choices=range(5), required=True)
        if name == "_ring-audit":
            command.add_argument("--source-run-id", required=True)
            command.add_argument("--run-name", required=True)
    return root


def dispatch(args):
    output = args.output_dir.resolve()
    if args.action == "preview":
        return budget.write_preview(output, budget.historical_preview())
    if args.action == "plan":
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=args.repo_root, text=True,
                                capture_output=True, check=False).stdout.strip() or "unknown"
        result = profile.plan(args.repo_root, args.data_root, output, args.external_features_root_dir,
                              args.feature_overlay_manifest, source_commit=commit)
        budget.write_preview(output, budget.historical_preview())
        return dict(profile=result["profile"], initial_runs=len(result["runs"]), output_dir=str(output),
                    status="planned_not_ready_for_training")
    if args.action == "prepare":
        return evidence.prepare(output)
    if args.action == "export-runtime-plan":
        from serial_metal_campaign.portability import export_runtime_plan
        result = export_runtime_plan(output, args.destination_dir, args.worker_repo_root,
                                     args.worker_data_root, args.worker_output_dir,
                                     args.worker_external_features_root, args.worker_python)
        return dict(profile=result["profile"], destination=str(args.destination_dir),
                    worker_output_dir=result["output_dir"], status="exported_fresh_worker_preparation_required")
    if args.action == "session-open":
        guard = runtime.read_json(args.host_receipt_json)
        if not guard or guard.get("session_id") != args.session_id or guard.get("allocation_started_epoch") != args.allocation_started_epoch:
            raise ValueError("Session registration must match the ownership-verified host allocation receipt")
        profile.verify_manifest(output)
        workflow.verify_host_guard(output, args.host_receipt_json,
                                   session=dict(session_id=args.session_id, started_epoch=args.allocation_started_epoch))
        result = runtime.open_session(output, args.session_id, args.allocation_started_epoch, evidence.hardware_probe())
        return result
    if args.action == "session-close":
        return runtime.close_session(output, args.session_id, args.stopped_epoch,
                                     stop_evidence=runtime.read_json(args.stop_evidence_json))
    if args.action == "readiness":
        return workflow.readiness(output, runtime.read_json(args.operations_plan_json) if args.operations_plan_json else None)
    if args.action == "forecast":
        return workflow.measured_forecast(output)
    if args.action == "advance":
        return workflow.advance(output)
    if args.action == "admit":
        return workflow.admit(output)
    if args.action == "execute":
        return workflow.execute(output, args.host_receipt_json)
    if args.action == "report":
        return workflow.report(output)
    if args.action == "reuse":
        queue = runtime.read_json(output / "queue.json")
        run = next((r for r in queue["runs"] if r["id"] == args.run_id), None)
        if run is None or run["stage"] != "discovery":
            raise ValueError("Reuse requires a planned discovery cell")
        reserved = {ident for block in runtime.read_json(output / "comparisons.json", []) for ident in block["run_ids"]}
        if args.run_id in reserved or any(a["run_id"] == args.run_id for a in runtime.read_json(output / "attempts.json", [])):
            raise ValueError("Import compatible evidence before reserving or attempting this cell")
        return evidence.import_reuse(output, run, args.source_campaign, args.source_run_id)
    if args.action == "verify-transfer":
        return runtime.record_persistence(output, args.attempt_id, runtime.read_json(args.receipt_json))
    if args.action == "export-state":
        return workflow.export_state(output)
    if args.action == "verify-state-transfer":
        return workflow.verify_state_transfer(output, runtime.read_json(args.receipt_json))
    if args.action == "reconcile":
        return runtime.reconcile_interrupted(output)
    if args.action == "authorize-budget":
        profile.verify_manifest(output)
        return runtime.authorize_budget(output, runtime.read_json(args.authorization_json))
    if args.action == "ring-audit":
        return workflow.ring_audit(output, args.fold_index, args.host_receipt_json)
    if args.action == "_ring-audit":
        if Path(args.run_name).name != args.run_name:
            raise ValueError("Audit run name must be one path component")
        queue = runtime.read_json(output / "queue.json")
        source = next(r for r in queue["runs"] if r["id"] == args.source_run_id)
        return evidence.ring_input_audit(output, source, directory=output / "ring_audits" / args.run_name)
    raise AssertionError(args.action)


def main(argv=None):
    args = parser().parse_args(argv)
    # The bounded audit child is already owned by the outer controller lock.
    if args.action == "_ring-audit":
        result = dispatch(args)
    else:
        with workflow.controller_lock(args.output_dir):
            result = dispatch(args)
    print(json.dumps(result, indent=2, default=str, allow_nan=False))


if __name__ == "__main__":
    main()
