"""Scoped PMM core continuation over the unchanged frozen scientific source.

Plan is read-only. Run admits one absent fold-1..4 fit, preserves the original
strict replay outcome, and applies the separately versioned core validator.
No retry, provider lifecycle, final training or held-out execution is exposed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
SCOPE_PATH = ROOT / "docs/plans/pmm_core_scope_v2.json"
EXPECTED = {
    "schema_version": 1, "scope_id": "pmm-core-v2", "campaign_id": "pmm_ion_metal_v2_context",
    "source_tree_sha256": "adc95c42261448dc9d35572a138a3b8349de79124d9708618f511c27a848dd23",
    "families": ["only_esm", "only_gvp", "gvp_late_fusion"],
    "targets": ["four_class", "five_class", "six_class"], "readouts": ["none"], "folds": [0, 1, 2, 3, 4],
    "model_seeds": [42], "epochs": 50, "required_core_fits": 45,
    "queued_folds_by_target": {"four_class": [1, 2, 3, 4], "five_class": [1, 2, 3, 4], "six_class": [1, 2, 3, 4]},
    "paused_readouts": ["first_shell_bias"], "legacy_full_grid_fits": 45,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_tree_sha256(root=ROOT):
    digest = hashlib.sha256()
    for path in sorted([*(root / "src").rglob("*.py"), *(root / "scripts").glob("*.py")]):
        if "__pycache__" not in path.parts:
            digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def load_scope(path):
    path = Path(path)
    content = path.read_bytes()
    scope = json.loads(content)
    for key, expected in EXPECTED.items():
        require(key in scope and json.dumps(scope[key], sort_keys=True) == json.dumps(expected, sort_keys=True),
                f"Unsupported core scope field {key!r}: expected {expected!r}")
    return scope, hashlib.sha256(content).hexdigest()


def unit_name(family, target, fold):
    return f"{family}__{target}__none__fold{fold}__seed42"


def validate_unit(family, target, fold):
    require(family in EXPECTED["families"] and target in EXPECTED["targets"]
            and type(fold) is int and fold in (1, 2, 3, 4),
            "Select one ordinary-readout core family/target and fold 1..4; all existing fold 0 fits are preserved")


def refuse_existing_artifacts(campaign_dir, family, target, fold):
    name = unit_name(family, target, fold)
    for path in (campaign_dir / "runs" / name, campaign_dir / "runs" / f"{name}.log",
                 campaign_dir / "commands" / f"{name}.json",
                 campaign_dir / f"run_status_core_{name}.json"):
        require(not path.exists() and not path.is_symlink(),
                f"Existing unit artifact requires explicit reconciliation, never automatic retry/reuse: {path}")
    for directory in (campaign_dir / "runs" / "_incomplete_attempts", campaign_dir / "commands"):
        require(not any(directory.glob(name + "__*")), "Archived unit attempt exists; automatic retry forbidden")


def candidate_command(args, family, target, fold):
    validate_unit(family, target, fold)
    return [args.python_bin, str(ROOT / "run_pmm_core_campaign.py"),
            "--campaign-dir", str(args.campaign_dir), "--train-dir", str(args.train_dir),
            "--action", "run", "--family", family, "--target", target, "--fold", str(fold)]


def build_command(paths, args):
    """Use the fixed family recipe; five classes share common-four loss weights."""
    from benchmarking import pmm_ion_campaign as campaign
    from training.config import config_to_payload, parse_args as parse_training
    from pmm_core_replay import POLICY_ID, POLICY_SHA256

    validate_unit(args.family, args.target, args.fold)
    target = "four_class" if args.target == "five_class" else args.target
    command, env, identity = campaign.build_train_command(
        paths, python_bin=args.python_bin, train_dir=args.train_dir,
        config=campaign.GridConfig(args.family, target, "none"), fold=args.fold,
        seed=42, device="cuda" if args.action == "run" else "cpu", runs_dir=paths.runs,
        epochs=50, load_workers=args.load_workers, save_epoch_checkpoints=True)
    index = command.index("--campaign-run-identity")
    del command[index:index + 2]
    if args.target == "five_class":
        command[command.index("--metal-label-scheme") + 1] = "five_class"
        command += ["--fe-loss-multiplier", command[command.index("--class-viii-loss-multiplier") + 1]]
    command[command.index("--run-name") + 1] = unit_name(args.family, args.target, args.fold)
    resolved = config_to_payload(parse_training(command[3:]))
    identity.update(target_scheme=args.target, resolved_config_sha256=campaign.stable_hash(
        {k: v for k, v in resolved.items() if k not in campaign.NON_IDENTITY_CONFIG_KEYS}),
        core_replay_contract={"policy_id": POLICY_ID, "policy_sha256": POLICY_SHA256})
    command += ["--campaign-run-identity", json.dumps(identity, sort_keys=True)]
    return command, env, identity


def readiness_hashes(campaign_dir):
    """Bind operational code and historical integration independently of training."""
    from pmm_core_replay import DIAGNOSTIC_INDEX
    paths = [ROOT / name for name in ("run_pmm_core_campaign.py", "pmm_core_replay.py",
                                      "pmm_core_assessment.py")]
    paths += [campaign_dir / DIAGNOSTIC_INDEX]
    return {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else "campaign/" + str(p.relative_to(campaign_dir)):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def certify_readiness(args, scope_hash):
    from pmm_core_replay import validate_core_unit, POLICY_ID, POLICY_SHA256, sha

    records, evidence = [], {}
    for family in EXPECTED["families"]:
        for target in EXPECTED["targets"]:
            checked = validate_core_unit(args.campaign_dir, args.train_dir, family, target, 0, source_root=ROOT)
            records.append({key: value for key, value in checked.items() if key not in {"rows", "receipt"}})
            run = args.campaign_dir / "runs" / unit_name(family, target, 0)
            for name in ("best_model_checkpoint.pt", "run_config.json", "run_metadata.json",
                         "selected_checkpoint.json", "val_predictions.csv",
                         "independent_validation_replay/val_predictions.csv"):
                evidence[str((run / name).relative_to(args.campaign_dir))] = sha(run / name)
    from pmm_core_assessment import assess_core, preview_core_refit, verify_core_refit
    require(all(callable(f) for f in (assess_core, preview_core_refit, verify_core_refit)), "Core bridge is incomplete")
    result = {"schema": "pmm-core-readiness-v1", "campaign_id": EXPECTED["campaign_id"],
              "scope_sha256": scope_hash, "source_tree_sha256": EXPECTED["source_tree_sha256"],
              "policy_id": POLICY_ID, "policy_sha256": POLICY_SHA256,
              "implementation_sha256": readiness_hashes(args.campaign_dir),
              "historical_units": records, "historical_evidence_sha256": evidence,
              "new_unit_folds": [1, 2, 3, 4], "new_unit_count": 36,
              "compute_authorized": False, "held_out_access": False,
              "note": "Scientific continuation readiness only; provider/session budget authority remains separate."}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "core_readiness.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def verify_readiness(args, scope_hash):
    from pmm_core_replay import POLICY_ID, POLICY_SHA256, sha, validate_core_unit

    require(args.readiness_file is not None, "Full-grid run needs the verified core readiness file")
    record = json.loads(args.readiness_file.read_text())
    require(record.get("schema") == "pmm-core-readiness-v1"
            and record.get("campaign_id") == EXPECTED["campaign_id"]
            and record.get("scope_sha256") == scope_hash
            and record.get("source_tree_sha256") == EXPECTED["source_tree_sha256"]
            and record.get("policy_id") == POLICY_ID and record.get("policy_sha256") == POLICY_SHA256
            and record.get("implementation_sha256") == readiness_hashes(args.campaign_dir),
            "Core readiness belongs to different source, scope, implementation or diagnostic integration")
    expected_names = {unit_name(f, t, 0) for f in EXPECTED["families"] for t in EXPECTED["targets"]}
    require({r["run_name"] for r in record.get("historical_units", [])} == expected_names
            and len(record["historical_units"]) == 9, "Core readiness lacks nine distinct qualified historical units")
    expected_evidence = {"runs/" + name + "/" + artifact for name in expected_names for artifact in (
        "best_model_checkpoint.pt", "run_config.json", "run_metadata.json", "selected_checkpoint.json",
        "val_predictions.csv", "independent_validation_replay/val_predictions.csv")}
    require(set(record.get("historical_evidence_sha256", {})) == expected_evidence,
            "Readiness historical evidence coverage is incomplete")
    for relative, digest in record["historical_evidence_sha256"].items():
        path = (args.campaign_dir / relative).resolve()
        require(path.is_relative_to(args.campaign_dir) and sha(path) == digest,
                f"Readiness historical evidence changed: {relative}")
    # Recheck complete provenance, every preserved replay and diagnostic, not
    # merely the compact readiness file's selected artifact hashes.
    for family in EXPECTED["families"]:
        for target in EXPECTED["targets"]:
            validate_core_unit(args.campaign_dir, args.train_dir, family, target, 0, source_root=ROOT)


def execute_one(paths, args, command, env, identity):
    from benchmarking import pmm_ion_campaign as campaign
    from benchmarking.pmm_execution import CampaignExecution, ExecutionPolicy
    from benchmarking.pmm_ion_features import verify_frozen_feature_inventory
    from training.access_guard import install_forbidden_read_guard
    from pmm_core_replay import validate_core_unit

    require(0 < args.execution_max_seconds <= 4 * 3600, "Execution exceeds existing four-hour ceiling")
    execution = CampaignExecution(paths.root, policy=ExecutionPolicy(
        deadline_unix=args.execution_deadline, max_total_seconds=args.execution_max_seconds,
        allocation_started_unix=args.allocation_started), session_id=args.session_id,
        durable_root=args.durable_root, persistence_mode=args.persistence_mode)
    with execution:
        refuse_existing_artifacts(paths.root, args.family, args.target, args.fold)
        verify_readiness(args, load_scope(args.scope_file)[1])
        install_forbidden_read_guard(campaign.forbidden_read_roots(args.train_dir))
        campaign.campaign_manifest_guard(paths, require_context=True)
        verify_frozen_feature_inventory(paths, args.train_dir)
        require(source_tree_sha256() == EXPECTED["source_tree_sha256"], "Frozen scientific source changed")
        require(build_command(paths, args)[2] == identity, "Inputs or replay contract changed before admission")
        name = unit_name(args.family, args.target, args.fold)
        execution.admit(name, args.estimated_fit_seconds)
        started = time.time()
        result = campaign.execute_unit(paths, command, env, identity, paths.runs, name, execution=execution)
        legacy_status = result["status"]
        # One original replay attempt only. Its exporter may fail its tighter
        # probability threshold; the independent core validator never retries it.
        if legacy_status in {"completed", "failed_independent_replay"}:
            try:
                verified = validate_core_unit(paths.root, args.train_dir, args.family, args.target,
                                             args.fold, source_root=ROOT)
                require(verified["identity"] == identity, "Validated identity differs from admitted unit")
                result.update(status="completed", core_verification={
                    key: value for key, value in verified.items() if key not in {"rows", "receipt"}})
            except (ValueError, KeyError, OSError, RuntimeError) as exc:
                result.update(status="failed_core_verification", validation_error=str(exc))
        result.pop("receipt", None)
        result.update(original_strict_execution_status=legacy_status,
                      elapsed_seconds=time.time() - started, source_root=str(ROOT))
        status = paths.root / f"run_status_core_{name}.json"
        campaign.write_json(status, result)
        execution.record_result(name, result["status"], result["elapsed_seconds"])
        # A child can fail before creating its run directory. Preserve every
        # artifact it did produce, including the terminal core status receipt.
        execution.persist([path for path in (paths.runs / name, paths.runs / f"{name}.log",
                                             paths.commands / f"{name}.json", status) if path.exists()])
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "completed" else 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--action", choices=("plan", "certify", "run", "assess", "refit-preview"), default="plan")
    parser.add_argument("--scope-file", type=Path, default=SCOPE_PATH)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--family", choices=EXPECTED["families"])
    parser.add_argument("--target", choices=EXPECTED["targets"])
    parser.add_argument("--fold", type=int, choices=EXPECTED["folds"])
    # Match the frozen runner's allocation/persistence inputs; no alternate
    # provisioning, spending authorization, or scientific overrides are added.
    parser.add_argument("--session-id")
    for name in ("execution-deadline", "allocation-started", "execution-max-seconds", "estimated-fit-seconds"):
        parser.add_argument("--" + name, type=float)
    parser.add_argument("--durable-root", type=Path)
    parser.add_argument("--persistence-mode", choices=("mounted", "host_pull"))
    parser.add_argument("--load-workers", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--decision-path", type=Path)
    parser.add_argument("--test-route")
    parser.add_argument("--readiness-file", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    scope, scope_hash = load_scope(args.scope_file)
    deferred = scope.get("execution_status") == "deferred_by_user"
    if args.action == "run":
        require(not deferred, "Core continuation is deferred by user; retain fold 0 until an explicit resume and renewed readiness/budget verification")
    require(source_tree_sha256() == scope["source_tree_sha256"], "Frozen scientific source changed; no plan or run allowed")
    args.campaign_dir, args.train_dir = args.campaign_dir.resolve(), args.train_dir.resolve()
    require(args.train_dir.name == "train", "Only the training-side directory named train is allowed")
    manifest = json.loads((args.campaign_dir / "campaign_manifest.json").read_text())
    require(manifest.get("campaign_id") == scope["campaign_id"], "Campaign identity differs from the core scope")
    require((ROOT / "scripts/run_metal_5fold_cv.py").is_file(), "Frozen runner is absent")
    runtime_fields = ("session_id", "execution_deadline", "allocation_started", "execution_max_seconds",
                      "estimated_fit_seconds", "durable_root", "persistence_mode")
    if args.action == "run":
        validate_unit(args.family, args.target, args.fold)
        refuse_existing_artifacts(args.campaign_dir, args.family, args.target, args.fold)
        missing = ["--" + key.replace("_", "-") for key in runtime_fields if getattr(args, key) is None]
        require(not missing, "Run requests require existing allocation/persistence fields: " + ", ".join(missing))
        require(args.load_workers is None or args.load_workers > 0, "load-workers must be positive")
        require(args.output_dir is None and args.decision_path is None and args.test_route is None,
                "Training cannot consume assessment/refit arguments")
        require(args.readiness_file is not None, "Full-grid run needs the verified core readiness file")
        sys.path.insert(0, str(ROOT / "src"))
        from benchmarking import pmm_ion_campaign as campaign
        paths = campaign.CampaignPaths(args.campaign_dir)
        command, env, identity = build_command(paths, args)
        return execute_one(paths, args, command, env, identity)
    if args.action in {"certify", "assess", "refit-preview"}:
        require(args.family is None and args.target is None and args.fold is None,
                "Core assessment always checks all 45 units")
        require(all(getattr(args, key) is None for key in (*runtime_fields, "load_workers")),
                "CPU assessment/preview accepts no allocation fields")
        require(args.output_dir is not None and not args.output_dir.exists(), "Use a new assessment output directory")
        require(args.readiness_file is None, "CPU actions create or independently verify readiness; no supplied readiness override")
        if args.action == "certify":
            require(args.decision_path is None and args.test_route is None, "Readiness takes no final-reporting inputs")
            result = certify_readiness(args, scope_hash)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        from pmm_core_assessment import assess_core, preview_core_refit
        if args.action == "assess":
            require(args.decision_path is None and args.test_route is None, "Assessment takes no final-reporting inputs")
            result = assess_core(args.campaign_dir, args.train_dir, source_root=ROOT, out_dir=args.output_dir,
                                 scope_path=args.scope_file)
        else:
            require(args.decision_path is not None and args.test_route, "Refit preview needs a complete core decision and declared test route")
            result = preview_core_refit(args.campaign_dir, args.train_dir, source_root=ROOT,
                                       decision_path=args.decision_path, out_dir=args.output_dir,
                                       route=args.test_route, python_bin=args.python_bin, device="cuda",
                                       scope_path=args.scope_file)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    require(args.family is None and args.target is None and args.fold is None,
            "The plan always enumerates the fixed 36 continuation selectors; unit selection belongs to --action run")
    require(all(getattr(args, key) is None for key in (*runtime_fields, "load_workers")),
            "The CPU-only plan does not accept allocation/runtime fields")
    require(args.output_dir is None and args.decision_path is None and args.test_route is None and args.readiness_file is None,
            "The read-only plan accepts no assessment arguments")
    units = [{"run_name": unit_name(family, target, fold), "family": family, "target": target,
              "readout": "none", "fold": fold, "seed": 42,
              "preview_argv_without_allocation_fields": candidate_command(args, family, target, fold)}
             for family in scope["families"] for target in scope["targets"] for fold in (1, 2, 3, 4)]
    result = {"scope_id": scope["scope_id"], "scope_sha256": scope_hash,
              "source_tree_sha256": scope["source_tree_sha256"], "execution_ready": False,
              "status": "deferred_by_user" if deferred else "preview_only_pending_operational_admission",
              "required_core_fits": 45, "candidate_units": len(units),
              "five_class_screen_entrypoint": "run_pmm_five_class_screen.py",
              "preserved_core_fold0_units": 9, "historical_awareness_fold0_units": 3,
              "paused_awareness_remaining_units": 12, "legacy_full_grid_fits": 45,
              "blocked_prerequisites": (["Explicit user resume of deferred fold-1–4 fits"] if deferred else [])
                                       + ["Approved remaining compute budget, source/feature/replay validation, session admission and independent persistence"],
              "run_artifacts_examined": False, "child_invoked": False, "files_written": False,
              "note": "Counts describe the continuation scope, not verified completion. All nine fold-0 fits are preserved with their original qualifications. Run-state, budget, persistence and provider admission are not certified by this plan. No final refit or held-out execution is exposed.",
              "units": units}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"Core scope refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
