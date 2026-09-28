"""CPU-only scope front end for the paused-binding-aware PMM campaign.

Prints preview commands without executing or rewriting the frozen runner. Run
requests remain blocked until the replay and core-assessment bridges exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SCOPE_PATH = ROOT / "docs/plans/pmm_core_scope_v2.json"
EXPECTED = {
    "schema_version": 1, "scope_id": "pmm-core-v2", "campaign_id": "pmm_ion_metal_v2_context",
    "source_tree_sha256": "adc95c42261448dc9d35572a138a3b8349de79124d9708618f511c27a848dd23",
    "families": ["only_esm", "only_gvp", "gvp_late_fusion"],
    "targets": ["four_class", "five_class", "six_class"], "readouts": ["none"], "folds": [0, 1, 2, 3, 4],
    "model_seeds": [42], "epochs": 50, "required_core_fits": 45,
    "queued_folds_by_target": {"four_class": [1, 2, 3, 4], "five_class": [0, 1, 2, 3, 4], "six_class": [1, 2, 3, 4]},
    "paused_readouts": ["first_shell_bias"], "legacy_full_grid_fits": 45,
}
EXECUTION_BLOCKERS = [
    "TECH-023: integrate an explicit prospective replay acceptance contract without changing frozen run identities; fold-0 v2.1 agreement is not a future-fold or legacy-runner pass.",
    "TECH-025: implement scope-aware four/five/six assessment and promotion; the frozen assessor/refit bridge requires the different historical 45-fit grid and ignores readout filters.",
]


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
            and type(fold) is int and fold in EXPECTED["queued_folds_by_target"][target],
            "Select one ordinary-readout core family/target and a queued fold; existing four/six fold 0 is preserved")


def refuse_existing_artifacts(campaign_dir, family, target, fold):
    name = unit_name(family, target, fold)
    for path in (campaign_dir / "runs" / name, campaign_dir / "runs" / f"{name}.log",
                 campaign_dir / "commands" / f"{name}.json"):
        require(not path.exists() and not path.is_symlink(),
                f"Existing unit artifact requires explicit reconciliation, never automatic retry/reuse: {path}")


def candidate_command(args, family, target, fold):
    validate_unit(family, target, fold)
    if target == "five_class":
        # The old runner cannot accept five classes. Future folds intentionally
        # have no command until their scope-aware execution bridge exists.
        if fold != 0:
            return None
        return [args.python_bin, str(ROOT / "run_pmm_five_class_screen.py"),
                "--campaign-dir", str(args.campaign_dir), "--train-dir", str(args.train_dir),
                "--action", "run", "--family", family]
    return [args.python_bin, str(ROOT / "scripts/run_metal_5fold_cv.py"),
            "--campaign-dir", str(args.campaign_dir), "--train-dir", str(args.train_dir),
            "--campaign-action", "run", "--families", family, "--targets", target,
            "--readouts", "none", "--folds", str(fold), "--device", "cuda",
            "--status-tag", "core_" + unit_name(family, target, fold)]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--action", choices=("plan", "run"), default="plan")
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
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    scope, scope_hash = load_scope(args.scope_file)
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
        raise ValueError("Execution remains blocked: " + " ".join(EXECUTION_BLOCKERS))
    require(args.family is None and args.target is None and args.fold is None,
            "The plan always enumerates the fixed 39 candidate selectors; unit selection belongs to --action run")
    require(all(getattr(args, key) is None for key in (*runtime_fields, "load_workers")),
            "The CPU-only plan does not accept allocation/runtime fields")
    units = [{"run_name": unit_name(family, target, fold), "family": family, "target": target,
              "readout": "none", "fold": fold, "seed": 42,
              "preview_argv_without_allocation_fields": candidate_command(args, family, target, fold)}
             for family in scope["families"] for target in scope["targets"] for fold in scope["queued_folds_by_target"][target]]
    result = {"scope_id": scope["scope_id"], "scope_sha256": scope_hash,
              "source_tree_sha256": scope["source_tree_sha256"], "execution_ready": False,
              "status": "preview_only_pending_replay_and_core_assessment_integration",
              "required_core_fits": 45, "candidate_units": len(units),
              "five_class_screen_entrypoint": "run_pmm_five_class_screen.py",
              "preserved_core_fold0_units": 6, "historical_awareness_fold0_units": 3,
              "paused_awareness_remaining_units": 12, "legacy_full_grid_fits": 45,
              "blocked_prerequisites": EXECUTION_BLOCKERS,
              "run_artifacts_examined": False, "child_invoked": False, "files_written": False,
              "note": "Counts describe the amended scope, not verified completion. Commands are non-executable previews until prerequisites and actual allocation fields are satisfied. Run-state, budget, persistence and provider admission are not certified by this plan.",
              "units": units}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"Core scope refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
