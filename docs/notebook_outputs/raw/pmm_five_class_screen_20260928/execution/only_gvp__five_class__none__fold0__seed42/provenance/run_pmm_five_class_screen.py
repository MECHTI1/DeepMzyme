"""Bounded five-class fold-0 screen using the unchanged PMM trainer and replay.

Plan is read-only CPU command expansion. Run admits exactly one new fit through
the existing allocation/persistence gates; it never retries an existing artifact.
This screen does not implement full-grid assessment, promotion or held-out use.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))
PROTOCOL_PATH = ROOT / "docs/plans/pmm_five_class_screen_v1.json"
EXPECTED = {
    "schema_version": 1, "protocol_id": "pmm-five-class-screen-v1",
    "campaign_id": "pmm_ion_metal_v2_context",
    "source_tree_sha256": "adc95c42261448dc9d35572a138a3b8349de79124d9708618f511c27a848dd23",
    "families": ["only_esm", "only_gvp", "gvp_late_fusion"], "target": "five_class",
    "readout": "none", "fold": 0, "seed": 42, "epochs": 50, "batch_size": 16,
    "selection_metric": "val_metal_balanced_acc", "independent_replay_atol": "0.000001",
    "weight_rule": "common_four_equalized_per_ion", "held_out_access": False,
    "allow_existing_artifact_retry": False,
}
NATIVE_LABELS = ("Mn", "Cu", "Zn", "Fe", "Class VIII")  # native Class VIII = Co+Ni
NATIVE_TARGETS = {"MN": 0, "CU": 1, "ZN": 2, "FE": 3, "CO": 4, "NI": 4}
RUNTIME_FIELDS = ("session_id", "execution_deadline", "allocation_started", "execution_max_seconds",
                  "estimated_fit_seconds", "durable_root", "persistence_mode")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_tree_sha256():
    digest = hashlib.sha256()
    for path in sorted([*(ROOT / "src").rglob("*.py"), *(ROOT / "scripts").glob("*.py")]):
        if "__pycache__" not in path.parts:
            digest.update(str(path.relative_to(ROOT)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def load_protocol(path):
    raw = Path(path).read_bytes()
    protocol = json.loads(raw)
    for key, expected in EXPECTED.items():
        require(json.dumps(protocol.get(key), sort_keys=True) == json.dumps(expected, sort_keys=True),
                f"Unsupported screen protocol field {key!r}")
    return protocol, hashlib.sha256(raw).hexdigest()


def unit_name(family):
    require(family in EXPECTED["families"], "Select exactly one planned family")
    return f"{family}__five_class__none__fold0__seed42"


def artifact_paths(paths, family):
    name = unit_name(family)
    return [paths.runs / name, paths.runs / f"{name}.log", paths.commands / f"{name}.json",
            paths.root / f"run_status_five_screen_{name}.json"]


def refuse_existing_artifacts(paths, family):
    for path in artifact_paths(paths, family):
        require(not path.exists() and not path.is_symlink(),
                f"Existing unit artifact requires reconciliation; automatic retry/reuse forbidden: {path}")
    for directory in (paths.runs / "_incomplete_attempts", paths.commands):
        require(not any(directory.glob(unit_name(family) + "__*")),
                "An archived unit attempt exists; automatic retry forbidden")


def build_command(paths, args, family, protocol_sha):
    from benchmarking import pmm_ion_campaign as campaign
    from training.config import config_to_payload, parse_args

    name = unit_name(family)
    command, env, identity = campaign.build_train_command(
        paths, python_bin=args.python_bin, train_dir=args.train_dir,
        config=campaign.GridConfig(family, "four_class", "none"), fold=0, seed=42,
        device="cuda" if args.action == "run" else "cpu", runs_dir=paths.runs,
        epochs=50, load_workers=args.load_workers, save_epoch_checkpoints=True,
    )
    index = command.index("--campaign-run-identity")
    del command[index:index + 2]
    command[command.index("--metal-label-scheme") + 1] = "five_class"
    command[command.index("--run-name") + 1] = name
    command += ["--fe-loss-multiplier", command[command.index("--class-viii-loss-multiplier") + 1]]
    resolved = config_to_payload(parse_args(command[3:]))
    identity.update(target_scheme="five_class", resolved_config_sha256=campaign.stable_hash(
        {key: value for key, value in resolved.items() if key not in campaign.NON_IDENTITY_CONFIG_KEYS}),
        five_class_screen={"protocol_id": EXPECTED["protocol_id"], "protocol_sha256": protocol_sha,
                           "adapter_sha256": file_sha(__file__)})
    command += ["--campaign-run-identity", json.dumps(identity, sort_keys=True)]
    return command, env, identity


def validate_five_rows(rows, membership, *, checkpoint_sha256, selected_epoch):
    """Validate five-native labels plus probability-first common-four collapse."""
    from benchmarking.pmm_comparator import PROBABILITY_TOLERANCE, validate_prediction_rows

    validate_prediction_rows(rows, membership, 0, seed=42, checkpoint_sha256=checkpoint_sha256)
    columns = {"p_native_" + label.replace(" ", "_") for label in NATIVE_LABELS}
    for row in rows:
        uid = row["source_uid"]
        require({key for key in row if key.startswith("p_native_")} == columns,
                f"{uid}: native probability vocabulary must contain exactly five classes")
        require(row["metal_label_scheme"] == "five_class" and row["binding_residue_pooling"] == "none"
                and int(row["selected_epoch"]) == selected_epoch, f"{uid}: screen configuration differs")
        require(int(row["y_native"]) == NATIVE_TARGETS[row["native_element"].upper()],
                f"{uid}: incorrect native-five target (Class VIII means Co+Ni)")
        pn = [float(row["p_native_" + label.replace(" ", "_")]) for label in NATIVE_LABELS]
        require(all(math.isfinite(p) and 0 <= p <= 1 for p in pn)
                and abs(sum(pn) - 1) <= PROBABILITY_TOLERANCE, f"{uid}: invalid native probabilities")
        pred = int(row["pred_native"])
        require(pred in range(5) and max(pn) - pn[pred] <= PROBABILITY_TOLERANCE,
                f"{uid}: native prediction disagrees with probabilities")
        p4 = [float(row["p_common4_" + label.replace(" ", "_")])
              for label in ("Mn", "Cu", "Zn", "Class VIII")]
        require(all(abs(a - b) <= PROBABILITY_TOLERANCE for a, b in zip(p4, pn[:3] + [sum(pn[3:])])),
                f"{uid}: common-four probabilities must sum Fe and Co+Ni before argmax")


def validate_completed_unit(paths, family, identity):
    from benchmarking import pmm_ion_campaign as campaign
    from training.campaign_runtime import read_fold_membership

    run = paths.runs / unit_name(family)
    receipt = campaign.completed_run_receipt(run, identity)
    require(receipt is not None, "Frozen strict independent-replay completion gate failed")
    require(receipt["selection_metric"] == "val_metal_balanced_acc"
            and receipt["tie_rule"] == "earliest epoch (strictly greater metric required to replace)",
            "Checkpoint must be selected by native balanced accuracy with earliest tie")
    membership = read_fold_membership(paths.fold_membership, identity["fold_membership_sha256"])
    hashes = {}
    for directory, record in ((run, receipt), (run / "independent_validation_replay",
                            campaign.read_json(run / "independent_validation_replay/replay_receipt.json"))):
        prediction = directory / record["validation_predictions"]["path"]
        require(prediction.resolve().parent == directory.resolve(), "Prediction artifact escaped its run directory")
        with prediction.open(newline="", encoding="utf-8") as handle:
            validate_five_rows(list(csv.DictReader(handle)), membership,
                               checkpoint_sha256=receipt["selected_checkpoint_sha256"],
                               selected_epoch=int(receipt["selected_epoch"]))
        hashes[str(prediction.relative_to(paths.root))] = file_sha(prediction)
    return {"status": "passed", "native_labels": list(NATIVE_LABELS),
            "native_class_viii": "Co+Ni", "common_four_class_viii": "Fe+Co+Ni",
            "prediction_sha256": hashes, "campaign_run_identity": identity}


def execute_one(paths, args, command, env, identity):
    from benchmarking import pmm_ion_campaign as campaign
    from benchmarking.pmm_execution import CampaignExecution, ExecutionPolicy
    from benchmarking.pmm_ion_features import verify_frozen_feature_inventory
    from training.access_guard import install_forbidden_read_guard

    require(args.execution_max_seconds <= 4 * 3600, "Execution exceeds existing four-hour ceiling")
    execution = CampaignExecution(paths.root, policy=ExecutionPolicy(
        deadline_unix=args.execution_deadline, max_total_seconds=args.execution_max_seconds,
        allocation_started_unix=args.allocation_started), session_id=args.session_id,
        durable_root=args.durable_root, persistence_mode=args.persistence_mode)
    with execution:
        refuse_existing_artifacts(paths, args.family)
        require(source_tree_sha256() == EXPECTED["source_tree_sha256"], "Frozen scientific source changed")
        require(load_protocol(args.protocol_file)[1] == identity["five_class_screen"]["protocol_sha256"]
                and file_sha(__file__) == identity["five_class_screen"]["adapter_sha256"], "Adapter/protocol changed")
        install_forbidden_read_guard(campaign.forbidden_read_roots(args.train_dir))
        campaign.campaign_manifest_guard(paths, require_context=True)
        verify_frozen_feature_inventory(paths, args.train_dir)
        # Bind guards' current inputs to the exact command already reviewed.
        require(build_command(paths, args, args.family, identity["five_class_screen"]["protocol_sha256"])[2]
                == identity, "Campaign inputs changed after command construction")
        name = unit_name(args.family)
        execution.admit(name, args.estimated_fit_seconds)
        result = campaign.execute_unit(paths, command, env, identity, paths.runs, name, execution=execution)
        if result["status"] == "completed":
            try:
                result["five_class_validation"] = validate_completed_unit(paths, args.family, identity)
            except (ValueError, KeyError, OSError) as exc:
                result.update(status="failed_five_class_validation", validation_error=str(exc))
        result.pop("receipt", None)
        status = artifact_paths(paths, args.family)[-1]
        campaign.write_json(status, {"protocol_id": EXPECTED["protocol_id"], "units": [result],
                                     "full_grid_assessed": False, "promotion_ready": False})
        execution.record_result(name, result["status"], result.get("elapsed_seconds", 0.0))
        persistence = execution.persist([path for path in artifact_paths(paths, args.family) if path.exists()])
    print(json.dumps({"unit": result, "persistence": persistence}, indent=2, sort_keys=True))
    return 0 if result["status"] == "completed" else 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--action", choices=("plan", "run"), default="plan")
    parser.add_argument("--protocol-file", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--family", choices=EXPECTED["families"])
    parser.add_argument("--session-id")
    for name in ("execution-deadline", "allocation-started", "execution-max-seconds", "estimated-fit-seconds"):
        parser.add_argument("--" + name, type=float)
    parser.add_argument("--durable-root", type=Path)
    parser.add_argument("--persistence-mode", choices=("mounted", "host_pull"))
    parser.add_argument("--load-workers", type=int)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    protocol, protocol_sha = load_protocol(args.protocol_file)
    require(source_tree_sha256() == protocol["source_tree_sha256"], "Frozen scientific source changed")
    args.campaign_dir, args.train_dir = args.campaign_dir.resolve(), args.train_dir.resolve()
    require(args.train_dir.name == "train", "Only the training-side directory named train is allowed")
    from benchmarking import pmm_ion_campaign as campaign

    paths = campaign.CampaignPaths(args.campaign_dir)
    require(campaign.load_campaign(paths)["campaign_id"] == protocol["campaign_id"], "Wrong campaign identity")
    if args.action == "run":
        unit_name(args.family)
        refuse_existing_artifacts(paths, args.family)
        missing = [key for key in RUNTIME_FIELDS if getattr(args, key) is None]
        require(not missing, "Missing existing allocation/persistence fields: " + ", ".join(missing))
        require(args.load_workers is None or args.load_workers > 0, "load-workers must be positive")
        command, env, identity = build_command(paths, args, args.family, protocol_sha)
        return execute_one(paths, args, command, env, identity)
    require(args.family is None and all(getattr(args, key) is None for key in (*RUNTIME_FIELDS, "load_workers")),
            "CPU plan enumerates all three families and accepts no unit or runtime selection")
    units = []
    for family in protocol["families"]:
        command, env, identity = build_command(paths, args, family, protocol_sha)
        units.append({"run_name": unit_name(family), "preview_training_argv": command,
                      "environment": env, "identity": identity})
    print(json.dumps({"protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_sha,
                      "status": "cpu_preview_only", "candidate_units": 3, "child_invoked": False,
                      "files_written": False, "runtime_admission_checked": False,
                      "full_grid_assessed": False, "promotion_ready": False,
                      "note": "Single fold is exploratory. TECH-023/025 full-grid bridges remain separate. "
                              "Run needs provider authorization, measured admission and independent persistence.",
                      "units": units}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print(f"Five-class screen refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
