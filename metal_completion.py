#!/usr/bin/env python3
"""CPU preparation, evidence collection and assessment; never launch a fit.

Run from the frozen package. New units require an external worker exit-code file,
claim, command, terminal receipt and independent prediction replay. Historical
controls are independently reverified using the immutable v3 collector. Prepared
manifests describe scientific scope and never authorize execution.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import metal_completion_assessment as assessment
import metal_completion_campaign as campaign
import pmm_v3_assessment as old_assessment
import pmm_v3_campaign as v3

ROOT = Path(__file__).resolve().parent
require = campaign.require
sha256 = campaign.sha256_file
CODE_FILES = ("metal_completion.py", "metal_completion_assessment.py", "metal_completion_campaign.py",
              "metal_completion_safety.py", "metal_completion_evaluation.py", "run_metal_completion.py")
ASSESSOR_RULES = {"minimum_gain": 0.002, "recall_drop_limit": 0.03, "tie_band": 0.002,
                  "bootstrap_resamples": 10000, "bootstrap_seed": 42, "confidence": 0.95,
                  "discovery_seed": 42, "confirmation_seeds": [43, 44], "folds": list(range(5)),
                  "nonbaseline_contrasts": 5, "baseline_contrasts": 2}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def bound_json(path: Path, digest: str) -> dict[str, Any]:
    require(sha256(path) == digest, f"Bound JSON changed: {path}")
    return read_json(path)


def write_once(path: Path, payload: dict, *, root: Path) -> None:
    path, root = Path(path), Path(root).resolve()
    require(path.resolve().is_relative_to(root) and path.resolve() != root, "Output must be inside successor root")
    require(not path.exists() and not path.is_symlink(), f"Immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def code_hashes() -> dict[str, str]:
    return {name: sha256(ROOT / name) for name in CODE_FILES}


def _verify_spec(path: Path) -> dict:
    spec = read_json(path)
    require(spec.get("campaign_id") == campaign.CAMPAIGN_ID, "Assessment specification belongs to another campaign")
    require(spec.get("assessor_rules") == ASSESSOR_RULES, "Assessment specification differs from implemented rules")
    return spec


def _check_implementation_rules() -> None:
    require(assessment.MIN_GAIN == ASSESSOR_RULES["minimum_gain"]
            and assessment.RECALL_DROP_LIMIT == ASSESSOR_RULES["recall_drop_limit"]
            and assessment.TIE_BAND == ASSESSOR_RULES["tie_band"]
            and assessment.INTERVAL_SPEC == {key: ASSESSOR_RULES[key]
                                             for key in ("confidence", "bootstrap_resamples", "bootstrap_seed")},
            "Assessor constants changed")


def _guard_training_paths(train_dir: Path) -> None:
    from training.access_guard import install_forbidden_read_guard

    require(Path(train_dir).name == "train", "Only the training-side directory named train is allowed")
    install_forbidden_read_guard(v3.v2.forbidden_read_roots(train_dir))


def prepare(*, parent_dir: Path, successor_root: Path, train_dir: Path, esm_dir: Path,
            assessment_spec_path: Path, phase: str = "discovery", selection_receipt_path: Path | None = None,
            selection_receipt_sha256: str | None = None) -> dict:
    """Freeze a phase manifest without creating an execution authorization."""
    require(phase in ("discovery", "confirmation"), "Unknown phase")
    _guard_training_paths(train_dir)
    parent = v3.V3Paths(Path(parent_dir).resolve())
    paths = campaign.isolated_paths(parent.root, Path(successor_root))
    require(ROOT == campaign.ROOT == v3.ROOT.resolve(), "Execute preparation inside the frozen code package")
    _verify_spec(assessment_spec_path)
    _check_implementation_rules()
    verified = v3.verify_campaign(parent, esm_dir=esm_dir)
    training_inputs = campaign.verify_training_inputs(parent, train_dir)
    v3.require_frozen_spec(parent)
    result = {"schema_version": 1, "campaign_id": campaign.CAMPAIGN_ID, "phase": phase,
              "execution_authorized": False, "held_out_accessed": False,
              "code_root": str(ROOT), "code_sha256": code_hashes(),
              "parent_campaign_dir": str(parent.root), "parent_manifest_sha256": sha256(parent.manifest),
              "parent_extension_sha256": sha256(parent.extension) if parent.extension.exists() else None,
              "source_tree_sha256": verified["frozen_source_tree_sha256"],
              "parent_cohort_sha256": verified["cohort"]["sha256"],
              "parent_fold_membership_sha256": verified["fold_set"]["fold_membership_sha256"],
              "training_input_verification": training_inputs,
              "successor_root": str(paths.root.resolve()), "train_dir": str(Path(train_dir).resolve()),
              "esm_dir": str(Path(esm_dir).resolve()),
              "assessment_spec_path": str(Path(assessment_spec_path).resolve()),
              "assessment_spec_sha256": sha256(assessment_spec_path)}
    if phase == "confirmation":
        require(selection_receipt_path is not None and selection_receipt_sha256 is not None,
                "Confirmation needs a hash-bound discovery selection receipt")
        selection = verify_assessment_receipt(selection_receipt_path, selection_receipt_sha256, esm_dir=esm_dir)
        require(selection["result"]["phase"] == "discovery", "Confirmation requires discovery, not a prior confirmation")
        selected = selection["result"]["shared_recipe"]
        require(selection["parent_manifest_sha256"] == result["parent_manifest_sha256"],
                "Discovery selection belongs to another frozen parent")
        result.update(shared_recipe=selected, selection_receipt_path=str(Path(selection_receipt_path).resolve()),
                      selection_receipt_sha256=selection_receipt_sha256)
        units = campaign.confirmation_units(selected)
    else:
        require(selection_receipt_path is None and selection_receipt_sha256 is None,
                "Discovery cannot import a selection receipt")
        units = campaign.discovery_units()
    result["authorized_grid_units"] = [unit.name for unit in units]
    output = paths.root / f"manifest_{phase}.json"
    write_once(output, result, root=paths.root)
    return {"manifest_path": str(output), "manifest_sha256": sha256(output), "manifest": result}


def _manifest(path: Path, digest: str, esm_dir: Path) -> dict:
    preliminary = bound_json(path, digest)
    _guard_training_paths(Path(preliminary["train_dir"]))
    manifest = campaign.verify_manifest(path, digest, esm_dir=esm_dir)
    require(manifest["code_sha256"] == code_hashes(), "Host/assessor code binding differs")
    _verify_spec(Path(manifest["assessment_spec_path"]))
    _check_implementation_rules()
    return manifest


def _safe_file(path: Path, root: Path) -> Path:
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(Path(root).resolve()),
            f"Missing or unsafe evidence file: {path}")
    return path.resolve()


def _run_evidence(run_dir: Path, status_path: Path) -> dict[str, str]:
    """Check receipt paths before delegating to legacy verifiers that join them."""
    receipt = read_json(_safe_file(run_dir / "selected_checkpoint.json", run_dir))
    prediction = receipt.get("validation_predictions", {}).get("path")
    require(isinstance(prediction, str) and Path(prediction).name == prediction,
            "Unsafe validation prediction path")
    replay_root = run_dir / "independent_validation_replay"
    replay = read_json(_safe_file(replay_root / "replay_receipt.json", run_dir))
    replay_prediction = replay.get("validation_predictions", {}).get("path")
    require(isinstance(replay_prediction, str) and Path(replay_prediction).name == replay_prediction,
            "Unsafe independent replay prediction path")
    checkpoint = receipt.get("selected_checkpoint")
    require(checkpoint == "terminal_model_checkpoint.pt", "Only the frozen terminal checkpoint is allowed")
    paths = [run_dir / name for name in ("selected_checkpoint.json", "run_metadata.json", "run_config.json",
                                        "epoch_metrics.csv", checkpoint, prediction)]
    paths += [replay_root / "replay_receipt.json", replay_root / replay_prediction]
    evidence = {str(_safe_file(path, run_dir)): sha256(path) for path in paths}
    evidence[str(status_path.resolve())] = sha256(status_path)
    return evidence


def _flat_record(unit, metrics, **extra) -> dict:
    return {"name": unit.name, "complete": True, **{key: getattr(unit, key) for key in
                                                    ("phase", "recipe", "target", "fold", "seed")},
            "metrics": metrics, **extra}


def _new_unit(unit, source: dict, manifest: dict, manifest_digest: str, membership: dict) -> dict:
    root = Path(manifest["successor_root"]).resolve()
    run_dir = Path(source["run_dir"])
    require(run_dir.name == unit.name and run_dir.resolve().is_relative_to(root), "Run directory differs from unit")
    status_path = _safe_file(Path(source["status_path"]), root)
    require(status_path == (run_dir.parent.parent / f"run_status_{unit.name}.json").resolve(),
            "Status path does not belong to the unit's lane")
    rc_path = _safe_file(Path(source["worker_rc_path"]), root)
    require(rc_path.name == f"{unit.name}.rc", "Exit-code file must name this exact worker unit")
    require(rc_path.read_text().strip() == "0", "Final worker exit code is missing or nonzero")
    status = read_json(status_path)
    require(status.get("status") == "completed" and status.get("run_name") == unit.name
            and status.get("unit") == unit.name, "Worker status is not this completed unit")
    require(type(status.get("lane")) is int and status["lane"] in range(3), "Invalid recorded lane")
    paths = v3.V3Paths(root)
    require(run_dir.resolve() == (paths.lane(status["lane"]) / "runs" / unit.name).resolve(), "Run lane differs")
    parent = v3.V3Paths(Path(manifest["parent_campaign_dir"]))
    expected = campaign.build_command(parent, unit, successor_root=root, manifest_sha256=manifest_digest,
                                      code_sha256=manifest["code_sha256"], python_bin="python",
                                      train_dir=Path(manifest["train_dir"]), esm_dir=Path(manifest["esm_dir"]),
                                      device="cuda", lane=status["lane"],
                                      amp=v3.resolve_amp(parent, probe=None, amp=None))[2]
    require(status.get("identity") == expected, "Unit identity differs from frozen completion command")
    claim_path = _safe_file(paths.claims / f"{unit.name}.json", root)
    claim = read_json(claim_path)
    require(all(claim.get(key) == status.get(key) for key in ("run_name", "lane", "session_id", "identity")),
            "Final worker claim differs from its status")
    require(isinstance(status.get("session_id"), str) and status["session_id"], "Missing worker session")
    command_path = _safe_file(paths.lane(status["lane"]) / "commands" / f"{unit.name}.json", root)
    command = read_json(command_path)
    require(command.get("identity") == expected and command.get("run_name") == unit.name,
            "Saved command identity differs")
    argv = command["argv"]
    require(argv[-2] == "--campaign-run-identity" and json.loads(argv[-1]) == expected,
            "Saved command has a different identity argument")
    require(campaign.resolved_config_sha256(argv[:-2]) == expected["resolved_config_sha256"],
            "Saved command configuration differs")
    evidence = _run_evidence(run_dir, status_path)
    receipt = v3.verify_completed_unit(run_dir, expected, v3.resolve_recipe("baseline"))
    v3.verify_independent_replay(run_dir, receipt)
    with (run_dir / receipt["validation_predictions"]["path"]).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    old_assessment.validate_rows(rows, membership, fold=unit.fold, target=unit.target, seed=unit.seed,
                                 checkpoint_sha256=receipt["selected_checkpoint_sha256"],
                                 selected_epoch=receipt["selected_epoch"])
    metrics = old_assessment.prediction_metrics(rows, unit.target)
    for key, block in (("val_metal_collapsed4_balanced_acc", "common4"), ("val_metal_balanced_acc", "native")):
        require(abs(receipt["metrics"][key] - metrics[block]["balanced_accuracy"]) <= 1e-9,
                "Recomputed predictions disagree with receipt")
    evidence.update({str(path): sha256(path) for path in (rc_path, claim_path, command_path)})
    return _flat_record(unit, metrics, reused=False, evidence=evidence, identity=expected,
                        run_dir=str(run_dir.resolve()), status_path=str(status_path))


def _reused_unit(unit, proposed: dict, parent, parent_manifest: dict, membership: dict) -> dict:
    require(unit.recipe == "B" and (unit.seed == 42 or (unit.target == "four_class" and unit.fold == 0 and unit.seed == 43)),
            "Historical reuse outside the fixed 16-control allowance")
    require(all(proposed.get(key) == getattr(unit, key) for key in ("name", "phase", "recipe", "target", "fold", "seed"))
            and proposed.get("complete") is True, "Reuse record identifies another unit")
    old_unit = v3.Unit(campaign.FAMILY, unit.target, "baseline", unit.fold, unit.seed)
    require(proposed.get("parent_unit_name") == old_unit.name, "Reuse record identifies another parent unit")
    run_dir, status_path = Path(proposed["run_dir"]), Path(proposed["status_path"])
    status = read_json(status_path)
    require(status.get("identity") == proposed.get("parent_identity"), "Reuse parent identity changed")
    require(status.get("run_name") == run_dir.name
            and status_path.resolve() == (run_dir.parent.parent / f"run_status_{run_dir.name}.json").resolve(),
            "Reuse status/run directory mismatch")
    evidence = _run_evidence(run_dir, status_path)
    require(all(proposed.get("evidence", {}).get(path) == digest for path, digest in evidence.items()),
            "Reused actual artifact bytes differ from the verified control packet")
    verified = old_assessment.load_unit(parent, old_unit,
                                       {status["run_name"]: {**status, "status_path": str(status_path)}},
                                       membership, parent_manifest, epochs=50)
    require(verified.get("status") == "completed", "Historical control is not verified complete")
    require(verified["metrics"] == proposed.get("metrics"), "Historical control metrics do not reproduce")
    return _flat_record(unit, verified["metrics"], reused=True, evidence=evidence,
                        parent_unit_name=old_unit.name, parent_identity=status["identity"],
                        run_dir=str(run_dir.resolve()), status_path=str(status_path.resolve()))


def collect(*, manifest_path: Path, manifest_sha256: str, esm_dir: Path, runs: dict,
            reuse_packet_path: Path | None = None, reuse_packet_sha256: str | None = None) -> dict:
    """Return a complete, reproducible collection; missing units fail before assessment."""
    manifest = _manifest(manifest_path, manifest_sha256, esm_dir)
    parent = v3.V3Paths(Path(manifest["parent_campaign_dir"]))
    parent_manifest, membership = old_assessment.read_contract(parent)
    require(isinstance(runs, dict), "Run source map must be an object")
    expected = {name: campaign.Unit.parse(name) for name in manifest["authorized_grid_units"]}
    require(set(runs) <= set(expected), "Run map contains units outside this phase")
    reused = {}
    if reuse_packet_path is not None:
        require(reuse_packet_sha256 is not None, "Reuse packet requires its SHA-256")
        packet = bound_json(reuse_packet_path, reuse_packet_sha256)
        require(packet.get("schema_version") == 1
                and packet.get("parent_manifest_sha256") == manifest["parent_manifest_sha256"],
                "Reuse packet describes another parent")
        require(isinstance(packet.get("records"), dict), "Malformed reuse records")
        reused = {name: record for name, record in packet["records"].items() if name in expected}
    require(not set(runs) & set(reused), "A unit cannot be both newly run and historically reused")
    require(set(runs) | set(reused) == set(expected), "Incomplete grid: every unit needs one verified source")
    records = {name: (_reused_unit(unit, reused[name], parent, parent_manifest, membership) if name in reused
                      else _new_unit(unit, runs[name], manifest, manifest_sha256, membership))
               for name, unit in expected.items()}
    # Schema/support checks are applied even before producing a collection receipt.
    units = list(expected.values())
    assessment._normalize(records, units)
    return {"schema_version": 1, "campaign_id": campaign.CAMPAIGN_ID, "phase": manifest["phase"],
            "manifest_path": str(Path(manifest_path).resolve()), "manifest_sha256": manifest_sha256,
            "parent_manifest_sha256": manifest["parent_manifest_sha256"], "code_sha256": code_hashes(),
            "records": records, "records_sha256": campaign.stable_hash(records),
            "sources": {"runs": runs, "reuse_packet_path": str(Path(reuse_packet_path).resolve()) if reuse_packet_path else None,
                        "reuse_packet_sha256": reuse_packet_sha256}, "held_out_accessed": False}


def assess(*, collection_path: Path, collection_sha256: str, esm_dir: Path, output_path: Path) -> dict:
    """Recollect source evidence and write an immutable, input-bound decision receipt."""
    saved = bound_json(collection_path, collection_sha256)
    current = collect(manifest_path=Path(saved["manifest_path"]), manifest_sha256=saved["manifest_sha256"],
                      esm_dir=esm_dir, runs=saved["sources"]["runs"],
                      reuse_packet_path=Path(saved["sources"]["reuse_packet_path"]) if saved["sources"]["reuse_packet_path"] else None,
                      reuse_packet_sha256=saved["sources"]["reuse_packet_sha256"])
    require(current == saved, "Collection does not reproduce from current bound evidence")
    manifest = read_json(Path(saved["manifest_path"]))
    result = (assessment.discover(current["records"]) if current["phase"] == "discovery" else
              assessment.confirmation(current["records"], manifest["shared_recipe"]))
    receipt = {"schema_version": 1, "campaign_id": campaign.CAMPAIGN_ID, "result": result,
               "collection_path": str(Path(collection_path).resolve()), "collection_sha256": collection_sha256,
               "manifest_sha256": current["manifest_sha256"], "records_sha256": current["records_sha256"],
               "parent_manifest_sha256": current["parent_manifest_sha256"], "code_sha256": code_hashes(),
               "held_out_accessed": False, "execution_authorized": False}
    write_once(output_path, receipt, root=Path(manifest["successor_root"]))
    return receipt


def verify_assessment_receipt(path: Path, digest: str, *, esm_dir: Path) -> dict:
    """Verify a selection receipt from its original collected inputs without writing."""
    receipt = bound_json(path, digest)
    require(receipt.get("schema_version") == 1 and receipt.get("campaign_id") == campaign.CAMPAIGN_ID
            and receipt.get("code_sha256") == code_hashes(), "Assessment receipt code/schema changed")
    saved = bound_json(Path(receipt["collection_path"]), receipt["collection_sha256"])
    current = collect(manifest_path=Path(saved["manifest_path"]), manifest_sha256=saved["manifest_sha256"],
                      esm_dir=esm_dir, runs=saved["sources"]["runs"],
                      reuse_packet_path=Path(saved["sources"]["reuse_packet_path"]) if saved["sources"]["reuse_packet_path"] else None,
                      reuse_packet_sha256=saved["sources"]["reuse_packet_sha256"])
    require(current == saved and receipt["records_sha256"] == current["records_sha256"]
            and receipt["manifest_sha256"] == current["manifest_sha256"]
            and receipt["parent_manifest_sha256"] == current["parent_manifest_sha256"], "Selection input binding differs")
    manifest = read_json(Path(saved["manifest_path"]))
    expected = (assessment.discover(current["records"]) if current["phase"] == "discovery" else
                assessment.confirmation(current["records"], manifest["shared_recipe"]))
    require(receipt.get("result") == expected, "Selection result does not reproduce")
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare", allow_abbrev=False)
    for name in ("parent-dir", "successor-root", "train-dir", "esm-dir", "assessment-spec-path"):
        prep.add_argument(f"--{name}", type=Path, required=True)
    prep.add_argument("--phase", choices=("discovery", "confirmation"), default="discovery")
    prep.add_argument("--selection-receipt-path", type=Path)
    prep.add_argument("--selection-receipt-sha256")
    coll = sub.add_parser("collect", allow_abbrev=False)
    coll.add_argument("--manifest-path", type=Path, required=True)
    coll.add_argument("--manifest-sha256", required=True)
    coll.add_argument("--esm-dir", type=Path, required=True)
    coll.add_argument("--runs-map", type=Path, required=True)
    coll.add_argument("--reuse-packet-path", type=Path)
    coll.add_argument("--reuse-packet-sha256")
    coll.add_argument("--output-path", type=Path, required=True)
    decision = sub.add_parser("assess", allow_abbrev=False)
    decision.add_argument("--collection-path", type=Path, required=True)
    decision.add_argument("--collection-sha256", required=True)
    decision.add_argument("--esm-dir", type=Path, required=True)
    decision.add_argument("--output-path", type=Path, required=True)
    args = vars(parser.parse_args(argv))
    action = args.pop("action")
    if action == "prepare":
        result = prepare(**args)
    elif action == "collect":
        output = args.pop("output_path")
        args["runs"] = read_json(args.pop("runs_map"))
        result = collect(**args)
        manifest = read_json(args["manifest_path"])
        write_once(output, result, root=Path(manifest["successor_root"]))
        result = {"collection_path": str(output), "collection_sha256": sha256(output),
                  "units": len(result["records"]), "execution_authorized": False}
    else:
        result = assess(**args)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
