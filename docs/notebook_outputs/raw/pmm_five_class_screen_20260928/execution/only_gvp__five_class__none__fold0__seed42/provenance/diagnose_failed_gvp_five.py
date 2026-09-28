"""Preserved-CSV diagnosis and failure evidence; never reruns or certifies a fit."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
UNIT = "only_gvp__five_class__none__fold0__seed42"
TRANSFER = "ce88c0be10764fd0b35a2a6e807ee2c5"
MANIFEST_SHA = "ea8c893054da02a551cff7452c7f1653fce5b2b6d9a7955609cdc6f784625d37"
TOLERANCE = Decimal("0.000001")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def module(name, path, digest):
    require(sha(path) == digest, f"Helper/source changed: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def load_verified_backup(copy):
    manifest_path = copy.safe_source(f"persistence_receipts/{TRANSFER}.json")
    require(sha(manifest_path) == MANIFEST_SHA, "Unexpected failure manifest")
    manifest = copy.read(manifest_path)
    ack_relative = f"persistence_receipts/{TRANSFER}.ack.json"
    require(manifest["ack_path"] == ack_relative and manifest["persistence_mode"] == "host_pull"
            and Path(manifest["destination_root"]).resolve() == copy.C, "Wrong host transfer")
    ack_path = copy.safe_source(ack_relative)
    ack = copy.read(ack_path)
    require(ack["manifest_sha256"] == MANIFEST_SHA and ack["file_count"] == len(manifest["files"]) == 70
            and Path(ack["verified_root"]).resolve() == copy.C and ack["verified_by"]
            and ack["verified_unix"] >= manifest["created_unix"], "Host acknowledgment failed")
    hashes = {str(manifest_path): MANIFEST_SHA, str(ack_path): sha(ack_path)}
    indexed = {}
    for item in manifest["files"]:
        path = copy.safe_source(item["path"])
        require(item["path"] not in indexed and path.stat().st_size == item["size_bytes"]
                and sha(path) == item["sha256"], f"Backup hash mismatch: {path}")
        indexed[item["path"]] = item
        hashes[str(path)] = item["sha256"]
    state_path = f"persistence_receipts/{TRANSFER}.state.json"
    require(state_path in indexed, "Missing hashed terminal snapshot")
    state = copy.read(copy.safe_source(state_path))
    require(state["pending_unit"]["run_name"] == UNIT
            and state["pending_unit"]["status"] == "failed_independent_replay"
            and state["pending_unit"]["session_id"] == manifest["session_id"]
            and state["pending_transfer"]["manifest_path"] == f"persistence_receipts/{TRANSFER}.json"
            and state["pending_transfer"]["ack_path"] == ack_relative, "Wrong terminal failure snapshot")
    run = copy.C / "runs" / UNIT
    require({str(p.relative_to(copy.C)) for p in run.rglob("*") if p.is_file()} <= set(indexed),
            "Run has artifacts outside verified backup")
    require(not (run / "independent_validation_replay/replay_receipt.json").exists(),
            "Unexpected strict replay receipt; stop for reconciliation")
    return run, manifest, ack, state, indexed, hashes


def compare_rows(saved, replay):
    left, right = ({row["source_uid"]: row for row in rows} for rows in (saved, replay))
    require(len(left) == len(saved) == len(right) == len(replay) and left.keys() == right.keys(),
            "Duplicate or different UID sets; this diagnosis does not infer an alignment")
    require(list(saved[0]) == list(replay[0]), "Export columns differ")
    probability_columns = [key for key in saved[0] if key.startswith("p_")]
    nonprobability = [(uid, key) for uid, row in left.items() for key in row
                      if key not in probability_columns and row[key] != right[uid][key]]
    statistics, affected = {}, set()
    for column in probability_columns:
        diffs = [(abs(Decimal(left[uid][column]) - Decimal(right[uid][column])), uid) for uid in left]
        worst, uid = max(diffs)
        over = [row_uid for diff, row_uid in diffs if diff > TOLERANCE]
        affected.update(over)
        statistics[column] = {"maximum_absolute_difference": str(worst), "worst_source_uid": uid,
                              "saved_probability": left[uid][column], "replay_probability": right[uid][column],
                              "fields_above_original_tolerance": len(over),
                              "nonzero_difference_fields": sum(diff != 0 for diff, _ in diffs)}
    return {"n_ions": len(left), "uid_set_exact": True, "column_order_exact": True,
            "row_order_exact": [r["source_uid"] for r in saved] == [r["source_uid"] for r in replay],
            "nonprobability_fields_exact": not nonprobability, "nonprobability_differences": nonprobability,
            "native_prediction_changes": sum(left[uid]["pred_native"] != right[uid]["pred_native"] for uid in left),
            "common4_prediction_changes": sum(left[uid]["pred_common4"] != right[uid]["pred_common4"] for uid in left),
            "probability_columns": statistics, "ions_above_original_tolerance": len(affected),
            "probability_fields_above_original_tolerance": sum(s["fields_above_original_tolerance"] for s in statistics.values()),
            "maximum_absolute_probability_difference": str(max(Decimal(s["maximum_absolute_difference"]) for s in statistics.values()))}


def diagnose():
    copy = module("portable", HERE / "copy_verified_five_unit.py", "de40603a0ab953e692753b3f7130c5e7bdef3fcd25c958b9740174a9f36ea59d")
    reporter = module("reporter", HERE / "report_five_screen.py", "e34c66156e71efeddb7d0df574cfc5c56b880ca4a7f27bcd689581d925e545b7")
    run, manifest, ack, state, indexed, hashes = load_verified_backup(copy)
    screen = module("screen", copy.CODE / "run_pmm_five_class_screen.py", copy.ADAPTER_SHA)
    from benchmarking import pmm_ion_campaign as campaign
    from training.campaign_runtime import read_fold_membership
    from training.access_guard import install_forbidden_read_guard

    install_forbidden_read_guard(campaign.forbidden_read_roots(copy.TRAIN))
    require(screen.source_tree_sha256() == copy.SOURCE_SHA and sha(screen.PROTOCOL_PATH) == copy.PROTOCOL_SHA,
            "Frozen scientific source or protocol changed")
    screen.load_protocol(screen.PROTOCOL_PATH)
    paths = campaign.CampaignPaths(copy.C)
    identity = screen.build_command(paths, SimpleNamespace(python_bin=sys.executable, train_dir=copy.TRAIN,
                                    action="plan", load_workers=None), "only_gvp", copy.PROTOCOL_SHA)[2]
    command = copy.read(copy.safe_source(f"commands/{UNIT}.json"))
    terminal = copy.read(copy.safe_source(f"run_status_five_screen_{UNIT}.json"))["units"]
    require(command["identity"] == identity and len(terminal) == 1 and terminal[0]["run_name"] == UNIT
            and terminal[0]["status"] == "failed_independent_replay", "Wrong command or failure identity")
    receipt = campaign.completed_run_receipt(run, identity, require_independent=False)
    require(receipt is not None and campaign.completed_run_receipt(run, identity) is None,
            "Expected a completed fit with failed independent replay")
    membership = read_fold_membership(paths.fold_membership, identity["fold_membership_sha256"])
    exports, metrics = {}, {}
    for label, directory in (("selected_training_export", run), ("failed_independent_export", run / "independent_validation_replay")):
        with (directory / "val_predictions.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        screen.validate_five_rows(rows, membership, checkpoint_sha256=receipt["selected_checkpoint_sha256"],
                                  selected_epoch=int(receipt["selected_epoch"]))
        export_receipt = copy.read(directory / "selected_checkpoint.json")
        require(export_receipt["campaign_run_identity"] == identity
                and export_receipt["selected_checkpoint_sha256"] == receipt["selected_checkpoint_sha256"],
                "Exports do not refer to the same selected checkpoint")
        exports[label] = rows
        metrics[label] = {"native5": reporter.classification(rows, "native", reporter.FIVE),
                          "common4": reporter.classification(rows, "common4", reporter.FOUR)}
        for metric_view, prefix in (("native5", "val_metal"), ("common4", "val_metal_collapsed4")):
            for metric, suffix in (("balanced_accuracy", "balanced_acc"), ("macro_f1", "macro_f1")):
                require(abs(metrics[label][metric_view][metric] - export_receipt["metrics"][prefix + "_" + suffix]) <= 1e-9,
                        "Derived metrics do not reconcile to this export receipt")
    difference = compare_rows(exports["selected_training_export"], exports["failed_independent_export"])
    require(difference["probability_fields_above_original_tolerance"] > 0, "Failure basis differs; do not label a numerical mismatch")
    control_evidence = reporter.Evidence()
    control = reporter.collect(screen, campaign, paths, membership, "only_gvp", "six_class",
                               copy.PROTOCOL_SHA, control_evidence)
    require(control["status"] == "historical_supplemental_v2_1_only" and not control["strict_replay_certified"]
            and control["uid_identity_sha256"] == reporter.unit_fingerprint(exports["selected_training_export"]),
            "Historical six-class control is not the verified matching supplemental control")
    control_evidence.recheck()
    hashes.update(control_evidence.files)
    common = metrics["selected_training_export"]["common4"]
    provisional_comparison = {
        "status": "provisional_point_estimate_only_not_certified_comparison",
        "five_status": "trained_but_strict_replay_failed", "six_status": control["status"],
        "n_paired_ions": difference["n_ions"], "common4_five": common, "common4_six": control["common4"],
        "five_minus_six_common4_ba_pp": 100 * (common["balanced_accuracy"] - control["common4"]["balanced_accuracy"]),
        "five_minus_six_common4_macro_f1_pp": 100 * (common["macro_f1"] - control["common4"]["macro_f1"]),
        "comparison_source": "Same selected checkpoints and matched fold0/seed42 validation ions; no checkpoint reselection or formal inference."}
    report = {"schema_version": 1, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": "trained_but_strict_replay_failed", "strict_replay_pass": False, "provisional_metrics": True,
              "run_name": UNIT, "selected_epoch": receipt["selected_epoch"], "completed_epochs": 50,
              "checkpoint_sha256": receipt["selected_checkpoint_sha256"], "campaign_run_identity": identity,
              "original_probability_tolerance": str(TOLERANCE), "threshold_changed": False,
              "rerun_performed": False, "held_out_access": False, "promotion": False,
              "input_tensor_equality_audited": False, "repeatability_measured": False, "cause_identified": False,
              "limitations": "Preserved serialized CSVs only. Identical class predictions do not satisfy strict probability replay and do not certify the run. No input-tensor equality or repeated-forward diagnostic was performed; numerical cause remains unproven. Single fold and seed, no formal inference. The strict screen report must continue excluding this five-class fit.",
              "export_comparison": difference, "metrics": metrics,
              "provisional_five_vs_six": provisional_comparison,
              "backup": {"manifest_id": TRANSFER, "manifest_sha256": MANIFEST_SHA, "verified_files": len(indexed),
                         "host_ack": ack, "terminal_status": state["pending_unit"]["status"]}}
    for path in (__file__, HERE / "copy_verified_five_unit.py", HERE / "report_five_screen.py",
                 copy.CODE / "run_pmm_five_class_screen.py", screen.PROTOCOL_PATH,
                 paths.manifest, paths.cohort, paths.fold_membership, paths.fold_class_weights, paths.feature_inventory):
        hashes[str(Path(path).resolve())] = sha(path)
    report["input_sha256"] = hashes
    return copy, run, indexed, report


def markdown(report):
    diff, metrics = report["export_comparison"], report["metrics"]["selected_training_export"]
    lines = ["# Only-GVP five-class: preserved strict replay failure", "",
             "**Status: trained_but_strict_replay_failed. All metric point estimates below are provisional.**", "",
             f"50 epochs completed; selected epoch {report['selected_epoch']}. Verified host backup: 70 files. No rerun or tolerance change.", "",
             f"Both exports cover {diff['n_ions']} identical validation ions. Exact nonprobability fields: {diff['nonprobability_fields_exact']}. Native/common-four prediction changes: {diff['native_prediction_changes']}/{diff['common4_prediction_changes']}.", "",
             f"Maximum absolute probability difference: {diff['maximum_absolute_probability_difference']}; {diff['probability_fields_above_original_tolerance']} fields across {diff['ions_above_original_tolerance']} ions exceed the original 0.000001 threshold.", "",
             "| Probability column | Maximum absolute difference | Fields > 0.000001 |", "|---|---:|---:|"]
    lines += [f"| {key} | {row['maximum_absolute_difference']} | {row['fields_above_original_tolerance']} |"
              for key, row in diff["probability_columns"].items()]
    lines += ["", "| Provisional view | Balanced accuracy | Macro-F1 |", "|---|---:|---:|"]
    lines += [f"| {view} | {100*values['balanced_accuracy']:.4f}% | {100*values['macro_f1']:.4f}% |" for view, values in metrics.items()]
    for view, values in metrics.items():
        lines += ["", view + " recalls: " + ", ".join(f"{label} {100*value:.4f}%" for label, value in values["recall"].items()) + "."]
    comparison = report["provisional_five_vs_six"]
    lines += ["", "**Separate provisional five-vs-six point estimate; excluded from the certified screen comparison:** "
              f"common-four BA {100*comparison['common4_five']['balanced_accuracy']:.4f}% versus "
              f"{100*comparison['common4_six']['balanced_accuracy']:.4f}% "
              f"({comparison['five_minus_six_common4_ba_pp']:+.4f} percentage points). "
              "GVP5 failed strict replay; historical GVP6 has supplemental v2.1 agreement only. Neither status is upgraded here."]
    lines += ["", "Native Class VIII means Co+Ni. Common-four Class VIII means Fe+Co+Ni, summed before argmax.", "",
              "The two exports produce identical class-based metrics when predictions match; that observation is not replay certification.", "", report["limitations"], "",
              "JSON retains every column's worst UID/value pair, metrics/confusion matrices and source hashes. Canonical checkpoint binaries and full metadata remain in the verified backup."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--write", action="store_true", help="Write new diagnostic and failed portable evidence folders")
    args = parser.parse_args(argv)
    copy, run, indexed, report = diagnose()
    diagnostic = HERE / "gvp5_failed_replay_diagnostic_01"
    portable = copy.DEST / UNIT
    if args.write:
        require(not diagnostic.exists() and not portable.exists() and not diagnostic.is_symlink() and not portable.is_symlink(),
                "Diagnostic and portable destinations must both be new")
        files = {relative: run / relative for relative in copy.RUN_FILES if relative != "independent_validation_replay/replay_receipt.json"}
        files.update({"training_and_failed_replay.log": copy.C / "runs" / (UNIT + ".log"),
                      "command_record.json": copy.C / "commands" / (UNIT + ".json"),
                      "terminal_failure_status.json": copy.C / f"run_status_five_screen_{UNIT}.json",
                      "provenance/diagnose_failed_gvp_five.py": Path(__file__).resolve(),
                      "provenance/five_class_screen_protocol.json": copy.CODE / "docs/plans/pmm_five_class_screen_v1.json",
                      "provenance/run_pmm_five_class_screen.py": copy.CODE / "run_pmm_five_class_screen.py"})
        files.update({"persistence_receipts/" + TRANSFER + suffix: copy.C / "persistence_receipts" / (TRANSFER + suffix)
                      for suffix in (".json", ".ack.json", ".state.json", ".events.jsonl")})
        derived = {Path(name).stem + "_excerpt.json": copy.excerpt(run / name, omitted, indexed[f"runs/{UNIT}/{name}"])
                   for name, omitted in copy.OMISSIONS.items()}
        data = {"failed_replay_diagnostic.json": (json.dumps(report, indent=2, sort_keys=True) + "\n").encode(),
                "failed_replay_diagnostic.md": markdown(report).encode()}
        for path, expected in report["input_sha256"].items():
            require(sha(path) == expected, f"Input changed during diagnosis: {path}")
        diagnostic.mkdir()
        for name, content in data.items():
            (diagnostic / name).write_bytes(content)
        portable.parent.mkdir(parents=True, exist_ok=True)
        portable.mkdir()
        descriptors = {}
        for relative, source in files.items():
            expected = report["input_sha256"][str(source.resolve())]
            target = portable / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as incoming, target.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
            require(sha(target) == expected and sha(source) == expected, f"Copy changed: {relative}")
            descriptors[relative] = {"sha256": expected, "exact_copy": True, "source_path": str(source)}
        for name, payload in derived.items():
            data[name] = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        for name, content in data.items():
            with (portable / name).open("xb") as stream:
                stream.write(content)
            descriptors[name] = {"sha256": hashlib.sha256(content).hexdigest(), "derived": True}
        manifest = {"status": report["status"], "strict_replay_pass": False, "run_name": UNIT,
                    "provisional_metrics": True, "host_manifest_sha256": MANIFEST_SHA,
                    "helper_sha256": sha(__file__), "artifacts": descriptors,
                    "note": "Failure evidence, not a successful completion receipt. No binary weights/full repeated metadata copied; originals remain canonical."}
        with (portable / "failed_portable_evidence_manifest.json").open("x") as stream:
            stream.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "diagnostic_dir": str(diagnostic), "portable_dir": str(portable),
                      "files_written": args.write, "comparison": report["export_comparison"]}, indent=2))


if __name__ == "__main__":
    main()
