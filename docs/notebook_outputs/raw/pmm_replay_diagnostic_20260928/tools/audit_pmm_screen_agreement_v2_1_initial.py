"""Append-only, CPU-only consumer of pmm-screen-agreement-v2.1; never a legacy receipt."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
POLICY_PATH = ROOT / "docs/plans/pmm_screen_agreement_v2_1.json"
POLICY_SHA256 = "44d595b2a73718c8c30540fd68f0781a26dc80c85616811ef607158805b983b6"
FOUR = ("Mn", "Cu", "Zn", "Class VIII")
SIX = ("Mn", "Cu", "Zn", "Fe", "Co", "Ni")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def checked_file(path, expected):
    path = Path(path).resolve()
    require(path.is_file() and sha(path) == expected, f"Artifact hash mismatch: {path}")
    return {"path": str(path), "sha256": expected}


def load_policy():
    checked_file(POLICY_PATH, POLICY_SHA256)
    return read_json(POLICY_PATH)


def read_rows(path):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        require(len(columns) == len(set(columns)) and columns, f"Repeated/empty CSV columns: {path}")
        rows = list(reader)
    require(rows and all(set(row) == set(columns) and None not in row.values() for row in rows),
            f"Malformed/empty predictions: {path}")
    require(len({row["source_uid"] for row in rows}) == len(rows), f"Duplicate prediction UIDs: {path}")
    return rows


def validate_probabilities(rows, labels, policy):
    """Decimal checks keep the frozen absolute boundary independent of float rounding."""
    allowance = Decimal(policy["serialized_probability_consistency_atol"])
    simplex_allowance = Decimal(policy["probability_simplex_atol"])
    columns = {"p_native_" + label.replace(" ", "_") for label in labels}
    columns |= {"p_common4_" + label.replace(" ", "_") for label in FOUR}
    minimum_margins = {"native": Decimal(1), "common4": Decimal(1)}
    for row in rows:
        require({key for key in row if key.startswith("p_")} == columns, "Probability vocabulary mismatch")
        views = {}
        for view, vocabulary in (("native", labels), ("common4", FOUR)):
            try:
                values = [Decimal(row[f"p_{view}_{label.replace(' ', '_')}"]) for label in vocabulary]
            except InvalidOperation as exc:
                raise ValueError("Malformed probability") from exc
            require(all(value.is_finite() and 0 <= value <= 1 for value in values), "Invalid finite probability range")
            require(abs(sum(values) - 1) <= simplex_allowance, "Probability vector does not sum to one")
            predicted = int(row[f"pred_{view}"])
            # Decimal serialization can tie an original FP32 argmax; no larger gap is accepted.
            require(predicted in range(len(values)) and max(values) == values[predicted],
                    "Prediction disagrees with serialized probability argmax")
            ordered = sorted(values, reverse=True)
            minimum_margins[view] = min(minimum_margins[view], ordered[0] - ordered[1])
            views[view] = values
        collapsed = views["native"][:3] + [sum(views["native"][3:])] if len(labels) == 6 else views["native"]
        require(all(abs(a - b) <= allowance for a, b in zip(collapsed, views["common4"])),
                "Native-to-common-four collapse mismatch")
    return {key: float(value) for key, value in minimum_margins.items()}


def compare_rows(original, replay, labels, policy):
    validate_probabilities(original, labels, policy)
    margins = validate_probabilities(replay, labels, policy)
    left = {row["source_uid"]: row for row in original}
    right = {row["source_uid"]: row for row in replay}
    require(len(left) == len(original) and len(right) == len(replay) and left.keys() == right.keys(),
            "Missing, extra or duplicate replay UIDs")
    atol, legacy = Decimal(policy["probability_atol"]), Decimal(policy["legacy_probability_atol"])
    worst = {"abs_difference": 0.0, "source_uid": None, "column": None}
    maximum = Decimal(0)
    over_legacy = over_policy = 0
    for uid, row in left.items():
        other = right[uid]
        require(row.keys() == other.keys(), f"Replay columns differ at {uid}")
        for key, value in row.items():
            if key.startswith("p_"):
                difference = abs(Decimal(value) - Decimal(other[key]))
                over_legacy += difference > legacy
                over_policy += difference > atol
                if difference > maximum:
                    maximum = difference
                    worst = {"abs_difference": float(difference), "source_uid": uid, "column": key}
            else:
                require(value == other[key], f"Replay discrete/metadata mismatch: {uid}/{key}")
    return {"qualified": over_policy == 0, "worst": worst, "above_legacy_tolerance_fields": over_legacy,
            "above_policy_tolerance_fields": over_policy, "minimum_prediction_margins": margins,
            "n_rows": len(original)}


def same_metrics(left, right, tolerance=1e-9):
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            same_metrics(left[key], right[key], tolerance) for key in left)
    if isinstance(left, list):
        return isinstance(right, list) and len(left) == len(right) and all(
            same_metrics(a, b, tolerance) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and not isinstance(left, bool):
        return isinstance(right, (int, float)) and math.isfinite(left) and math.isfinite(right) and abs(left - right) <= tolerance
    return left == right


def diagnostic_gate(path, policy):
    path = Path(path).resolve()
    summary = read_json(path)
    for key in ("predictive_inputs_matched", "original_variable", "all_classes_stable", "no_mutation"):
        require(summary.get(key) is True, f"Diagnostic gate not met: {key}")
    require(summary.get("source_tree_sha256") == policy["source_tree_sha256"], "Diagnostic scientific source differs")
    require(summary.get("run_name") == "only_gvp__six_class__none__fold0__seed42", "Wrong diagnostic arm")
    difference = summary.get("original_max_abs_probability_difference")
    require(isinstance(difference, (float, int)) and math.isfinite(difference)
            and 0 < difference <= float(policy["probability_atol"]), "Diagnostic probability bound not met")
    bound_files = [checked_file(path, sha(path))]
    evidence = summary.get("evidence_files", [])
    require(evidence, "Diagnostic evidence files are absent")
    for item in evidence:
        bound_files.append(checked_file(path.parent / item["path"], item["sha256"]))
    manifests = [Path(item["path"]) for item in bound_files if Path(item["path"]).name == "input_manifest.json"]
    require(len(manifests) == 1, "Exactly one bound diagnostic input manifest is required")
    manifest = read_json(manifests[0])
    require(manifest.get("status") == "predictive_inputs_matched"
            and manifest.get("n_examples") == policy["validation_ions"]
            and manifest["campaign_run_identity"]["source_tree_sha256"] == policy["source_tree_sha256"],
            "Diagnostic input manifest did not pass or describes other inputs")
    require(summary["checkpoint_sha256"] in [digest for name, digest in manifest["input_files"].items()
                                             if Path(name).name == "best_model_checkpoint.pt"],
            "Diagnostic checkpoint hash is not bound to its inputs")
    bound_files.append(checked_file(manifests[0].parent / "input_snapshot.pt", manifest["snapshot_sha256"]))
    input_hashes = [item["fresh_sha256"] for item in manifest["batches"]]
    require(input_hashes and len(input_hashes) == manifest["n_batches"], "Missing diagnostic batch identities")
    model_hashes = []
    processes = summary.get("processes", [])
    require(len(processes) == 4 and all(sum(item.get("condition") == condition for item in processes) == 2
                                      for condition in ("original", "strict")), "Diagnostic needs two processes per condition")
    report_paths = []
    for item in processes:
        report_path = (path.parent / item["report_path"]).resolve()
        report_paths.append(report_path)
        bound_files.append(checked_file(report_path, item["report_sha256"]))
        report = read_json(report_path)
        require(report.get("input_manifest_sha256") == sha(manifests[0])
                and report.get("tool_sha256") == manifest["tool_sha256"], "Diagnostic report input/tool hash differs")
        require(report.get("condition") == item["condition"] and report.get("planned_repeats") == 5,
                "Diagnostic process condition/repeats mismatch")
        require(report.get("deterministic_warn_only") is False
                and report.get("deterministic_algorithms") is (item["condition"] == "strict"),
                "Diagnostic effective determinism settings differ")
        preserved = report.get("preserved_passes", [])
        require(len(preserved) == item["completed_passes"] == report.get("completed_repeats"),
                "Diagnostic preserved-pass count differs")
        if item["status"] == "passed":
            require(item["completed_passes"] == 5 and report.get("status") == "diagnosis_complete",
                    "Diagnostic process did not complete all five passes")
            require(report.get("input_hashes_before_and_after") == input_hashes,
                    "Diagnostic input invariant hashes differ")
            model_hashes.append(report.get("model_state_sha256_before_and_after"))
            for comparisons in (report.get("comparisons_to_saved", []), report.get("pairwise_comparisons", [])):
                require(comparisons, "Diagnostic probability comparisons missing")
                for comparison in comparisons:
                    for view in ("native", "common4"):
                        require(comparison[view]["changed_predictions"] == 0, "Diagnostic class predictions changed")
                        if item["condition"] == "original":
                            require(comparison[view]["max_abs_difference"] <= float(policy["probability_atol"]),
                                    "Original diagnostic pass exceeds fixed bound")
        else:
            require(item["condition"] == "strict" and item["status"] == "unsupported"
                    and report.get("status") == "evaluation_failed" and "determin" in report.get("error", "").lower(),
                    "Only explicitly recorded unsupported strict evaluation can replace successful passes")
        for saved in preserved:
            bound_files.append(checked_file(report_path.parent / saved["path"], saved["sha256"]))
        if report.get("forward_passes_sha256"):
            bound_files.append(checked_file(report_path.parent / "forward_passes.pt", report["forward_passes_sha256"]))
    require(len(set(report_paths)) == 4, "Diagnostic processes reuse a report")
    require(model_hashes and isinstance(model_hashes[0], str) and len(model_hashes[0]) == 64
            and len(set(model_hashes)) == 1, "Diagnostic model state invariant hashes differ")
    return summary, bound_files


def replay_directories(run):
    directories = [run / "independent_validation_replay"]
    archive = run / "_incomplete_replays"
    if archive.exists():
        directories += sorted(path for path in archive.iterdir() if path.is_dir())
    require(all((path / "val_predictions.csv").is_file() and (path / "selected_checkpoint.json").is_file()
                for path in directories), "Preserved replay attempt lacks predictions/receipt")
    return directories


def verify_legacy_consistency(legacy_receipt_pass, current_comparison):
    require(not legacy_receipt_pass or current_comparison["above_legacy_tolerance_fields"] == 0,
            "Legacy receipt claims a pass but current predictions fail its unchanged tolerance")


def inspect_unit(paths, train_dir, config, membership, policy):
    import torch
    from dataclasses import fields
    from benchmarking.pmm_ion_analysis import expected_identity, native_labels_for, prediction_metrics
    from benchmarking.pmm_ion_campaign import completed_run_receipt, run_name, NON_IDENTITY_CONFIG_KEYS, stable_hash
    from benchmarking.pmm_comparator import validate_prediction_rows
    from training.config import TrainConfig
    from training.run import to_jsonable

    identity = expected_identity(paths, train_dir, config, 0, 42)
    require(identity["source_tree_sha256"] == policy["source_tree_sha256"], "Frozen scientific source changed")
    run = paths.runs / run_name(config, 0, 42)
    receipt = completed_run_receipt(run, identity, require_independent=False)
    require(receipt is not None, f"Completed fit identity/hash checks failed: {run.name}")
    bound = {str(run / name): sha(run / name) for name in (
        "run_config.json", "run_metadata.json", "dataset_summary.json", "best_model_checkpoint.pt",
        "selected_checkpoint.json", "val_predictions.csv")}
    payload = read_json(run / "run_config.json")
    checkpoint = torch.load(run / "best_model_checkpoint.pt", map_location="cpu", weights_only=False)
    require(to_jsonable(checkpoint["config"]) == payload["config"], "Checkpoint/config mismatch")
    require(to_jsonable(checkpoint["normalization_stats"]) == payload["normalization_stats"], "Normalization mismatch")
    require(to_jsonable(checkpoint["dataset_summary"]) == payload["dataset_summary"]
            == read_json(run / "dataset_summary.json"), "Checkpoint/dataset-summary mismatch")
    require(json.loads(payload["config"]["campaign_run_identity"]) == identity, "Saved configuration identity differs")
    require(checkpoint["epoch"] == receipt["selected_epoch"], "Checkpoint/receipt epoch mismatch")
    require(receipt["completed_epochs"] == receipt["planned_epochs"] == policy["epochs"], "Receipt epoch count differs")
    cfg = payload["config"]
    scientific = {field.name: cfg[field.name] for field in fields(TrainConfig)
                  if field.name not in NON_IDENTITY_CONFIG_KEYS}
    require(stable_hash(scientific) == identity["resolved_config_sha256"], "Resolved scientific configuration hash differs")
    require(cfg.get("task") == "metal" and cfg.get("metal_example_unit") == "ion", "Wrong scientific task/unit")
    require(not any(cfg.get(key) for key in ("run_test_eval", "test_structure_dir", "test_summary_csv",
                                            "allow_final_refit_test_eval", "allow_train_loss_test_eval_debug")),
            "Held-out configuration forbidden")
    history = payload["history"]
    require(len(history) == 50 and [row["epoch"] for row in history] == list(range(1, 51)), "Incomplete epoch history")
    chosen = max(history, key=lambda row: row["val_metal_balanced_acc"])
    require(chosen["epoch"] == receipt["selected_epoch"] and receipt["selection_metric"] == "val_metal_balanced_acc",
            "Native-BA checkpoint selection/earliest-tie rule differs")
    require(same_metrics(receipt["selection_metric_value"], chosen["val_metal_balanced_acc"]),
            "Selected metric value differs from epoch history")
    labels = native_labels_for(config)
    require([checkpoint["metal_labels"][key] for key in sorted(checkpoint["metal_labels"], key=int)] == list(labels),
            "Checkpoint target vocabulary differs")
    del checkpoint
    original = read_rows(run / "val_predictions.csv")
    require(len(original) == policy["validation_ions"], "Wrong validation-ion count")
    require(receipt["selected_checkpoint"] == "best_model_checkpoint.pt"
            and receipt["validation_predictions"]["path"] == "val_predictions.csv"
            and receipt["validation_predictions"]["n_rows"] == len(original), "Receipt artifact/count differs")
    require(all(int(row["selected_epoch"]) == receipt["selected_epoch"]
                and row["metal_label_scheme"] == cfg["metal_label_scheme"]
                and row["binding_residue_pooling"] == config.readout for row in original),
            "Prediction epoch/target/readout differs from the fit")
    validate_prediction_rows(original, membership, 0, native_labels=labels, seed=42,
                             checkpoint_sha256=receipt["selected_checkpoint_sha256"])
    margins = validate_probabilities(original, labels, policy)
    metrics = prediction_metrics(original, native_labels=labels)
    for view, prefix in (("native", "val_metal_"), ("common4", "val_metal_collapsed4_")):
        for key, field in (("balanced_accuracy", "balanced_acc"), ("macro_f1", "macro_f1")):
            require(same_metrics(metrics[view][key], receipt["metrics"][prefix + field]), "Selected receipt metric mismatch")
        require(metrics[view]["confusion_matrix"] == chosen[prefix + "confusion_matrix"], "Selected confusion matrix mismatch")
    attempts = []
    directories = replay_directories(run)
    if config.config_id == "only_gvp__six_class__none":
        require(len({sha(directory / "val_predictions.csv") for directory in directories})
                >= policy["gvp_six_class_minimum_preserved_attempts"], "Missing distinct preserved failed GVP replays")
    for directory in directories:
        replay_receipt = read_json(directory / "selected_checkpoint.json")
        replay_csv = directory / "val_predictions.csv"
        require(replay_receipt["validation_predictions"]["path"] == replay_csv.name
                and replay_receipt["validation_predictions"]["n_rows"] == len(original), "Replay artifact/count differs")
        checked_file(replay_csv, replay_receipt["validation_predictions"]["sha256"])
        for key in ("selected_checkpoint_sha256", "selected_epoch", "campaign_run_identity", "selection_metric",
                    "selection_metric_value", "completed_epochs", "planned_epochs", "fit_status", "reconciliation_status"):
            require(replay_receipt[key] == receipt[key], f"Replay receipt differs: {directory}/{key}")
        require(same_metrics(replay_receipt["metrics"], receipt["metrics"]), "Replay metrics differ")
        rows = read_rows(replay_csv)
        validate_prediction_rows(rows, membership, 0, native_labels=labels, seed=42,
                                 checkpoint_sha256=receipt["selected_checkpoint_sha256"])
        comparison = compare_rows(original, rows, labels, policy)
        require(prediction_metrics(rows, native_labels=labels) == metrics, "Replayed discrete metrics differ")
        attempts.append({"directory": str(directory), **comparison})
        for name in ("val_predictions.csv", "selected_checkpoint.json", "replay_receipt.json"):
            path = directory / name
            if path.is_file():
                bound[str(path)] = sha(path)
    legacy = completed_run_receipt(run, identity) is not None
    verify_legacy_consistency(legacy, attempts[0])
    return {"run_name": run.name, "qualified": all(item["qualified"] for item in attempts),
            "legacy_replay_pass": legacy, "selected_epoch": receipt["selected_epoch"],
            "checkpoint_sha256": receipt["selected_checkpoint_sha256"], "identity": identity,
            "metrics": metrics, "minimum_prediction_margins": margins, "attempts": attempts,
            "input_files": bound}


def new_output(path, protected):
    path = Path(path).resolve()
    for original in map(lambda item: Path(item).resolve(), protected):
        require(path != original and path not in original.parents and original not in path.parents,
                f"Output overlaps protected input: {original}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--diagnostic-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    from benchmarking.pmm_ion_campaign import CampaignPaths, forbidden_read_roots, grid_configs
    from benchmarking.pmm_comparator import read_campaign_contract
    from training.access_guard import install_forbidden_read_guard

    install_forbidden_read_guard(forbidden_read_roots(args.train_dir))
    policy = load_policy()
    summary, diagnostic_files = diagnostic_gate(args.diagnostic_summary, policy)
    paths = CampaignPaths(args.campaign_dir.resolve())
    _, membership, _ = read_campaign_contract(paths.root)
    configs = {config.config_id: config for config in grid_configs()}
    require(set(configs) == set(policy["required_configurations"]), "Planned configurations differ from the fixed policy")
    output = new_output(args.out_dir, (paths.runs, args.train_dir, args.diagnostic_summary.parent, POLICY_PATH, __file__))
    report = {"policy_id": policy["policy_id"], "policy_sha256": POLICY_SHA256, "consumer_sha256": sha(__file__),
              "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "diagnostic_files": diagnostic_files,
              "held_out_access": False, "promotion": False, "legacy_runner_gate_changed": False,
              "scope": policy["scope"], "limitations": policy["limitations"], "arms": {}}
    try:
        for config_id in policy["required_configurations"]:
            print(f"Checking {config_id}", flush=True)
            report["arms"][config_id] = inspect_unit(paths, args.train_dir, configs[config_id], membership, policy)
        require(report["arms"]["only_gvp__six_class__none"]["checkpoint_sha256"] == summary["checkpoint_sha256"],
                "Diagnostic checkpoint differs from screened GVP checkpoint")
        report["agreement_qualified_configurations"] = sum(item["qualified"] for item in report["arms"].values())
        report["legacy_replay_pass_configurations"] = sum(item["legacy_replay_pass"] for item in report["arms"].values())
        report["status"] = "agreement_qualified" if report["agreement_qualified_configurations"] == 9 else "agreement_not_qualified"
    except Exception as exc:
        report.update(status="agreement_check_failed", error=f"{type(exc).__name__}: {exc}")
        (output / "screen_agreement_v2_1.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        raise
    # Recheck every bound file immediately before writing the append-only result.
    for item in diagnostic_files:
        checked_file(item["path"], item["sha256"])
    for item in report["arms"].values():
        for path, digest in item["input_files"].items():
            checked_file(path, digest)
    checked_file(POLICY_PATH, POLICY_SHA256)
    lines = ["# PMM fold-0 retrospective screen agreement v2.1", "",
             f"V2.1 agreement: {report['agreement_qualified_configurations']}/9. Legacy replay: {report['legacy_replay_pass_configurations']}/9.",
             "", policy["scope"], "", "Four/six in configuration names describes training; the first metric columns evaluate four classes.", "",
             "| Configuration | Epoch | Common-four BA | Common-four macro-F1 | Native BA | Fe/Co/Ni recall | V2.1 agreement |",
             "|---|---:|---:|---:|---:|---|---|"]
    for name, item in report["arms"].items():
        common, native = item["metrics"]["common4"], item["metrics"]["native"]
        recalls = "/".join(f"{100*native['recall'][label]:.3f}%" for label in ("Fe", "Co", "Ni")) if "Fe" in native["recall"] else "—"
        lines.append(f"| {name} | {item['selected_epoch']} | {100*common['balanced_accuracy']:.3f}% | {100*common['macro_f1']:.3f}% | {100*native['balanced_accuracy']:.3f}% | {recalls} | {item['qualified']} |")
    lines += ["", "No legacy receipt changed. TECH-023's legacy gate remains open; do not resume that arm through the frozen runner.",
              "", "Single-fold, single-seed evidence; no confidence intervals or promotion. See JSON for every attempt, worst differences and provenance."]
    (output / "screen_agreement_v2_1.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "screen_agreement_v2_1.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": report["status"], "output": str(output)}))
    return 0 if report["status"] == "agreement_qualified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
