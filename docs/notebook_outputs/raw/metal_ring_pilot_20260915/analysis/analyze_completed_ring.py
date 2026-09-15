"""Describe the frozen RING results only after verified allocation-3 closeout.

Usage: python analyze_completed_ring.py --campaign-dir <local campaign>
Writes ring_post_stop_analysis.json and ring_post_stop_analysis.csv under
campaign/analysis. Uses the standard library; no training, inference or network.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import tarfile


PROFILE = "metal_ring_pilot_v1"
METRIC = "val_metal_balanced_acc"
CLASSES = ("Mn", "Cu", "Zn", "Class VIII")
FAMILIES = {"GVP": "Only-GVP", "LATE": "GVP + late fusion"}
LRS = (3e-5, 1e-4)
SEEDS = (42, 43)
PRIOR_SECONDS = 19263.44616508484


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def number(value):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), "Expected a finite numeric value")
    return float(value)


def same_number(left, right):
    return math.isclose(number(left), number(right), rel_tol=0, abs_tol=1e-10)


def load_evidence(campaign):
    """Require closed ownership, archive receipts and bound terminal evidence."""
    inputs = {}

    def read(relative):
        path = campaign / relative
        require(path.resolve().is_relative_to(campaign), "Input escapes campaign")
        content = path.read_bytes()
        inputs[relative] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
        return json.loads(content)

    close_root = "host_closeout_allocation3/"
    close = read(close_root + "post_stop_receipt.json")
    require(close["status"] == "stopped_verified_and_final_archive_preserved",
            "A verified post-stop closeout is required")
    for name, expected in close["files"].items():
        path = campaign / close_root / name
        require(path.resolve().is_relative_to(campaign / close_root)
                and sha(path) == expected, "Closeout artifact differs: " + name)
    stopped = read(close_root + "session_stopped.json")
    owned = read(close_root + "owned_session.json")
    config = read(close_root + "session_config.json")
    handoff = read(close_root + "budget_handoff.json")
    ledger = read(close_root + "allocation_ledger.json")
    require(config["profile"] == PROFILE and stopped["status"] == "stopped_verified",
            "Wrong profile or unverified stop")
    require(stopped["session_name"] == owned["session_name"] == config["session_name"]
            and stopped["endpoint"] == owned["endpoint"]
            and stopped["started_epoch"] == config["allocation_started_epoch"],
            "Stop receipt does not match this owned allocation")
    require(stopped["sessions_output"] and stopped["session_name"] not in stopped["sessions_output"]
            and stopped["endpoint"] not in stopped["sessions_output"], "Owned session absence is unverified")
    intervals = ledger["intervals"]
    require(len(intervals) == 3 and intervals[:2] == handoff["prior_intervals"],
            "Expected the two original closed allocations plus allocation 3")
    require(all(number(row["ended_epoch"]) >= number(row["started_epoch"]) for row in intervals)
            and all(intervals[i]["ended_epoch"] <= intervals[i + 1]["started_epoch"] for i in (0, 1)),
            "Invalid closed allocation intervals")
    prior = sum(row["ended_epoch"] - row["started_epoch"] for row in intervals[:2])
    total = sum(row["ended_epoch"] - row["started_epoch"] for row in intervals)
    require(same_number(prior, PRIOR_SECONDS) and same_number(prior, handoff["prior_allocated_seconds"])
            and handoff["total_cap_seconds"] == 36000 and handoff["training_cap_seconds"] == 34200
            and handoff["main_cap_seconds"] == 27000, "Original allocation budget differs")
    require(intervals[-1]["started_epoch"] == stopped["started_epoch"]
            and intervals[-1]["ended_epoch"] == stopped["stopped_epoch"]
            and same_number(total, stopped["cumulative_allocated_seconds"])
            and same_number(total, close["cumulative_allocated_seconds"])
            and same_number(36000 - total, close["remaining_seconds"]), "Post-stop accounting differs")

    receipt = read("finalization/transfer_receipts/finalization.json")
    require(receipt == read(close_root + "final_transfer_receipt.json"), "Final receipt differs from closeout")
    require(receipt["attempt_id"] == "finalization" and receipt["local_sha256_verified"]
            and receipt["drive_verified"] and receipt.get("drive_file_id")
            and receipt["drive_file_id"] == close["final_drive_file_id"], "Final persistence is unverified")
    archive = Path(receipt["archive"]).resolve()
    require(archive.is_relative_to(campaign) and sha(archive) == receipt["archive_sha256"]
            == close["final_archive_sha256"] and archive.stat().st_size == receipt["bytes"],
            "Final archive checksum or byte size differs")
    inputs[str(archive.relative_to(campaign))] = {"sha256": receipt["archive_sha256"], "bytes": receipt["bytes"]}
    manifest = read("finalization/campaign_manifest.json")
    state = read("finalization/campaign_state.json")
    summary = read("finalization/ring_validation_results.json")
    capture = read("finalization/final_capture_receipt.json")
    audit = read("finalization/ring_input_audit.json")
    attempts = read("finalization/campaign_attempt_ledger.json")
    require(state["status"] in ("completed", "budget_stopped")
            and state == summary["state"] == capture["terminal_state"], "A matching terminal snapshot is required")
    require(capture["status"] == "terminal_metadata_verified_gpu_not_stopped"
            and capture == read(close_root + "final_capture_receipt.json")
            and number(capture["captured_epoch"]) <= number(stopped["stopped_epoch"]),
            "Final capture and actual stop are inconsistent")
    require(manifest["profile"] == summary["profile"] == PROFILE
            and manifest["selection_metric"] == METRIC and summary["promoted"] is False
            and manifest["held_out_evaluation"] is False
            and summary["held_out_evaluation"] is False and capture["held_out_evaluation"] is False
            and summary["confidence_intervals_computed"] is False
            and summary["independent_validation_folds"] == 1, "Wrong scientific reporting identity")
    require(capture["completed_smokes"] == summary["completed_smokes"]
            and capture["completed_full_runs"] == summary["completed_full_runs"]
            and capture["required_gvp_complete"] == summary["coverage"]["GVP"]["complete"],
            "Final capture coverage differs from the result summary")
    require(inputs["finalization/campaign_manifest.json"]["sha256"] == capture["manifest_sha256"]
            == receipt["manifest_sha256"] and capture["source_archive_sha256"] == config["source_sha256"],
            "Frozen manifest/source identity differs")
    require(audit["status"] == "passed" and audit["expected_cohort_sha256"] == manifest["cohort_sha256"]
            and audit["all_node_and_site_tensors_identical"] and audit["shared_radius_geometry_identical"],
            "The matched input audit is not certified")
    require(not any(row["status"] == "running" for row in attempts), "An attempt remains marked running")
    with tarfile.open(archive) as stream:
        for name in ("campaign_manifest.json", "campaign_state.json", "ring_validation_results.json",
                     "final_capture_receipt.json", "ring_input_audit.json", "campaign_attempt_ledger.json"):
            member = stream.getmember(name)
            require(member.isfile() and hashlib.sha256(stream.extractfile(member).read()).hexdigest()
                    == inputs["finalization/" + name]["sha256"], "Final artifact differs from archive: " + name)
    return summary, manifest, capture, audit, attempts, stopped, close, inputs, read


def metrics(row):
    recalls = row["per_class_recall"]
    require(set(recalls) == set(CLASSES), "Wrong direct-four recall labels")
    values = {"balanced_accuracy": number(row["balanced_accuracy"]),
              **{"recall_" + label: number(recalls[label]) for label in CLASSES}}
    require(all(0 <= value <= 1 for value in values.values()), "Metric outside [0, 1]")
    require(same_number(statistics.mean(recalls.values()), values["balanced_accuracy"]),
            "Balanced accuracy differs from the mean class recall")
    return values


def calculate(summary, manifest, capture, read):
    planned = {row["id"]: row for row in manifest["runs"]}
    bindings = {row["run_id"]: row for row in capture["runs"]}
    rows = summary["runs"]
    require(len({row["id"] for row in rows}) == len(rows) == summary["completed_full_runs"],
            "Completed full-run identities/count differ")
    by_key = {}
    for row in rows:
        plan, binding = planned[row["id"]], bindings[row["id"]]
        require(all(row[key] == plan[key] for key in ("block", "family", "ring", "lr", "seed", "scheme"))
                and row["scheme"] == "four_class" and plan["epochs"] == 50
                and type(row["ring"]) is bool, "Completed row differs from planned full fit")
        require(row["selected_epoch"] == binding["selected_epoch"]
                and same_number(row["balanced_accuracy"], binding["balanced_accuracy"])
                and row["normalization_stats_sha256"] == binding["normalization_stats_sha256"],
                "Completed metric differs from selected-checkpoint binding")
        relative = "finalization/completed_runs/" + row["id"] + "/run_config.json"
        saved = read(relative)
        selected = next(item for item in saved["history"] if item["epoch"] == row["selected_epoch"])
        require(selected["val_metal_per_class_recall"] == row["per_class_recall"]
                and same_number(selected[METRIC], row["balanced_accuracy"]), "Selected-history recalls differ")
        key = (row["block"], row["lr"], row["seed"], row["ring"])
        require(key not in by_key, "Duplicate family/LR/seed/RING cell")
        by_key[key] = row
        metrics(row)
    pairs, aggregates, table = [], [], []
    for block, family in FAMILIES.items():
        expected = {(block, lr, seed, ring) for lr in LRS for seed in SEEDS for ring in (False, True)}
        actual = {key for key in by_key if key[0] == block}
        coverage = summary["coverage"][block]
        require(actual <= expected and coverage["expected_runs"] == 8
                and coverage["completed_runs"] == len(actual)
                and coverage["complete"] == (actual == expected), "Family coverage disagrees with completed rows")
        if not coverage["complete"]:
            continue
        for lr in LRS:
            lr_pairs = []
            for seed in SEEDS:
                off, on = (by_key[(block, lr, seed, ring)] for ring in (False, True))
                off_values, on_values = metrics(off), metrics(on)
                deltas = {name: on_values[name] - value for name, value in off_values.items()}
                pair = dict(family=family, block=block, lr=lr, seed=seed,
                            ring_off_run_id=off["id"], ring_on_run_id=on["id"],
                            ring_off_selected_epoch=off["selected_epoch"], ring_on_selected_epoch=on["selected_epoch"],
                            normalization_identical=off["normalization_stats_sha256"] == on["normalization_stats_sha256"],
                            ring_off=off_values, ring_on=on_values, delta_on_minus_off=deltas)
                pairs.append(pair)
                lr_pairs.append(pair)
                for name, delta in deltas.items():
                    table.append(dict(row_type="paired_seed", family=family, lr=lr, seed=seed, metric=name,
                                      ring_off_value=off_values[name], ring_on_value=on_values[name],
                                      delta_on_minus_off=delta, ring_off_run_id=off["id"], ring_on_run_id=on["id"]))
            measures = {}
            for name in lr_pairs[0]["ring_off"]:
                off = [pair["ring_off"][name] for pair in lr_pairs]
                on = [pair["ring_on"][name] for pair in lr_pairs]
                delta = [pair["delta_on_minus_off"][name] for pair in lr_pairs]
                measures[name] = dict(ring_off_mean=statistics.mean(off), ring_off_sample_sd=statistics.stdev(off),
                                      ring_on_mean=statistics.mean(on), ring_on_sample_sd=statistics.stdev(on),
                                      mean_paired_delta=statistics.mean(delta), paired_delta_sample_sd=statistics.stdev(delta))
                table.append(dict(row_type="two_seed_summary", family=family, lr=lr, metric=name, **measures[name]))
            aggregates.append(dict(family=family, block=block, lr=lr, seeds=list(SEEDS), metrics=measures))
    if summary["state"]["status"] == "completed":
        require(all(summary["coverage"][block]["complete"] for block in ("S", "GVP", "LATE")),
                "Completed queue has incomplete coverage")
    return pairs, aggregates, table


def analyze(campaign):
    campaign = Path(campaign).resolve()
    summary, manifest, capture, audit, attempts, stopped, close, inputs, read = load_evidence(campaign)
    pairs, aggregates, table = calculate(summary, manifest, capture, read)
    bindings = {row["run_id"]: row for row in capture["runs"]}
    for row in summary["runs"]:
        name = "finalization/completed_runs/" + row["id"] + "/run_config.json"
        require(inputs[name]["sha256"] == bindings[row["id"]]["files"]["run_config.json"],
                "Copied run configuration differs from verified checkpoint capture")
    added = sum(part["ring_added_undirected_pairs"] for part in audit["splits"].values())
    audit_normalization_equal = audit["normalization"]["off_sha256"] == audit["normalization"]["on_sha256"]
    completed_groups = {}
    for row in summary["runs"]:
        completed_groups.setdefault((row["block"], row["lr"], row["seed"]), {})[row["ring"]] = row
    normalization_pairs = []
    unpaired_completed_runs = []
    for (block, lr, seed), group in sorted(completed_groups.items()):
        if set(group) != {False, True}:
            unpaired_completed_runs.extend(row["id"] for row in group.values())
            continue
        normalization_pairs.append(dict(block=block, lr=lr, seed=seed,
                                        complete_family=summary["coverage"][block]["complete"],
                                        ring_off_sha256=group[False]["normalization_stats_sha256"],
                                        ring_on_sha256=group[True]["normalization_stats_sha256"],
                                        identical=group[False]["normalization_stats_sha256"]
                                                  == group[True]["normalization_stats_sha256"]))
    paired_normalization_equal = (all(pair["identical"] for pair in normalization_pairs)
                                  if normalization_pairs else None)
    result = dict(profile=PROFILE, analysis_version=1, source_inputs=inputs,
                  helper_sha256=sha(Path(__file__)), terminal_state=summary["state"], coverage=summary["coverage"],
                  completed_smokes=summary["completed_smokes"], completed_full_runs=summary["completed_full_runs"],
                  completed_runs=summary["runs"], attempts=attempts, paired_runs=pairs, family_lr_summaries=aggregates,
                  metrics_unit="fraction; multiply differences by 100 for percentage points",
                  seeds=list(SEEDS), learning_rates=list(LRS), selection_metric=METRIC,
                  independent_validation_folds=1, promoted=False, winner_selected=False,
                  confidence_intervals_computed=False, inference_performed=False, held_out_evaluation=False,
                  observed_inputs=dict(splits=audit["splits"], added_undirected_pairs=added,
                                       audited_normalization_identical=audit_normalization_equal,
                                       available_completed_pair_normalization=normalization_pairs,
                                       unpaired_completed_runs=unpaired_completed_runs,
                                       all_available_completed_pairs_normalization_identical=paired_normalization_equal,
                                       raw_ring_angle_column_consumed=audit["raw_ring_angle_column_consumed"],
                                       no_topology_expansion_observed=added == 0,
                                       annotation_only_in_audit_and_available_pairs=(added == 0 and audit_normalization_equal
                                                                                    and paired_normalization_equal is True)),
                  general_recipe_estimand=summary["estimand"],
                  actual_closeout=dict(session_name=stopped["session_name"], stopped_epoch=stopped["stopped_epoch"],
                                       cumulative_allocated_seconds=close["cumulative_allocated_seconds"],
                                       remaining_seconds=close["remaining_seconds"]),
                  limitations=["Two model seeds share the same validation proteins; sample SD is not an unseen-protein CI.",
                               "Every LR is retained; only complete family blocks receive paired summaries.",
                               "Incomplete or unadmitted optional coverage is preserved, not evidence against that family.",
                               "The actual graph intervention is described by the input audit and completed-pair normalization hashes.",
                               "No Stage 6 confirmation, architecture promotion, held-out evaluation or auxiliary-learning claim."])
    fields = ("row_type", "family", "lr", "seed", "metric", "ring_off_value", "ring_on_value", "delta_on_minus_off",
              "ring_off_mean", "ring_off_sample_sd", "ring_on_mean", "ring_on_sample_sd", "mean_paired_delta",
              "paired_delta_sample_sd", "ring_off_run_id", "ring_on_run_id")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(table)
    outputs = {"ring_post_stop_analysis.json": json.dumps(result, indent=2, allow_nan=False) + "\n",
               "ring_post_stop_analysis.csv": buffer.getvalue()}
    destination = campaign / "analysis"
    for name, content in outputs.items():
        path = destination / name
        require(not path.exists() or path.read_bytes() == content.encode(),
                "Existing analysis differs; preserve it and use a separately versioned helper/output: " + name)
    for name, proof in inputs.items():
        require(sha(campaign / name) == proof["sha256"], "Input changed during analysis: " + name)
    destination.mkdir(exist_ok=True)
    for name, content in outputs.items():
        path = destination / name
        if not path.exists():
            with path.open("x") as stream:
                stream.write(content)
    return {"status": "analyzed_after_verified_stop", "paired_rows": len(pairs),
            "family_lr_summaries": len(aggregates), "outputs": [str(destination / name) for name in outputs]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", required=True, type=Path)
    print(json.dumps(analyze(parser.parse_args().campaign_dir), indent=2))
