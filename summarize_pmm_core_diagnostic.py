"""Summarize every fixed-input replay diagnostic pass without certifying a fit.

Reads trusted project tensor files on CPU. No GPU, inference, retries, training,
source edits or existing artifact rewrites. The summary is exclusively created.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
LABELS = {
    "four_class": ("Mn", "Cu", "Zn", "Class VIII"),
    "five_class": ("Mn", "Cu", "Zn", "Fe", "Class VIII"),
    "six_class": ("Mn", "Cu", "Zn", "Fe", "Co", "Ni"),
}
FOUR = LABELS["four_class"]
LOGIT_PROBABILITY_ATOL = 2e-7  # Diagnostic serialization/backend consistency only.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def preserved_references(run, campaign):
    """All known replay archive layouts, retaining duplicate files as evidence."""
    result = [run / "val_predictions.csv", run / "independent_validation_replay/val_predictions.csv"]
    for pattern in ("_incomplete_replays/*/val_predictions.csv",
                    "independent_validation_replay.incomplete.*/val_predictions.csv",
                    "independent_validation_replay_incomplete_*/val_predictions.csv"):
        result.extend(sorted(run.glob(pattern)))
    first = campaign / "runtime/fold0_completion_20260927/failed_replay_attempt1_backup/runs" / run.name
    if (first / "independent_validation_replay/val_predictions.csv").is_file():
        result.append(first / "independent_validation_replay/val_predictions.csv")
    return list(dict.fromkeys(result))


def read_reference(path, uids, labels):
    import torch

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        require(columns and len(set(columns)) == len(columns), "Duplicate reference columns")
        rows = list(reader)
    by_uid = {row["source_uid"]: row for row in rows}
    require(len(rows) == len(by_uid) == len(uids) and set(by_uid) == set(uids), "Reference UID mismatch")
    rows = [by_uid[uid] for uid in uids]
    views = {}
    expected = set()
    for view, vocabulary in (("native", labels), ("common4", FOUR)):
        keys = [f"p_{view}_{label.replace(' ', '_')}" for label in vocabulary]
        expected.update(keys)
        values = torch.tensor([[float(row[key]) for key in keys] for row in rows], dtype=torch.float64)
        require(bool(torch.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()),
                "Invalid reference probabilities")
        require(bool(torch.allclose(values.sum(1), torch.ones(len(rows), dtype=torch.float64), atol=2e-6, rtol=0)),
                "Reference probability simplex differs")
        require(values.argmax(1).tolist() == [int(row[f"pred_{view}"]) for row in rows], "Reference argmax differs")
        views[view] = values
    require({key for key in columns if key.startswith("p_")} == expected, "Wrong reference class vocabulary")
    return rows, views


def check_pass(item, *, uids, rows, labels):
    import torch

    require(item["y_native"].tolist() == [int(row["y_native"]) for row in rows], "Pass target order differs")
    for view, vocabulary in (("native", labels), ("common4", FOUR)):
        values = item[view]
        require(values.dtype == torch.float32 and tuple(values.shape) == (len(uids), len(vocabulary)),
                "Pass dtype or shape differs")
        require(bool(torch.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()),
                "Non-finite or invalid pass probability")
        require(bool(torch.allclose(values.sum(1), torch.ones(len(uids)), atol=2e-6, rtol=0)),
                "Pass probability simplex differs")
        require(torch.equal(item[f"pred_{view}"], values.argmax(1)), "Pass saved prediction differs from argmax")
    recomputed = torch.softmax(item["logits"].float(), dim=-1)
    # The audit also softmaxes on CPU, but portable runtime rounding need not be
    # bitwise. This fixed consistency allowance does not replace replay policy.
    require(bool(torch.allclose(recomputed, item["native"], atol=LOGIT_PROBABILITY_ATOL, rtol=0))
            and torch.equal(recomputed.argmax(1), item["native"].argmax(1)),
            "Pass probabilities or classes differ from its saved logits")
    collapsed = torch.cat((item["native"][:, :3], item["native"][:, 3:].sum(1, keepdim=True)), 1)
    require(bool(torch.allclose(collapsed, item["common4"], atol=2e-7, rtol=0)), "Pass common-four collapse differs")


def summarize(args):
    import torch
    import audit_pmm_replay as audit
    from audit_pmm_replay import descriptor, graph_hash

    root, campaign, prepared = args.diagnostic_dir.resolve(), args.campaign_dir.resolve(), args.prepared_dir.resolve()
    destination = root / "diagnostic_summary.json"
    require(root.is_dir() and prepared.is_relative_to(root), "Prepared inputs must be under the existing diagnostic directory")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Diagnostic summary already exists; never overwrite history")
    binding = {}

    def bind(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        require(expected is None or digest == expected, f"Artifact hash mismatch: {path}")
        binding[os.path.relpath(path, root)] = digest
        return digest

    manifest_path = prepared / "input_manifest.json"
    manifest = read(manifest_path)
    manifest_hash = bind(manifest_path)
    require(manifest["schema"] in {"pmm-replay-diagnostic-v1", "pmm-replay-diagnostic-v1.1-disabled-ec-target"}
            and manifest["status"] == "predictive_inputs_matched", "Input preparation did not pass")
    require(manifest.get("certifies_fit") is False and manifest.get("training_performed") is False
            and manifest.get("normalization_refitted") is False, "Diagnostic purpose invariants differ")
    identity = manifest["campaign_run_identity"]
    target = identity["target_scheme"]
    require(target in LABELS, "Unsupported native target scheme")
    labels = LABELS[target]
    name = f"{identity['family']}__{target}__{identity['readout']}__fold{identity['fold']}__seed{identity['model_seed']}"
    require(Path(manifest["run_dir"]).name == name and identity["family"] == "only_gvp"
            and identity["readout"] == "none" and identity["fold"] == 0, "Wrong diagnostic unit")
    run = campaign / "runs" / name
    for original_path, expected in manifest["input_files"].items():
        bind(run / Path(original_path).name, expected)
    receipt = read(run / "selected_checkpoint.json")
    checkpoint_hash = bind(run / "best_model_checkpoint.pt", receipt["selected_checkpoint_sha256"])
    require(receipt["campaign_run_identity"] == identity, "Selected checkpoint identity differs")
    for key, local in (("cohort_file", "train_cohort.csv"), ("fold_file", "fold_membership.csv")):
        bind(campaign / local, manifest[key]["sha256"])
    bind(args.audit_tool, manifest["tool_sha256"])
    require(sha(audit.__file__) == manifest["tool_sha256"], "Imported diagnostic helpers differ from the bound audit source")
    require(args.support_file and any(p.suffix == ".md" for p in args.support_file)
            and any(p.suffix == ".sh" for p in args.support_file), "Bind a written protocol and execution script")
    for path in args.support_file:
        bind(path)
    bind(__file__)
    snapshot_path = prepared / "input_snapshot.pt"
    bind(snapshot_path, manifest["snapshot_sha256"])
    snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    examples = manifest["examples"]
    uids = [row["source_uid"] for row in examples]
    require(len(set(uids)) == len(uids) == manifest["n_examples"], "Input UID count differs")
    require(snapshot["schema"] == manifest["schema"] and snapshot["ordered_uids"] == uids, "Snapshot identity differs")
    snapshot_rows = snapshot["saved_rows"]
    hashes = [graph_hash(batch) for batch in snapshot["batches"]]
    require(hashes == [b["fresh_sha256"] for b in manifest["batches"]]
            and len(hashes) == manifest["n_batches"], "Snapshot batch hashes differ")
    del snapshot
    excluded_counts = {}
    if manifest["schema"] == "pmm-replay-diagnostic-v1.1-disabled-ec-target":
        require(manifest["excluded_from_predictive_equality"] == ["y_ec"], "Recovery exclusions differ")
        previous_path = root / "failed_input_manifest.json"
        previous = read(previous_path)
        bind(previous_path, manifest["previous_failed_manifest_sha256"])
        require(previous["status"] == "input_mismatch" and previous["campaign_run_identity"] == identity
                and previous["input_files"] == manifest["input_files"], "Previous failure identity differs")
        old_sources = [p for p in [args.audit_tool, *args.support_file] if sha(p) == manifest["previous_tool_sha256"]]
        require(old_sources, "Previous diagnostic source is not bound")
        effect = manifest["disabled_ec_cpu_check"]
        require(effect["predict_ec"] is False and effect["predict_metal"] is True
                and effect["logits_metal_bitwise_equal"] is True and effect["loss_bitwise_equal"] is True,
                "Disabled EC equivalence check failed")
        for row in examples:
            require(row["fresh_fields"].keys() == row["cached_fields"].keys(), "Raw graph field inventory differs")
            for key in row["fresh_fields"]:
                if row["fresh_fields"][key] != row["cached_fields"][key]:
                    excluded_counts[key] = excluded_counts.get(key, 0) + 1
        require(set(excluded_counts) <= {"y_ec"}, "Predictive raw graph field differs")
    else:
        require(not manifest["excluded_from_predictive_equality"], "Exact input diagnosis cannot exclude fields")
        for row in examples:
            require(not row["raw_field_differences"] and row["matching_cache_files"]
                    and len(row["matching_cache_files"]) == len(row["matching_cache_sha256"]),
                    "Exact raw graph matching evidence differs")

    references = preserved_references(run, campaign) + list(args.reference)
    unique = {}
    original_rows, original_views = read_reference(references[0], uids, labels)
    require(snapshot_rows == original_rows, "Snapshot reference rows differ from preserved training export")
    for path in dict.fromkeys(references):
        require(path.resolve().is_relative_to(campaign), "Reference is outside this campaign")
        digest = bind(path)
        rows, views = read_reference(path, uids, labels)
        require([{k: v for k, v in row.items() if not k.startswith("p_")} for row in rows] ==
                [{k: v for k, v in row.items() if not k.startswith("p_")} for row in original_rows],
                "Preserved export metadata, targets or class predictions differ")
        unique.setdefault(digest, (path, views))
    passes = {"original": [], "strict": []}
    processes, model_hashes = [], []
    for condition in passes:
        for process in (1, 2):
            directory = root / f"{condition}_{process}"
            report_path = directory / "evaluation_report.json"
            report = read(report_path)
            report_hash = bind(report_path)
            require(report["schema"] == manifest["schema"] and report["input_manifest_sha256"] == manifest_hash
                    and report["tool_sha256"] == manifest["tool_sha256"], "Process input/tool identity differs")
            require(report["condition"] == condition and report["planned_repeats"] == report["completed_repeats"] == 5
                    and report["status"] == "diagnosis_complete", "All four predeclared processes must complete five passes")
            require(report["deterministic_warn_only"] is False
                    and report["deterministic_algorithms"] is (condition == "strict"), "Effective determinism differs")
            require(report["input_hashes_before_and_after"] == hashes, "Process input mutation detected")
            model_hashes.append(report["model_state_sha256_before_and_after"])
            arrays_path = directory / "forward_passes.pt"
            bind(arrays_path, report["forward_passes_sha256"])
            arrays = torch.load(arrays_path, map_location="cpu", weights_only=False)
            require(arrays["ordered_uids"] == uids and len(arrays["passes"]) == 5, "Process array UID/pass order differs")
            require(all(torch.equal(arrays["saved_probabilities"][view], values)
                        for view, values in original_views.items()), "Process reference probability arrays differ")
            saved = report["preserved_passes"]
            require([item["pass"] for item in saved] == list(range(5)), "Preserved pass indices differ")
            for index, record in enumerate(saved):
                pass_path = directory / record["path"]
                require(pass_path.resolve().parent == directory.resolve(), "Preserved pass path escaped process directory")
                bind(pass_path, record["sha256"])
                item = torch.load(pass_path, map_location="cpu", weights_only=False)
                require(descriptor(item) == descriptor(arrays["passes"][index]), "Individual/aggregate pass arrays differ")
                check_pass(item, uids=uids, rows=original_rows, labels=labels)
                passes[condition].append(item)
            processes.append(dict(condition=condition, status="passed", completed_passes=5,
                                  report_path=str(report_path.relative_to(root)), report_sha256=report_hash))
    require(len(set(model_hashes)) == 1 and len(model_hashes[0]) == 64, "Model state differs between processes")
    if "disabled_ec_cpu_check" in manifest:
        require(manifest["disabled_ec_cpu_check"]["model_state_sha256_before_and_after"] == model_hashes[0],
                "CPU equivalence check used different model state")
    comparisons = []
    for digest, (path, views) in unique.items():
        for condition, items in passes.items():
            for index, item in enumerate(items):
                for view, reference in views.items():
                    comparisons.append(dict(reference=str(path.relative_to(campaign)), reference_sha256=digest,
                                            condition=condition, pass_index=index, view=view,
                                            maximum_abs_difference=float((item[view].double() - reference).abs().max()),
                                            changed_predictions=int((item[view].argmax(1) != reference.argmax(1)).sum())))
    pairwise = {}
    for condition, items in {**passes, "all_conditions": passes["original"] + passes["strict"]}.items():
        pairwise[condition] = {}
        for view in ("native", "common4"):
            pairs = list(itertools.combinations(items, 2))
            differences = [float((a[view].double() - b[view].double()).abs().max()) for a, b in pairs]
            changes = [int((a[view].argmax(1) != b[view].argmax(1)).sum()) for a, b in pairs]
            pairwise[condition][view] = dict(maximum_abs_difference=max(differences),
                                             maximum_changed_predictions=max(changes),
                                             nonidentical_pairs=sum(x > 0 for x in differences), pairs=len(pairs))
    maximum = max(item["maximum_abs_difference"] for item in comparisons if item["condition"] == "original")
    summary = dict(schema="pmm-replay-diagnostic-summary-v1", run_name=name, target_scheme=target,
                   native_labels=list(labels), common_four_labels=list(FOUR), checkpoint_sha256=checkpoint_hash,
                   certifies_fit=False, predictive_inputs_matched=True, n_examples=len(uids), n_batches=len(hashes),
                   original_variable=any(x["maximum_abs_difference"] > 0 for x in pairwise["original"].values()),
                   all_classes_stable=all(x["changed_predictions"] == 0 for x in comparisons)
                       and all(x["maximum_changed_predictions"] == 0 for views in pairwise.values() for x in views.values()),
                   no_mutation=True, maximum_original_probability_difference=maximum,
                   original_max_abs_probability_difference=maximum,
                   completed_original_passes=10, completed_strict_passes=10, strict_failures=[], strict_results_reported=True,
                   pairwise=pairwise, comparisons_to_all_preserved_exports=comparisons,
                   input_model_and_artifact_checks_passed=True, excluded_metadata_difference_counts=excluded_counts,
                   ambiguous_identical_content_examples=manifest["ambiguous_identical_content_examples"],
                   retrospective_limit=manifest["retrospective_limit"], artifact_sha256=binding,
                   source_tree_sha256=identity["source_tree_sha256"], model_state_sha256=model_hashes[0],
                   tool_sha256=manifest["tool_sha256"], summary_tool_sha256=sha(__file__),
                   training_performed=False, held_out_access=False, processes=processes,
                   reference_files=len(dict.fromkeys(references)), distinct_reference_exports=len(unique),
                   logit_probability_consistency_atol=LOGIT_PROBABILITY_ATOL,
                   interpretation="All twenty predeclared diagnostic passes; numerical observations do not certify or promote a fit.")
    for key in ("disabled_ec_cpu_check", "previous_failed_manifest_sha256"):
        if key in manifest:
            summary[key] = manifest[key]
    summary["evidence_files"] = [dict(path=path, sha256=digest) for path, digest in sorted(binding.items())]
    # Detect concurrent source/artifact mutation before creating the immutable summary.
    for relative, expected in binding.items():
        require(sha(root / relative) == expected, f"Evidence changed during summary: {relative}")
    with destination.open("x") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--diagnostic-dir", required=True, type=Path)
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument("--prepared-dir", required=True, type=Path)
    parser.add_argument("--audit-tool", required=True, type=Path)
    parser.add_argument("--support-file", action="append", type=Path, default=[])
    parser.add_argument("--reference", action="append", type=Path, default=[])
    result = summarize(parser.parse_args())
    print(json.dumps({key: result[key] for key in ("run_name", "n_examples", "all_classes_stable",
                     "completed_original_passes", "completed_strict_passes", "maximum_original_probability_difference")}, indent=2))


if __name__ == "__main__":
    main()
