import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from report_remote_homology import (
    aggregate_ec_remoteness,
    classification_metrics,
    join_predictions,
    paired_component_bootstrap,
    validate_pairs,
    verify_replay_receipt,
    sha256,
    build_report,
)


def paired_rows():
    left, right = [], []
    for seed in (42, 43):
        for target in range(2):
            for component in range(12):
                # The same component has near and remote evaluation examples.
                for bin_name, tag in ((">30", "near"), ("<=15", "remote")):
                    row = dict(seed=seed, target=target, group_id=f"g{target}_{component}_{tag}",
                               example_id=f"x{target}_{component}_{tag}", component_id=f"c{target}_{component}",
                               bin=bin_name, status="qualified_hit", fold_id="fixed")
                    left.append(dict(row, prediction=target))
                    right.append(dict(row, prediction=target if tag == "near" else 1 - target))
    return left, right


def test_missing_class_does_not_change_full_task_metric():
    result = classification_metrics([[7, 1, 0], [0, 4, 0], [0, 0, 0]])
    assert result["balanced_accuracy"] is None
    assert result["macro_f1"] is None
    assert result["recall"] == [0.875, 1.0, None]
    assert len(result["confusion_matrix"]) == 3


def test_full_vocabulary_metrics():
    result = classification_metrics([[3, 1], [2, 2]])
    assert result["balanced_accuracy"] == 0.625
    assert result["macro_f1"] == pytest.approx((6 / 9 + 4 / 7) / 2)


def test_direct_interaction_and_seed_dependence():
    left, right = paired_rows()
    result = paired_component_bootstrap(left, right, 2, replicates=100, seed=1)
    interaction = result["contrasts"]["interaction_<=20_minus_>30"]
    assert interaction["difference"] == 1
    assert interaction["paired_confidence_interval"] == [1, 1]
    assert interaction["support_gate_passed"]
    assert result["independent_components"] == 24
    assert interaction["support"]["<=20"][0]["components"] == 12
    assert interaction["support"]["<=20"][0]["examples"] == 12


def test_identical_models_have_exact_zero_paired_interaction():
    left, _ = paired_rows()
    result = paired_component_bootstrap(left, left, 2, replicates=100)
    assert result["contrasts"]["interaction_<=20_minus_>30"]["paired_confidence_interval"] == [0, 0]


def test_missing_class_draws_and_underpowered_bins_explicit():
    left, right = paired_rows()
    left = [row for row in left if row["target"] == 0]
    right = [row for row in right if row["target"] == 0]
    result = paired_component_bootstrap(left, right, 2, replicates=30)
    interaction = result["contrasts"]["interaction_<=20_minus_>30"]
    assert interaction["difference"] is None
    assert interaction["paired_confidence_interval"] is None
    assert interaction["valid_replicates"] == 0
    assert interaction["interpretation"] == "descriptive_or_inconclusive"


@pytest.mark.parametrize("field,value", [("target", 1), ("component_id", "other"),
                                          ("fold_id", "other"), ("bin", "(20,30]")])
def test_pair_membership_and_component_mismatches_rejected(field, value):
    left, right = paired_rows()
    right[0][field] = value
    with pytest.raises(ValueError, match="mismatch"):
        validate_pairs(left, right)


def test_oof_duplicate_rejected_even_with_different_fold_id():
    left, right = paired_rows()
    left.append(dict(left[0], fold_id="another_fold"))
    with pytest.raises(ValueError, match="OOF overlap"):
        validate_pairs(left, right)


def test_unequal_seed_coverage_rejected():
    left, right = paired_rows()
    with pytest.raises(ValueError, match="Every active model seed"):
        validate_pairs(left[:-1], right[:-1])


def test_prediction_join_requires_exact_ids_targets_groups():
    remote = [dict(example_id="a", group_id="g", target=0, component_id="c")]
    prediction = [dict(example_id="a", group_id="g", target=0, prediction=1)]
    assert join_predictions(prediction, remote, 2)[0]["prediction"] == 1
    with pytest.raises(ValueError, match="memberships differ"):
        join_predictions([], remote, 2)
    with pytest.raises(ValueError, match="Duplicate"):
        join_predictions(prediction * 2, remote, 2)
    with pytest.raises(ValueError, match="Target/group mismatch"):
        join_predictions([dict(prediction[0], target=1)], remote, 2)


def test_ec_group_uses_maximum_and_does_not_hide_unknown_pocket():
    rows = [dict(example_id="a", group_id="protein", target=0, component_id="c", bin="<=15", max_identity=0.1),
            dict(example_id="b", group_id="protein", target=0, component_id="c", bin=">30", max_identity=0.5)]
    result = aggregate_ec_remoteness(rows)
    assert result[0]["example_id"] == "protein"
    assert result[0]["max_identity"] == 0.5
    assert result[0]["bin"] == ">30"
    rows[1].update(bin=None, status="no_qualifying_hit", max_identity=None)
    result = aggregate_ec_remoteness(rows)
    assert result[0]["bin"] is None
    assert result[0]["status"] == "incomplete_group_remoteness"


def replay_receipt_fixture(tmp_path):
    prediction = tmp_path / "predictions.csv"
    prediction.write_text("example_id,group_id,target,prediction\na,g,0,0\n")
    artifact = {"path": str(prediction), "sha256": sha256(prediction)}
    run = {"run_id": "selected", "task": "metal", "family": "only_gvp", "seed": 42,
           "fold_id": "fixed", "validation_only": True, "reproduction_passed": True,
           "target_vocabulary": ["Mn", "Cu", "Zn", "Class VIII"],
           "counts_freeze_sha256": "frozen-counts", "pocket_predictions": artifact}
    receipt = {key: value for key, value in run.items() if key != "pocket_predictions"}
    receipt.update(prediction_artifacts={"pocket_predictions": artifact}, training_performed=False,
                   optimizer_created=False, checkpoint_reselection=False, feature_generation=False,
                   normalization_refitted=False)
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt))
    run["receipt"] = {"path": str(path), "sha256": sha256(path)}
    return run, receipt, path, prediction


def test_reproduction_receipt_binds_actual_predictions_and_counts(tmp_path):
    run, _, _, prediction = replay_receipt_fixture(tmp_path)
    assert verify_replay_receipt(tmp_path, run, "frozen-counts") == prediction
    with pytest.raises(ValueError, match="counts freeze"):
        verify_replay_receipt(tmp_path, run, "different-counts")
    replacement = tmp_path / "other_predictions.csv"
    replacement.write_text(prediction.read_text().replace("a,g,0,0", "a,g,0,1"))
    run["pocket_predictions"] = {"path": str(replacement), "sha256": sha256(replacement)}
    with pytest.raises(ValueError, match="not bound"):
        verify_replay_receipt(tmp_path, run, "frozen-counts")


@pytest.mark.parametrize("field,value", [("reproduction_passed", False), ("training_performed", True),
                                          ("seed", 43), ("counts_freeze_sha256", "different-counts")])
def test_manifest_flags_cannot_override_receipt_contents(tmp_path, field, value):
    run, receipt, path, _ = replay_receipt_fixture(tmp_path)
    receipt[field] = value
    path.write_text(json.dumps(receipt))
    run["receipt"]["sha256"] = sha256(path)
    with pytest.raises(ValueError):
        verify_replay_receipt(tmp_path, run, "frozen-counts")


def report_fixture(root, *, task="metal"):
    """Create a complete two-family evidence chain, including frozen strata."""
    vocabulary = ["Mn", "Cu", "Zn", "Class VIII"] if task == "metal" else [str(i) for i in range(1, 8)]
    protocol = {"vocabulary": {task: vocabulary}, "primary_shorter_coverage": 0.8, "audit_shorter_coverage": 0.5,
                "statistics": {"bootstrap_replicates": 20, "bootstrap_seed": 7, "min_components_per_class": 1, "confidence_level": 0.95},
                "comparisons": [{"name": "gvp_minus_esm", "left_family": "only_gvp", "right_family": "only_esm"}]}
    protocol_path, remote_path, counts_path, prediction_path = [root / name for name in
        ("protocol.json", "remoteness.json", "counts_freeze.json", "predictions.json")]
    protocol_path.write_text(json.dumps(protocol))
    remote, export_rows = [], {}
    for family in ("only_gvp", "only_esm"):
        rows = []
        for target in range(len(vocabulary)):
            for tag in ("near", "remote_a", "remote_b"):
                group = f"protein_{target}_{tag}"
                example = group if task == "ec" else f"{group}_pocket"
                correct = tag == "near" or (tag == "remote_a" and family == "only_gvp") or (tag == "remote_b" and family == "only_esm")
                rows.append({"example_id": example, "group_id": group, "target": target,
                             "prediction": target if correct else (target + 1) % len(vocabulary)})
                # EC remoteness is pocket-level even though predictions average logits by group.
                remote.append({"task": task, "run_id": family, "example_id": f"{group}_pocket",
                               "group_id": group, "target": target, "component_id": f"component_{target}_{tag}",
                               "bin": ">30" if tag == "near" else "<=15", "status": "qualifying_hit",
                               "max_identity": .8 if tag == "near" else .1,
                               "audit50_bin": "<=15" if tag == "remote_b" else ">30",
                               "audit50_status": "qualifying_hit", "audit50_max_identity": .1 if tag == "remote_b" else .8})
        export_rows[family] = rows
    remote_path.write_text(json.dumps({"endpoint": "represented_coordinate_chain", "rows": remote}))
    counts_path.write_text(json.dumps({"protocol_sha256": sha256(protocol_path), "remoteness_manifest_sha256": sha256(remote_path),
                                      "primary_shorter_sequence_coverage": .8,
                                      "sensitivity_audit": {"shorter_sequence_coverage": .5}}))
    runs = []
    for family, rows in export_rows.items():
        folder = root / family
        folder.mkdir()
        table = folder / "predictions.csv"
        with table.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        artifact = {"path": str(table), "sha256": sha256(table)}
        artifact_name = "group_predictions" if task == "ec" else "pocket_predictions"
        run = {"run_id": family, "task": task, "family": family, "seed": 42, "fold_id": "fixed",
               "validation_only": True, "reproduction_passed": True, "target_vocabulary": vocabulary,
               "counts_freeze_sha256": sha256(counts_path), artifact_name: artifact}
        receipt = {key: value for key, value in run.items() if key != artifact_name}
        receipt.update(prediction_artifacts={artifact_name: artifact}, training_performed=False,
                       optimizer_created=False, checkpoint_reselection=False, feature_generation=False,
                       normalization_refitted=False, ec_aggregation="mean_logits" if task == "ec" else None)
        receipt_path = folder / "receipt.json"
        receipt_path.write_text(json.dumps(receipt))
        run["receipt"] = {"path": str(receipt_path), "sha256": sha256(receipt_path)}
        runs.append(run)
    prediction_path.write_text(json.dumps({"runs": runs}))
    return protocol_path, prediction_path, remote_path, counts_path


@pytest.mark.parametrize("task", ["metal", "ec"])
def test_complete_report_keeps_primary_and_sensitivity_interactions_separate(tmp_path, task):
    inputs = report_fixture(tmp_path, task=task)
    report = build_report(task, *inputs)
    key = "interaction_<=20_minus_>30"
    assert report["comparisons"]["gvp_minus_esm"]["contrasts"][key]["difference"] == 0
    audit = report["sensitivity_audit"]
    assert audit["comparisons"]["gvp_minus_esm"]["contrasts"][key]["difference"] == -1.5
    assert audit["shorter_sequence_coverage"] == .5
    assert report["primary_shorter_sequence_coverage"] == .8
    assert report["runs"][0]["bins"]["<=20"]["support"][0] == 2
    assert audit["runs"][0]["bins"]["<=20"]["support"][0] == 1


def test_end_to_end_report_rejects_postfreeze_strata_and_unbound_predictions(tmp_path):
    inputs = report_fixture(tmp_path)
    _, prediction_path, remote_path, _ = inputs
    original = remote_path.read_text()
    remote = json.loads(original)
    remote["rows"][0]["audit50_bin"] = "<=15"
    remote_path.write_text(json.dumps(remote))
    with pytest.raises(ValueError, match="Remoteness differs"):
        build_report("metal", *inputs)
    remote_path.write_text(original)
    predictions = json.loads(prediction_path.read_text())
    artifact = predictions["runs"][0]["pocket_predictions"]
    table = tmp_path / "altered.csv"
    table.write_text("example_id,group_id,target,prediction\nwrong,wrong,0,0\n")
    artifact.update(path=str(table), sha256=sha256(table))
    prediction_path.write_text(json.dumps(predictions))
    with pytest.raises(ValueError, match="not bound"):
        build_report("metal", *inputs)


def test_audit50_unknown_protein_remains_explicit_after_group_aggregation():
    from report_remote_homology import remoteness_view
    rows = [{"example_id": "a", "group_id": "protein", "target": 0, "component_id": "component",
             "bin": ">30", "status": "qualifying_hit", "max_identity": .4,
             "audit50_bin": None, "audit50_status": "incomplete_training_sequences", "audit50_max_identity": .5}]
    assert aggregate_ec_remoteness(remoteness_view(rows))[0]["bin"] == ">30"
    audit = aggregate_ec_remoteness(remoteness_view(rows, audit=True))[0]
    assert audit["bin"] is None and audit["status"] == "incomplete_group_remoteness"
    del rows[0]["audit50_status"]
    with pytest.raises(ValueError, match="Missing frozen audit50"):
        remoteness_view(rows, audit=True)
