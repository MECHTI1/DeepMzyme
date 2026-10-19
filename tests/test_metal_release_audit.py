"""Synthetic development-only evidence exercises; no repository data or models."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from metal_release.audit import COMMON4, ELEMENTS, NATIVE, audit, sequential_walk
from metal_release.manifest import ReleaseError, sha256_file


def write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return {"path": str(path), "sha256": sha256_file(path)}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"path": str(path), "sha256": sha256_file(path)}


@pytest.fixture
def release(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("release_case")
    rows = [{"source_uid": f"u{fold}-{element}-{rep}", "physical_ion_id": f"ion{fold}-{element}-{rep}",
             "group_id": f"related{fold}-{element}", "pdbid": f"pdb{fold}-{element}",
             "native_element": element, "fold": fold}
            for fold in range(5) for element in ELEMENTS for rep in range(2)]
    fold_ref = write_csv(tmp_path / "fold_membership.csv", rows)
    manifest = {"schema": "deepmzyme-metal-release-v1", "reference": "reference",
                "n_folds": 5, "fold_membership": fold_ref, "members": {}, "candidates": {}}
    groups = []
    for family, target in (("four", "four_class"), ("six", "six_class")):
        ids = []
        for fold in range(5):
            member_id = f"{family}__fold{fold}__seed42"
            ids.append(member_id)
            directory = tmp_path / member_id
            directory.mkdir()
            checkpoint = directory / "terminal_model_checkpoint.pt"
            checkpoint.write_bytes(f"synthetic checkpoint {member_id}".encode())
            checkpoint_ref = {"path": str(checkpoint), "sha256": sha256_file(checkpoint)}
            identity = {"fold": fold, "model_seed": 42, "target_scheme": target,
                        "fold_membership_sha256": fold_ref["sha256"]}
            config = {"run_test_eval": False, "test_structure_dir": None, "test_summary_csv": None,
                      "fold_index": fold, "seed": 42, "metal_label_scheme": target,
                      "metal_example_unit": "ion", "task": "metal", "fold_split_source": "membership",
                      "fold_membership_sha256": fold_ref["sha256"], "checkpoint_rule": "terminal",
                      "epochs": 50, "campaign_run_identity": json.dumps(identity)}
            config_ref = write_json(directory / "run_config.json", {"config": config})
            predictions = []
            for row in [r for r in rows if r["fold"] == fold]:
                y_native = min(ELEMENTS.index(row["native_element"]), len(NATIVE[target]) - 1)
                native = np.full(len(NATIVE[target]), 0.01)
                predicted = 0 if family == "four" and fold == 0 and y_native == 1 else y_native
                native[predicted] = 1 - 0.01 * (len(native) - 1)
                common = np.concatenate((native[:3], [native[3:].sum()]))
                item = {"source_uid": row["source_uid"], "physical_ion_id": row["physical_ion_id"],
                        "structure_id": f"structure-{row['source_uid']}", "example_id": f"example-{row['source_uid']}",
                        "group_id": row["pdbid"], "native_element": row["native_element"], "fold": fold,
                        "model_seed": 42, "metal_label_scheme": target, "selected_epoch": 50,
                        "checkpoint_sha256": checkpoint_ref["sha256"], "y_native": y_native,
                        "pred_native": int(native.argmax()), "y_common4": min(y_native, 3),
                        "pred_common4": int(common.argmax())}
                item.update({f"p_native_{name.replace(' ', '_')}": float(p) for name, p in zip(NATIVE[target], native)})
                item.update({f"p_common4_{name.replace(' ', '_')}": float(p) for name, p in zip(COMMON4, common)})
                predictions.append(item)
            prediction_ref = write_csv(directory / "val_predictions.csv", predictions)
            native_ba = 0.75 if family == "four" and fold == 0 else 1.0
            receipt = {"checkpoint_rule": "terminal", "selected_epoch": 50, "completed_epochs": 50,
                       "fit_status": "completed", "reconciliation_status": "match",
                       "selected_checkpoint": checkpoint.name, "selected_checkpoint_sha256": checkpoint_ref["sha256"],
                       "validation_predictions": {"path": "val_predictions.csv", "sha256": prediction_ref["sha256"],
                                                  "n_rows": len(predictions)}, "campaign_run_identity": identity,
                       "metrics": {"val_metal_balanced_acc": native_ba, "val_metal_collapsed4_balanced_acc": native_ba}}
            receipt_ref = write_json(directory / "selected_checkpoint.json", receipt)
            retained = {}
            for partition, selected in (("train", [r for r in rows if r["fold"] != fold]),
                                        ("validation", [r for r in rows if r["fold"] == fold])):
                examples = [{"structure_id": f"structure-{row['source_uid']}",
                             "pocket_id": f"example-{row['source_uid']}", "group": row["pdbid"]} for row in selected]
                retained[partition] = {"examples": examples, "n_examples": len(examples),
                                       "ordered_examples_sha256": hashlib.sha256(json.dumps(
                                           examples, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
            identity_text = "\n".join(f"{r['source_uid']}\t{r['physical_ion_id']}\t{r['group_id']}\t{r['fold']}"
                                      for r in sorted(rows, key=lambda r: r["source_uid"]))
            metadata_ref = write_json(directory / "dataset_summary.json", {
                "retained_split_identity": retained, "fold_membership": {
                    "verified": True, "fold_index": fold, "n_train": 48, "n_val": 12,
                    "membership_identity_sha256": hashlib.sha256(identity_text.encode()).hexdigest()}})
            manifest["members"][member_id] = {"target": target, "fold": fold, "seed": 42,
                                               "checkpoint": checkpoint_ref, "run_config": config_ref,
                                               "selected_checkpoint": receipt_ref, "validation_predictions": prediction_ref,
                                               "metadata": {"dataset_summary.json": metadata_ref}}
        groups.append({"id": family, "weight": 0.4 if family == "four" else 0.6, "members": ids})
    manifest["candidates"] = {
        "reference": {"groups": groups, "reported_oof_ba_percent": 100.0},
        "compact": {"groups": [dict(groups[0], weight=1.0)], "reported_oof_ba_percent": 95.0},
    }
    return manifest


def amend_json(member, role, change):
    path = Path(member[role]["path"])
    payload = json.loads(path.read_text())
    change(payload)
    member[role] = write_json(path, payload)


def amend_predictions(member, change):
    path = Path(member["validation_predictions"]["path"])
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    change(rows)
    member["validation_predictions"] = write_csv(path, rows)
    amend_json(member, "selected_checkpoint", lambda receipt: receipt["validation_predictions"].update(
        sha256=member["validation_predictions"]["sha256"], n_rows=len(rows)))


def test_complete_reconstructs_fold_eligible_scores_and_paired_diagnostics(release, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Audit must not discover files or use network")
    monkeypatch.setattr(Path, "glob", forbidden)
    monkeypatch.setattr(Path, "rglob", forbidden)
    import socket
    monkeypatch.setattr(socket, "create_connection", forbidden)
    report = audit(release, tmp_path / "audit", bootstrap_samples=400)
    assert report["status"] == "complete", report
    assert report["held_out_accessed"] is False
    assert report["candidates"]["compact"]["common_four"]["balanced_accuracy"] == 0.95
    assert report["candidates"]["reference"]["common_four"]["balanced_accuracy"] == 1.0
    assert report["candidates"]["compact"]["historical_score_reconciliation"]["matches_reported_rounding"]
    comparison = report["comparisons"]["compact"]
    assert comparison["fold_mean_delta"] == -0.05
    assert comparison["fold_deltas"] == [-0.25, 0.0, 0.0, 0.0, 0.0]
    assert comparison["fold_mean_delta_pp"] == -5.0
    assert comparison["prediction_disagreement_n"] == 2
    assert comparison["related_group_bootstrap"]["n_groups"] == 30
    assert comparison["selection_optimism_corrected"] is False
    assert report["member_diagnostics"]["six__fold0__seed42"]["native_metrics"]["support"]["Co"] == 2
    reliability = report["candidates"]["reference"]["reliability"]
    assert sum(row["n"] for row in reliability["reliability_bins"]) == 60
    assert reliability["coverage"][0]["coverage"] == 1.0
    assert len(reliability["coverage"][0]["by_true_class"]) == 4
    assert len(reliability["by_class"]["Cu"]["reliability_bins"]) == 10
    assert (tmp_path / "audit" / "development_oof_predictions.csv").is_file()
    assert (tmp_path / "audit" / "comparison.csv").is_file()


def test_public_summary_reports_bar_rule_and_calibration_without_paths(release, tmp_path):
    report = audit(release, tmp_path / "audit", bootstrap_samples=200)
    compact, reference = report["candidates"]["compact"], report["candidates"]["reference"]
    assert (compact["checkpoints"], compact["members_per_oof_example"]) == (5, 1)
    assert (reference["checkpoints"], reference["members_per_oof_example"]) == (10, 2)
    contrast = report["comparisons"]["compact"]
    assert contrast["pooled_delta_pp"] == pytest.approx(-5.0)
    assert contrast["pooled_recall_delta_pp"]["Cu"] == pytest.approx(-20.0)
    assert contrast["meets_similarity_bar"] is False
    # Only one fold differs, so the fold bootstrap cannot exclude zero.
    assert contrast["reference_advantage_supported"] is False
    rule = report["simplicity_rule"]
    assert rule["complexity_order"] == ["compact", "reference"]
    assert rule["walks"]["fold_interval"] == {"compact": ["compact"]}
    assert set(map(tuple, rule["stopping_points"].values())) == {("compact",)}
    assert rule["pairs"][0]["fold_interval_excludes_zero"] is False
    summary = json.loads((tmp_path / "audit" / "audit_summary.json").read_text())
    assert "manifest_path" not in summary and "member_diagnostics" not in summary
    calibration = summary["candidates"]["reference"]["calibration"]
    assert calibration["max_probability_threshold"] == 0.9
    assert set(calibration["by_true_class_at_threshold"]) == set(COMMON4)
    assert "reliability" not in summary["candidates"]["reference"]
    assert set(summary["raw_evidence"]) == {"audit.json", "comparison.csv", "development_oof_predictions.csv"}
    assert str(tmp_path) not in json.dumps(summary)
    with (tmp_path / "audit" / "comparison.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["candidate"] for row in rows] == ["compact", "reference"]
    assert rows[0]["meets_similarity_bar"] == "False" and rows[1]["meets_similarity_bar"] == ""


@pytest.mark.parametrize("mutation, expected", [
    (lambda rows: rows.append(rows[0].copy()), "exactly its excluded fold once"),
    (lambda rows: rows[0].update(fold=4), "binding differs"),
    (lambda rows: rows[0].update(y_common4=3), "labels differ"),
    (lambda rows: rows[0].update(checkpoint_sha256="0" * 64), "binding differs"),
    (lambda rows: rows[0].update(p_common4_Mn="nan"), "Invalid p_common4"),
    (lambda rows: rows[0].update(group_id="unrelated"), "example identity differs"),
])
def test_prediction_corruption_blocks_reference_and_suppresses_all_scores(release, tmp_path, mutation, expected):
    amend_predictions(release["members"]["four__fold0__seed42"], mutation)
    report = audit(release, tmp_path / "audit", bootstrap_samples=40)
    assert report["status"] == "blocked"
    assert report["comparisons"] == {}
    assert report["member_diagnostics"] == {}
    assert expected in str(report["candidates"]["reference"])
    assert not (tmp_path / "audit" / "development_oof_predictions.csv").exists()
    assert all("common_four" not in row for row in report["candidates"].values())


def test_checkpoint_hash_mismatch_is_not_recovered_by_dropping_member(release, tmp_path):
    Path(release["members"]["six__fold2__seed42"]["checkpoint"]["path"]).write_bytes(b"changed")
    report = audit(release, tmp_path / "audit", bootstrap_samples=40)
    assert report["status"] == "blocked"
    assert "SHA-256 mismatch" in str(report)
    assert report["comparisons"] == {}
    summary = (tmp_path / "audit" / "audit_summary.json").read_text()
    assert "SHA-256 mismatch: <path>" in summary
    assert str(Path(release["fold_membership"]["path"]).parent) not in summary


def test_sequential_walk_moves_only_on_supported_steps():
    order = ["a", "b", "c", "d"]
    # a -> c is supported although neither a -> b nor b -> c is: the walk must reach c, not stop at b.
    assert sequential_walk(order, {("a", "c")}, "a") == ["a", "c"]
    assert sequential_walk(order, {("a", "b"), ("a", "c"), ("b", "d")}, "a") == ["a", "b", "d"]
    assert sequential_walk(order, set(), "b") == ["b"]


def test_missing_reference_is_a_blocked_artifact_not_a_fallback_selection(release, tmp_path):
    release["members"]["six__fold1__seed42"]["checkpoint"] = None
    report = audit(release, tmp_path / "audit", bootstrap_samples=40)
    assert report["status"] == "blocked"
    assert "all numeric scores suppressed" in str(report["blockers"])
    assert report["candidates"]["compact"]["status"] == "verified"
    assert "common_four" not in report["candidates"]["compact"]


def test_fold_group_crossing_fails_before_any_checkpoint_read(release, tmp_path, monkeypatch):
    path = Path(release["fold_membership"]["path"])
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[-1]["group_id"] = rows[0]["group_id"]
    release["fold_membership"] = write_csv(path, rows)
    report = audit(release, tmp_path / "audit", bootstrap_samples=40)
    assert report["status"] == "blocked"
    assert "crosses training/validation folds" in str(report)


def test_heldout_path_is_refused_before_file_read(release, tmp_path, monkeypatch):
    member = release["members"]["six__fold1__seed42"]
    member["validation_predictions"] = {"path": str(tmp_path / "test_predictions.csv"), "sha256": "0" * 64}
    report = audit(release, tmp_path / "audit", bootstrap_samples=40)
    assert "Refusing held-out path" in str(report)
    assert report["status"] == "blocked"


@pytest.mark.parametrize("change, expected", [
    (lambda value: value["config"].update(run_test_eval=True), "not development-only"),
    (lambda value: value["config"].update(fold_membership_sha256="0" * 64), "frozen development membership"),
    (lambda value: value["config"].update(seed=44), "config fold/seed/target differs"),
])
def test_run_config_binding_is_mandatory(release, tmp_path, change, expected):
    amend_json(release["members"]["four__fold1__seed42"], "run_config", change)
    report = audit(release, tmp_path / "audit", bootstrap_samples=40)
    assert report["status"] == "blocked"
    assert expected in str(report)


def test_proxy_is_explicitly_not_deployed_refit_oof(release, tmp_path):
    member = copy.deepcopy(release["members"]["four__fold0__seed42"])
    directory = Path(member["checkpoint"]["path"]).parent.parent / "refit"
    directory.mkdir()
    config = json.loads(Path(member["run_config"]["path"]).read_text())
    config["config"].update(fold_index=None, campaign_run_identity={"phase": "stage6b_full_non_test_refit",
                                                                   "validation": "none", "model_seed": 42,
                                                                   "target_scheme": "four_class"})
    receipt = {"passed": True, "parameters_finite": True, "n_ions": 60,
               "checkpoint_sha256": member["checkpoint"]["sha256"]}
    member.update(fold=None, run_config=write_json(directory / "run_config.json", config),
                  selected_checkpoint=write_json(directory / "refit_reload_check.json", receipt))
    release["members"]["refit"] = member
    release["candidates"]["single_refit"] = {"groups": [{"id": "refit", "weight": 1.0, "members": ["refit"]}],
                                                "oof_proxy": "compact"}
    report = audit(release, tmp_path / "audit", bootstrap_samples=40)
    assert report["status"] == "complete"
    assert "no deployed full-refit OOF score" in report["candidates"]["single_refit"]["interpretation"]
    assert report["candidates"]["single_refit"]["oof_source"] == "compact"


def test_proxy_does_not_hide_missing_deployed_refit(release, tmp_path):
    release["members"]["refit"] = {"target": "four_class", "fold": None, "seed": 42}
    release["candidates"]["single_refit"] = {"groups": [{"id": "refit", "weight": 1.0, "members": ["refit"]}],
                                                "oof_proxy": "compact"}
    report = audit(release, tmp_path / "audit", bootstrap_samples=40)
    assert report["status"] == "blocked"
    assert report["candidates"]["single_refit"]["status"] == "blocked"
    assert "single_refit" not in report["comparisons"]


def test_retained_training_metadata_cannot_contain_validation_groups(release, tmp_path):
    member = release["members"]["four__fold0__seed42"]
    path = Path(member["metadata"]["dataset_summary.json"]["path"])
    value = json.loads(path.read_text())
    train = value["retained_split_identity"]["train"]
    train["examples"][0]["group"] = "pdb0-MN"
    train["ordered_examples_sha256"] = hashlib.sha256(json.dumps(
        train["examples"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    member["metadata"]["dataset_summary.json"] = write_json(path, value)
    report = audit(release, tmp_path / "audit", bootstrap_samples=40)
    assert report["status"] == "blocked"
    assert "Retained split PDB groups differ" in str(report)


def test_never_overwrites_existing_output(release, tmp_path):
    output = tmp_path / "audit"
    output.mkdir()
    with pytest.raises(ReleaseError, match="already exists"):
        audit(release, output)
