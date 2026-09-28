"""Targeted refusals for the external v2.1 consumer; synthetic inputs, no GPU."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("screen_agreement", Path(__file__).parents[1] / "audit_pmm_screen_agreement.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.fixture
def policy():
    return audit.load_policy()


def row(uid="one", probabilities=("0.6", "0.1", "0.1", "0.1", "0.05", "0.05")):
    result = {"source_uid": uid, "checkpoint_sha256": "a" * 64, "pred_native": "0", "pred_common4": "0",
              "y_native": "0", "y_common4": "0", "selected_epoch": "31"}
    result.update({"p_native_" + label: value for label, value in zip(audit.SIX, probabilities)})
    collapsed = list(probabilities[:3]) + [str(sum(audit.Decimal(value) for value in probabilities[3:]))]
    result.update({"p_common4_" + label.replace(" ", "_"): value for label, value in zip(audit.FOUR, collapsed)})
    return result


def test_absolute_boundary_and_legacy_count(policy):
    original = [row()]
    replay = [row(probabilities=("0.60001", "0.09999", "0.1", "0.1", "0.05", "0.05"))]
    result = audit.compare_rows(original, replay, audit.SIX, policy)
    assert result["qualified"] and result["above_legacy_tolerance_fields"] == 4
    assert result["worst"]["abs_difference"] == 1e-5
    replay = [row(probabilities=("0.60001001", "0.09998999", "0.1", "0.1", "0.05", "0.05"))]
    assert not audit.compare_rows(original, replay, audit.SIX, policy)["qualified"]


@pytest.mark.parametrize("mn,accepted", [("0.600001", True), ("0.600002", True), ("0.60000201", False)])
def test_simplex_uses_preexisting_fp32_allowance_without_changing_replay_bound(policy, mn, accepted):
    # Serialized FP32 softmax probabilities need not sum to exactly one.
    example = row(probabilities=(mn, "0.1", "0.1", "0.1", "0.05", "0.05"))
    assert policy["probability_atol"] == "0.00001"
    assert policy["serialized_probability_consistency_atol"] == "0.0000002"
    if accepted:
        audit.validate_probabilities([example], audit.SIX, policy)
    else:
        with pytest.raises(ValueError, match="sum to one"):
            audit.validate_probabilities([example], audit.SIX, policy)


def test_collapse_keeps_its_stricter_bound_despite_simplex_allowance(policy):
    example = row()
    example.update(p_common4_Mn="0.6000003", p_common4_Cu="0.0999997")
    with pytest.raises(ValueError, match="collapse"):
        audit.validate_probabilities([example], audit.SIX, policy)


def test_tiny_class_flip_is_rejected_despite_probability_agreement(policy):
    original = row(probabilities=("0.5000001", "0.4999999", "0", "0", "0", "0"))
    replay = row(probabilities=("0.4999999", "0.5000001", "0", "0", "0", "0"))
    replay.update(pred_native="1", pred_common4="1")
    with pytest.raises(ValueError, match="discrete/metadata"):
        audit.compare_rows([original], [replay], audit.SIX, policy)


@pytest.mark.parametrize("field,value", [("p_native_Mn", "NaN"), ("p_native_Mn", "Infinity"),
                                         ("p_native_Mn", "-0.1"), ("p_native_Mn", "1.1"),
                                         ("p_native_Cu", "0.09"), ("pred_native", "2")])
def test_bad_probabilities_and_argmax_rejected(policy, field, value):
    changed = row()
    changed[field] = value
    with pytest.raises(ValueError):
        audit.validate_probabilities([changed], audit.SIX, policy)


def test_wrong_collapse_and_vocabulary_rejected(policy):
    changed = row()
    changed.update(p_common4_Class_VIII="0.21", p_common4_Cu="0.09")
    with pytest.raises(ValueError, match="collapse"):
        audit.validate_probabilities([changed], audit.SIX, policy)
    changed = row()
    changed["p_native_ZN"] = changed.pop("p_native_Zn")
    with pytest.raises(ValueError, match="vocabulary"):
        audit.validate_probabilities([changed], audit.SIX, policy)


@pytest.mark.parametrize("changed", [[row("different")], [row(), row()], []])
def test_missing_extra_duplicate_uids_rejected(policy, changed):
    with pytest.raises(ValueError, match="UID"):
        audit.compare_rows([row()], changed, audit.SIX, policy)


def test_metadata_and_receipt_metrics_not_tolerated(policy):
    changed = row()
    changed["checkpoint_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="metadata"):
        audit.compare_rows([row()], [changed], audit.SIX, policy)
    assert not audit.same_metrics({"confusion": [[1, 2]]}, {"confusion": [[2, 1]]})
    assert not audit.same_metrics({"metric": float("nan")}, {"metric": float("nan")})


def test_legacy_receipt_does_not_override_recomputed_probability_failure():
    audit.verify_legacy_consistency(True, {"above_legacy_tolerance_fields": 0})
    audit.verify_legacy_consistency(False, {"above_legacy_tolerance_fields": 24})
    with pytest.raises(ValueError, match="Legacy receipt"):
        audit.verify_legacy_consistency(True, {"above_legacy_tolerance_fields": 24})


def test_hash_and_policy_changes_rejected(tmp_path, monkeypatch):
    path = tmp_path / "policy.json"
    path.write_bytes(audit.POLICY_PATH.read_bytes())
    audit.checked_file(path, audit.POLICY_SHA256)
    path.write_text(path.read_text() + " ")
    monkeypatch.setattr(audit, "POLICY_PATH", path)
    with pytest.raises(ValueError, match="hash mismatch"):
        audit.load_policy()


def test_output_is_append_only_and_outside_originals(tmp_path):
    original = tmp_path / "run"
    original.mkdir()
    for path in (original, original / "output", tmp_path):
        with pytest.raises(ValueError, match="overlaps"):
            audit.new_output(path, [original])
    output = audit.new_output(tmp_path / "report", [original])
    with pytest.raises(FileExistsError):
        audit.new_output(output, [original])


def diagnostic_fixture(tmp_path, policy):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    snapshot = prepared / "input_snapshot.pt"
    snapshot.write_bytes(b"synthetic immutable snapshot")
    old_tool, current_tool = tmp_path / "original_tool.py", tmp_path / "current_tool.py"
    old_tool.write_text("# preserved original diagnostic")
    current_tool.write_text("# amended diagnostic")
    manifest = {"schema": audit.DIAGNOSTIC_SCHEMA, "status": "predictive_inputs_matched", "n_examples": 1492, "n_batches": 1,
                "batches": [{"fresh_sha256": "b" * 64}],
                "campaign_run_identity": {"source_tree_sha256": policy["source_tree_sha256"]},
                "input_files": {"/original/best_model_checkpoint.pt": "c" * 64},
                "tool_sha256": audit.sha(current_tool), "snapshot_sha256": audit.sha(snapshot),
                "excluded_from_predictive_equality": ["y_ec"], "certifies_fit": False,
                "training_performed": False, "normalization_refitted": False,
                "disabled_ec_cpu_check": {"predict_ec": False, "predict_metal": True,
                    "logits_metal_bitwise_equal": True, "loss_bitwise_equal": True,
                    "model_state_sha256_before_and_after": "e" * 64}}
    previous = {"schema": "pmm-replay-diagnostic-v1", "status": "input_mismatch",
                "campaign_run_identity": manifest["campaign_run_identity"], "input_files": manifest["input_files"],
                "tool_sha256": audit.sha(old_tool)}
    previous_path = tmp_path / "failed_input_manifest.json"
    previous_path.write_text(json.dumps(previous))
    manifest.update(previous_failed_manifest_sha256=audit.sha(previous_path), previous_tool_sha256=audit.sha(old_tool))
    manifest_path = prepared / "input_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    summary = {"source_tree_sha256": policy["source_tree_sha256"],
               "run_name": "only_gvp__six_class__none__fold0__seed42", "checkpoint_sha256": "c" * 64,
               "predictive_inputs_matched": True, "original_variable": True, "all_classes_stable": True,
               "no_mutation": True, "original_max_abs_probability_difference": 3.51e-6,
               "evidence_files": [{"path": str(item.relative_to(tmp_path)), "sha256": audit.sha(item)}
                                  for item in (manifest_path, previous_path, old_tool, current_tool)],
               "processes": []}
    for condition in ("original", "strict"):
        for index in range(2):
            directory = tmp_path / f"{condition}_{index}"
            directory.mkdir()
            saved = []
            for number in range(5):
                path = directory / f"pass_{number}.pt"
                path.write_bytes(f"synthetic pass {number}".encode())
                saved.append({"path": path.name, "sha256": audit.sha(path)})
            comparison = {view: {"changed_predictions": 0, "max_abs_difference": 3.51e-6}
                          for view in ("native", "common4")}
            report = {"schema": audit.DIAGNOSTIC_SCHEMA, "condition": condition, "planned_repeats": 5, "completed_repeats": 5,
                      "status": "diagnosis_complete", "preserved_passes": saved,
                      "input_manifest_sha256": audit.sha(manifest_path), "tool_sha256": audit.sha(current_tool),
                      "deterministic_warn_only": False, "deterministic_algorithms": condition == "strict",
                      "input_hashes_before_and_after": ["b" * 64], "model_state_sha256_before_and_after": "e" * 64,
                      "comparisons_to_saved": [comparison], "pairwise_comparisons": [comparison]}
            path = directory / "evaluation_report.json"
            path.write_text(json.dumps(report))
            summary["processes"].append({"condition": condition, "status": "passed", "completed_passes": 5,
                                         "report_path": str(path.relative_to(tmp_path)), "report_sha256": audit.sha(path)})
    path = tmp_path / "diagnostic_summary.json"
    path.write_text(json.dumps(summary))
    return path, summary


def test_valid_diagnostic_gate_binds_every_pass(tmp_path, policy):
    path, _ = diagnostic_fixture(tmp_path, policy)
    summary, bound = audit.diagnostic_gate(path, policy)
    assert summary["no_mutation"] is True
    assert sum("/pass_" in item["path"] for item in bound) == 20


@pytest.mark.parametrize("key,value", [("predictive_inputs_matched", False), ("original_variable", False),
                                     ("all_classes_stable", False), ("no_mutation", False),
                                     ("source_tree_sha256", "wrong"),
                                     ("original_max_abs_probability_difference", 1.00001e-5),
                                     ("original_max_abs_probability_difference", float("nan"))])
def test_diagnostic_summary_guard_keys(tmp_path, policy, key, value):
    path, summary = diagnostic_fixture(tmp_path, policy)
    summary[key] = value
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        audit.diagnostic_gate(path, policy)


@pytest.mark.parametrize("change", ["model_mutation", "input_mutation", "pass_tampered", "missing_process", "warn_only"])
def test_diagnostic_evidence_refusals(tmp_path, policy, change):
    path, summary = diagnostic_fixture(tmp_path, policy)
    item = summary["processes"][0]
    report_path = tmp_path / item["report_path"]
    report = json.loads(report_path.read_text())
    if change == "model_mutation":
        report["model_state_sha256_before_and_after"] = "f" * 64
    elif change == "input_mutation":
        report["input_hashes_before_and_after"] = ["f" * 64]
    elif change == "warn_only":
        report["deterministic_warn_only"] = True
    elif change == "pass_tampered":
        (report_path.parent / report["preserved_passes"][0]["path"]).write_bytes(b"changed")
    else:
        summary["processes"].pop()
    report_path.write_text(json.dumps(report))
    item["report_sha256"] = audit.sha(report_path)
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        audit.diagnostic_gate(path, policy)


@pytest.mark.parametrize("error,accepted", [("operation has no deterministic implementation", True),
                                           ("CUDA out of memory", False)])
def test_only_explicit_unsupported_strict_processes_can_replace_passes(tmp_path, policy, error, accepted):
    path, summary = diagnostic_fixture(tmp_path, policy)
    for item in summary["processes"]:
        if item["condition"] != "strict":
            continue
        report_path = tmp_path / item["report_path"]
        report = json.loads(report_path.read_text())
        report.update(status="evaluation_failed", error=error, completed_repeats=0, preserved_passes=[])
        report_path.write_text(json.dumps(report))
        item.update(status="unsupported", completed_passes=0, report_sha256=audit.sha(report_path))
    path.write_text(json.dumps(summary))
    if accepted:
        audit.diagnostic_gate(path, policy)
    else:
        with pytest.raises(ValueError, match="unsupported strict"):
            audit.diagnostic_gate(path, policy)


def test_replay_discovery_requires_archived_evidence(tmp_path):
    current = tmp_path / "independent_validation_replay"
    current.mkdir()
    for name in ("selected_checkpoint.json", "val_predictions.csv"):
        (current / name).write_text("preserved")
    assert audit.replay_directories(tmp_path) == [current]
    (tmp_path / "_incomplete_replays/attempt_1").mkdir(parents=True)
    with pytest.raises(ValueError, match="lacks"):
        audit.replay_directories(tmp_path)


@pytest.mark.parametrize("change", ["missing_schema", "unknown_schema", "exclude_metal", "missing_cpu_check",
                                   "active_ec", "inactive_metal", "cpu_logits_differ", "cpu_loss_differs",
                                   "cpu_model_differs", "missing_failed_link", "wrong_failed_source",
                                   "wrong_failed_inputs", "wrong_previous_tool", "missing_previous_tool",
                                   "missing_current_tool", "unknown_report_schema"])
def test_diagnostic_amendment_semantics_fail_closed(tmp_path, policy, change):
    path, summary = diagnostic_fixture(tmp_path, policy)
    manifest_path = tmp_path / "prepared/input_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    previous_path = tmp_path / "failed_input_manifest.json"
    previous = json.loads(previous_path.read_text())
    if change == "missing_schema":
        manifest.pop("schema")
    elif change == "unknown_schema":
        manifest["schema"] = "unreviewed-future-schema"
    elif change == "exclude_metal":
        manifest["excluded_from_predictive_equality"] = ["y_ec", "y_metal"]
    elif change == "missing_cpu_check":
        manifest.pop("disabled_ec_cpu_check")
    elif change in {"active_ec", "inactive_metal", "cpu_logits_differ", "cpu_loss_differs", "cpu_model_differs"}:
        field, value = {"active_ec": ("predict_ec", True), "inactive_metal": ("predict_metal", False),
                        "cpu_logits_differ": ("logits_metal_bitwise_equal", False),
                        "cpu_loss_differs": ("loss_bitwise_equal", False),
                        "cpu_model_differs": ("model_state_sha256_before_and_after", "f" * 64)}[change]
        manifest["disabled_ec_cpu_check"][field] = value
    elif change == "missing_failed_link":
        manifest.pop("previous_failed_manifest_sha256")
    elif change == "wrong_failed_source":
        previous["campaign_run_identity"] = {"source_tree_sha256": "wrong"}
    elif change == "wrong_failed_inputs":
        previous["input_files"] = {"/original/best_model_checkpoint.pt": "wrong"}
    elif change == "wrong_previous_tool":
        manifest["previous_tool_sha256"] = "f" * 64
    elif change in {"missing_previous_tool", "missing_current_tool"}:
        filename = "original_tool.py" if change == "missing_previous_tool" else "current_tool.py"
        summary["evidence_files"] = [item for item in summary["evidence_files"] if item["path"] != filename]
    previous_path.write_text(json.dumps(previous))
    if change != "missing_failed_link":
        manifest["previous_failed_manifest_sha256"] = audit.sha(previous_path)
    manifest_path.write_text(json.dumps(manifest))
    # Rebind hashes deliberately, so refusal proves semantic checks rather than
    # merely stale checksums detecting the changed fixture.
    for item in summary["evidence_files"]:
        item["sha256"] = audit.sha(tmp_path / item["path"])
    for item in summary["processes"]:
        report_path = tmp_path / item["report_path"]
        report = json.loads(report_path.read_text())
        report["input_manifest_sha256"] = audit.sha(manifest_path)
        if change == "unknown_report_schema":
            report["schema"] = "unreviewed-future-schema"
        report_path.write_text(json.dumps(report))
        item["report_sha256"] = audit.sha(report_path)
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        audit.diagnostic_gate(path, policy)
