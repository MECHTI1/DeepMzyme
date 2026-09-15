"""Pilot orchestration tests using synthetic outputs; never allocate a GPU."""
import json
from pathlib import Path
import signal
import sys
import time
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import run_metal_architecture_pilot as pilot


@pytest.fixture(scope="module")
def planned(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    temporary = tmp_path_factory.mktemp("pilot_plan")
    data, output = temporary / "data", temporary / "output"
    train = data / pilot.DATASET / "train"
    train.mkdir(parents=True)
    (train / "final_data_summarazing_table_transition_metals_only_catalytic.csv").write_text(
        "structure,chain_resi,metaltype,ecnumber\n")
    (train / "structure_manifest.csv").write_text("logical_filename,sha256\n")
    (data / "esm_embeddings").mkdir()
    (data / "esm_embeddings/fixture_esmc.pt").touch()
    manifest = pilot.plan(root, data, output, "test_commit")
    return root, data, output, manifest


def test_plan_uses_canonical_commands_native_metrics_and_protected_test(planned):
    root, data, output, manifest = planned
    assert manifest["profile"] == pilot.PROFILE
    assert manifest["block_order"] == ["S", "A1", "A2", "T1", "T2", "H", "R", "RE", "RH"]
    assert len(manifest["runs"]) == 51  # 7 smokes, 22 first-seed, 22 repeat templates.
    assert len({r["id"] for r in manifest["runs"]}) == 51
    assert len([r for r in manifest["runs"] if r["block"] == "S"]) == 7
    assert manifest["maximum_main_runs"] == 33
    for row in manifest["runs"]:
        config = pilot.parse_config(row["command"])
        pilot.validate(config, row["epochs"])
        assert config.structure_dir == data / pilot.DATASET / "train"
        assert config.selection_metric == pilot.METRIC
        assert row["env"]["DEEPGM_METAL_LABEL_SCHEME"] == row["scheme"]
        assert not any(x in row["command"] for x in ("--run-test-eval", "--test-structure-dir", "--test-summary-csv"))
        assert str(output / "runs" / row["id"]) == row["run_dir"]
    assert pilot.plan(root, data, output, "test_commit") == manifest
    with pytest.raises(ValueError, match="manifest differs"):
        pilot.plan(root, data, output, "different_commit")


def test_dataset_mutation_is_rejected(planned):
    root, data, output, manifest = planned
    summary = data / pilot.DATASET / "train/structure_manifest.csv"
    original = summary.read_text()
    try:
        summary.write_text(original + "changed\n")
        with pytest.raises(ValueError, match="Dataset changed"):
            pilot.verify_manifest(root, output)
    finally:
        summary.write_text(original)


def test_headless_notebook_expansion_inside_colab_does_not_mount_drive(planned, tmp_path, monkeypatch):
    root, data, _, _ = planned
    original_exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda path: True if str(path) == "/content" else original_exists(path))
    colab = ModuleType("google.colab")

    def unexpected_mount(*args, **kwargs):
        raise AssertionError("Headless planning must not mount Google Drive")

    colab.drive = SimpleNamespace(mount=unexpected_mount)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    templates = pilot._templates(root, data, tmp_path / "headless_output")
    assert set(templates) == set(pilot.FAMILIES)
    for row in templates.values():
        command = list(map(str, row["command"]))
        assert command[command.index("--runs-dir") + 1].startswith(str(tmp_path / "headless_output"))
        assert not any("/content/drive" in value for value in command)


def make_result(row, score=.7, recall=.4):
    return {**row, "balanced_accuracy": score, "minimum_recall": recall}


def test_required_target_blocks_precede_hybrid_and_repeat_lr_selection(planned):
    manifest = planned[-1]
    rows = [make_result(r) for r in manifest["runs"] if r["block"] in ("S", "A1", "A2")]
    for row in rows:
        if row["family"] == pilot.EARLY:
            row["balanced_accuracy"] = .72
    assert pilot.hybrid_gate(rows)["status"] == "prioritized"
    assert pilot.next_block(manifest, rows)[0] == "T1"
    rows.extend(make_result(r) for r in manifest["runs"] if r["block"] in ("T1", "T2"))
    assert pilot.next_block(manifest, rows)[0] == "H"
    rows.extend(make_result(r) for r in manifest["runs"] if r["block"] == "H")
    block, active, pending = pilot.next_block(manifest, rows)
    assert block == "R" and len(active) == len(pending) == 9
    assert all(r["lr"] == 3e-5 and r["seed"] == 43 for r in active)  # Deterministic tied-LR selection.
    better = next(r for r in rows if r["block"] == "T2" and r["family"] == pilot.CORE[0] and r["scheme"] == "six_class")
    better["balanced_accuracy"] = .8
    repeat = pilot.active_block_runs(manifest, "R", rows)
    assert next(r for r in repeat if r["family"] == pilot.CORE[0] and r["scheme"] == "six_class")["lr"] == 1e-4


def test_hybrid_gate_defers_weak_or_rare_class_harming_early(planned):
    rows = [make_result(r) for r in planned[-1]["runs"] if r["block"] in ("A1", "A2")]
    assert pilot.hybrid_gate(rows)["status"] == "deferred"
    for row in rows:
        if row["family"] == pilot.EARLY:
            row.update(balanced_accuracy=.75, minimum_recall=.36)
    assert pilot.hybrid_gate(rows)["status"] == "deferred"
    assert pilot.active_block_runs(planned[-1], "H", rows) == []
    assert pilot.hybrid_gate(rows[:1])["status"] == "pending"


def test_allocation_time_counts_setup_transfer_and_multiple_intervals(tmp_path):
    assert pilot.allocation_elapsed(tmp_path, 1000, now=1060) == 60
    assert pilot.allocation_elapsed(tmp_path, 1000, now=1600) == 600
    with pytest.raises(ValueError, match="Previous allocation"):
        pilot.allocation_elapsed(tmp_path, 1800, now=1900)
    pilot.close_allocation(tmp_path, ended_epoch=1700)
    assert pilot.allocation_elapsed(tmp_path, 1800, now=1900) == 800
    assert pilot.budget_available(569 * 60, [], "A1") == 60
    assert pilot.budget_available(570 * 60, [], "A1") == 0


def test_failures_and_retries_count_against_correct_allowances():
    attempts = [dict(block="P", elapsed_seconds=600), dict(block="S", elapsed_seconds=300),
                dict(block="A1", elapsed_seconds=1200),
                dict(block="A1", elapsed_seconds=1000, retry_of="previous")]
    assert pilot.budget_available(3100, attempts, "S") == 500  # Setup/transfer time also counts.
    assert pilot.budget_available(3100, attempts, "A1") == 450 * 60 - 1200
    assert pilot.budget_available(570 * 60, attempts, "S") == 0


def test_forecast_extrapolates_epochs_preserving_setup(planned):
    manifest = planned[-1]
    smoke = next(r for r in manifest["runs"] if r["block"] == "S" and r["family"] == pilot.CORE[0])
    main = next(r for r in manifest["runs"] if r["block"] == "A1" and r["family"] == pilot.CORE[0])
    attempts = [dict(run_id=smoke["id"], status="completed", setup_seconds=100, epoch_seconds=5)]
    assert pilot.forecast_seconds(main, attempts, manifest) == 350
    assert 4 * pilot.forecast_seconds(main, attempts, manifest) * pilot.LIMITS["admission_factor"] == 1750


def reference_summary():
    return {"retained_split_identity": {
        "train": {"examples": [{"structure_id": "trainA", "pocket_id": "siteA", "group": "train", "y_metal": 0}]},
        "validation": {"examples": [{"structure_id": "valB", "pocket_id": "siteB", "group": "val", "y_metal": 3}]}}}


def completed_fixture(tmp_path, epochs=50):
    run = dict(id="test", block="T1", family="Only-GVP", scheme="six_class", lr=3e-5, seed=42, epochs=epochs)
    summary = reference_summary()
    pilot.save(tmp_path / "dataset_summary.json", summary)
    pilot.save(tmp_path / "run_metadata.json", dict(selected_checkpoint_epoch=2, test_report=None))
    history = [dict(epoch=i, val_metal_balanced_acc=.8 if i == 2 else .7,
                    val_metal_collapsed4_balanced_acc=.6 if i == 2 else .9,
                    val_metal_per_class_recall={"Fe": .8, "Ni": .2},
                    val_metal_collapsed4_per_class_recall={"Mn": .9, "VIII": .3}) for i in range(1, epochs + 1)]
    pilot.save(tmp_path / "run_config.json", {"history": history})
    (tmp_path / "epoch_metrics.csv").write_text("epoch\n" + "\n".join(str(i) for i in range(1, epochs + 1)))
    for name in ("best_model_checkpoint.pt", "last_model_checkpoint.pt"):
        (tmp_path / name).write_bytes(b"fixture")
    return run, summary, history


def test_native_checkpoint_and_same_checkpoint_collapsed_reporting(tmp_path):
    run, reference, history = completed_fixture(tmp_path)
    row = pilot.completed_result(run, tmp_path, reference)
    assert row["balanced_accuracy"] == .8
    assert row["collapsed4_balanced_accuracy"] == .6  # Do not reselect epoch by collapsed .9.
    history[-1][pilot.METRIC] = .85
    pilot.save(tmp_path / "run_config.json", {"history": history})
    with pytest.raises(ValueError, match="maximum native"):
        pilot.completed_result(run, tmp_path, reference)


def test_cohort_comparison_ignores_intended_target_change_but_rejects_membership_change(tmp_path):
    run, reference, _ = completed_fixture(tmp_path)
    changed = json.loads(json.dumps(reference))
    changed["retained_split_identity"]["validation"]["examples"][0]["y_metal"] = 5
    pilot.save(tmp_path / "dataset_summary.json", changed)
    pilot.completed_result(run, tmp_path, reference)
    changed["retained_split_identity"]["validation"]["examples"][0]["pocket_id"] = "different"
    pilot.save(tmp_path / "dataset_summary.json", changed)
    with pytest.raises(ValueError, match="cohort mismatch"):
        pilot.completed_result(run, tmp_path, reference)


def test_partial_epochs_and_heldout_outputs_cannot_be_reused(tmp_path):
    run, reference, _ = completed_fixture(tmp_path, epochs=2)
    run["epochs"] = 50
    with pytest.raises(ValueError, match="Incomplete or duplicated"):
        pilot.completed_result(run, tmp_path, reference)
    run["epochs"] = 2
    pilot.save(tmp_path / "test_report.json", {"score": .9})
    with pytest.raises(ValueError, match="Held-out"):
        pilot.completed_result(run, tmp_path, reference)


def test_transfer_receipt_tied_to_attempt_and_manifest(tmp_path):
    pilot.save(tmp_path / "campaign_manifest.json", {"profile": pilot.PROFILE})
    attempts = [{"attempt_id": "attempt_001"}]
    storage = {"method": "verified_archive_transfer"}
    with pytest.raises(ValueError, match="archive attempt"):
        pilot._require_transfer(tmp_path, attempts, storage)
    pilot.save(tmp_path / "transfer_receipts/attempt_001.json", dict(attempt_id="attempt_001", drive_verified=True,
        local_sha256_verified=True, archive_sha256="abc", manifest_sha256=pilot.digest(tmp_path / "campaign_manifest.json")))
    pilot._require_transfer(tmp_path, attempts, storage)


def test_changed_feature_cache_cannot_reuse_readiness(tmp_path):
    feature = tmp_path / "external.json"
    feature.write_text("original")
    pilot.save(tmp_path / "training_cache_audit.json", {"failures": [], "files": [
        {"path": str(feature), "sha256": pilot.digest(feature), "bytes": feature.stat().st_size,
         "mtime_ns": feature.stat().st_mtime_ns}]})
    pilot.verify_training_cache(tmp_path)
    feature.write_text("changed payload")
    with pytest.raises(ValueError, match="changed after"):
        pilot.verify_training_cache(tmp_path)


def test_overlay_requires_completed_manifest_and_checks_portable_identity_hashes(tmp_path, monkeypatch):
    import structure_store
    reference = SimpleNamespace(structure_name="fixture__chain_A.pdb", sha256="a" * 64)
    monkeypatch.setattr(structure_store, "read_structure_manifest", lambda path: [reference])
    overlay = tmp_path / "remote_overlay"
    feature = overlay / "fixture__chain_A/residue_features.json"
    pilot.save(feature, {"tooling": {"pka": "propka"}})
    report = dict(status="complete", failures=[], dataset=pilot.DATASET, split="train", test_evaluation=False,
                  total_structures=1, selected_structures=1,
                  files=[dict(structure=reference.structure_name, structure_sha256=reference.sha256,
                              path="/host/local/path/not/present/remotely", sha256=pilot.digest(feature))])
    manifest_path = overlay / "feature_overlay_manifest.json"
    pilot.save(manifest_path, report)
    assert pilot.validate_feature_overlay(tmp_path, overlay, manifest_path) == report
    report["status"] = "partial"
    pilot.save(manifest_path, report)
    with pytest.raises(ValueError, match="not complete"):
        pilot.validate_feature_overlay(tmp_path, overlay, manifest_path)
    report["status"] = "complete"
    report["files"][0]["sha256"] = "b" * 64
    pilot.save(manifest_path, report)
    with pytest.raises(ValueError, match="feature checksum"):
        pilot.validate_feature_overlay(tmp_path, overlay, manifest_path)
    report["files"] = []
    pilot.save(manifest_path, report)
    with pytest.raises(ValueError, match="complete training cohort"):
        pilot.validate_feature_overlay(tmp_path, overlay, manifest_path)


def test_deadline_sends_termination_without_waiting_for_training(tmp_path, monkeypatch):
    calls = []

    class Process:
        pid = 1234
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.returncode = -signal.SIGTERM

    monkeypatch.setattr(pilot.subprocess, "Popen", lambda *a, **kw: Process())
    monkeypatch.setattr(pilot.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    attempt = {"started_epoch": time.time(), "log": str(tmp_path / "child.log"), "status": "running"}
    pilot._run_process(["fixture"], tmp_path, tmp_path, attempt, {}, time.time() + 5)
    assert calls == [(1234, signal.SIGTERM)]
    assert attempt["status"] == "deadline_stopped" and attempt["elapsed_seconds"] >= 0


def test_recovery_charges_interrupted_attempt_and_links_single_retry(tmp_path, monkeypatch):
    attempts = [dict(attempt_id="attempt_001", run_id="test", block="S", status="running", started_epoch=time.time() - 50)]
    pilot._recover_interrupted(tmp_path, attempts)
    assert attempts[0]["status"] == "interrupted" and attempts[0]["elapsed_seconds"] >= 50
    run = dict(id="test", block="P", run_dir=str(tmp_path / "preflight"))

    def fake_process(command, root, output, attempt, env, deadline):
        attempt.update(returncode=0, ended_epoch=time.time(), elapsed_seconds=1)
        pilot.save(output / "readiness.json", {"status": "passed"})

    monkeypatch.setattr(pilot, "_run_process", fake_process)
    result = pilot._record_attempt(tmp_path, tmp_path, run, ["fixture"], {}, 100, attempts)
    assert result["retry_of"] == "attempt_001" and result["status"] == "completed"
    with pytest.raises(ValueError, match="Retry allowance exhausted"):
        pilot._record_attempt(tmp_path, tmp_path, run, ["fixture"], {}, 100, attempts)
