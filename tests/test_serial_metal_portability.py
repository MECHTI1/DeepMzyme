"""Export paths without GPU allocation, scientific changes, or state migration."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from serial_metal_campaign import portability, profile, workflow
from test_serial_metal_profile import manifest


@pytest.fixture
def local_plan(manifest, tmp_path):
    output = tmp_path / "local-plan"
    value = deepcopy(manifest)
    config = profile.base.parse_config(next(iter(value["templates"].values()))["command"])
    data = config.structure_dir.parents[1]
    overlay = Path(config.external_features_root_dir) / "feature_overlay_manifest.json"
    overlay.write_text('{"fixture":true}\n')
    files = [config.structure_dir / "structure_manifest.csv", config.summary_csv]
    value.update(profile=profile.PROFILE, policy=profile.POLICY, root=str(ROOT), data_root=str(data),
                 output_dir=str(output), external_features_root_dir=str(overlay.parent),
                 feature_overlay_manifest=dict(path=str(overlay), sha256=profile.base.digest(overlay)),
                 dataset_files={str(path.relative_to(data)): profile.base.digest(path) for path in files},
                 source_files={"src/train.py": profile.base.digest(ROOT / "src/train.py")},
                 arms=profile.canonical(profile.ARMS), bundle_sha256=profile.base.BUNDLE_SHA)
    value["runs"] = profile.initial_runs(value)
    profile.base.save(output / "campaign_manifest.json", value)
    profile.base.save(output / "queue.json", dict(runs=value["runs"], phase="screen"))
    return output, value


def export(output, destination):
    return portability.export_runtime_plan(output, destination, "/worker/repo", "/worker/data",
        "/worker/campaign", "/worker/features", "/worker/python/bin/python")


def test_real_plan_exports_without_mistaking_notebook_planning_files_for_execution(manifest, tmp_path, monkeypatch):
    config = profile.base.parse_config(next(iter(manifest["templates"].values()))["command"])
    data = config.structure_dir.parents[1]
    external = Path(config.external_features_root_dir)
    overlay = external / "real-plan-feature-overlay.json"
    overlay.write_text('{"fixture":true}\n')
    output = tmp_path / "actual-plan"
    monkeypatch.setattr(profile.base, "validate_feature_overlay", lambda *args: None)
    planned = profile.plan(ROOT, data, output, external, overlay, source_commit="fixture")
    assert (output / "notebook_planning").is_dir()
    assert any((output / "notebook_planning").rglob("active_run_config.json"))
    assert not (output / "runs").exists()
    assert all(Path(row["config"]["runs_dir"]) == output / "runs" for row in planned["runs"])
    exported = export(output, tmp_path / "actual-export")
    assert len(exported["runs"]) == 65
    assert all(row["run_dir"].startswith("/worker/campaign/runs/") for row in exported["runs"])
    assert not (output / "runs").exists()


def test_export_rewrites_only_paths_and_keeps_native_science_and_source_untouched(local_plan, tmp_path):
    output, original = local_plan
    before = {str(path): profile.base.digest(path) for path in output.rglob("*") if path.is_file()}
    # Local preparation is not portable evidence and must not be copied.
    profile.base.save(output / "preparation.json", {"status": "local-only-fixture"})
    before[str(output / "preparation.json")] = profile.base.digest(output / "preparation.json")
    destination = tmp_path / "export"
    result = export(output, destination)
    assert result["root"] == "/worker/repo" and result["output_dir"] == "/worker/campaign"
    assert result["data_root"] == "/worker/data"
    assert result["feature_overlay_manifest"]["path"] == "/worker/features/feature_overlay_manifest.json"
    assert result["source_files"] == original["source_files"]
    assert result["dataset_files"] == original["dataset_files"] and result["policy"] == original["policy"]
    assert not (destination / "preparation.json").exists()
    for old, new in zip(original["runs"], result["runs"]):
        for key in ("id", "recipe_id", "parameters", "family", "scheme", "seed", "epochs", "fold_index", "ring"):
            assert new[key] == old[key]
        assert new["command"][0] == "/worker/python/bin/python"
        assert new["command"][1].startswith("/worker/repo/src/")
        assert new["config"]["selection_metric"] == "val_metal_balanced_acc"
        assert new["config"]["run_test_eval"] is False
        assert new["config"]["epochs"] == old["config"]["epochs"]
        assert new["config_sha256"] == profile.base.fingerprint(new["config"])
        assert new["config_sha256"] != old["config_sha256"]
        assert new["run_dir"].startswith("/worker/campaign/runs/")
    assert before == {str(path): profile.base.digest(path) for path in output.rglob("*") if path.is_file()}
    receipt = profile.base.read(destination / "runtime_export.json")
    assert receipt["worker_preparation_required"] and not receipt["execution_state_copied"]
    assert receipt["certified_reuse"] == 0
    assert profile.base.read(destination / "budget_forecast.json")["status"] == "preview_only_not_admitted"
    assert export(output, destination) == result  # identical review artifacts are idempotent


@pytest.mark.parametrize("state", ["attempts", "session", "queue", "reuse", "confirmation"])
def test_export_rejects_started_or_adapted_plan_without_writing(local_plan, tmp_path, state):
    output, value = local_plan
    if state == "attempts":
        profile.base.save(output / "attempts.json", [{"status": "failed"}])
    elif state == "session":
        profile.base.save(output / "sessions.json", [{"session_id": "allocated"}])
    elif state == "queue":
        profile.base.save(output / "queue.json", dict(runs=value["runs"], phase="repeat"))
    elif state == "reuse":
        profile.base.save(output / "reuse/any.json", {"certified": True})
    else:
        profile.base.save(output / "confirmation_manifest.json", {"runs": []})
    destination = tmp_path / "must-not-exist"
    with pytest.raises(ValueError):
        export(output, destination)
    assert not destination.exists()


def test_export_rejects_changed_scientific_source_before_writing(local_plan, tmp_path):
    output, value = local_plan
    value["source_files"]["src/train.py"] = "wrong-hash"
    profile.base.save(output / "campaign_manifest.json", value)
    with pytest.raises(ValueError, match="Source changed"):
        export(output, tmp_path / "export")


def test_environment_path_lists_and_nested_prefixes_are_remapped():
    value = {"env": {"PYTHONPATH": "/old/repo/src:/old/repo/data/helpers"}, "command": ["/old/python", "/old/repo/src/train.py"]}
    result = portability._remap(value, [("/old/repo/data", "/worker/data"), ("/old/repo", "/worker/repo")], "/worker/python")
    assert result["env"]["PYTHONPATH"] == "/worker/repo/src:/worker/data/helpers"
    assert result["command"] == ["/worker/python", "/worker/repo/src/train.py"]


def test_workflow_timings_prefer_recorded_setup_over_copied_mtime(tmp_path):
    run = dict(id="smoke", arm="gvp_four", epochs=1, parameters={})
    directory = tmp_path / "runs/smoke"
    profile.base.save(directory / "prepare_status.json", {"status": "ready"})
    profile.base.save(directory / "performance_profile.json", {"setup_seconds": 12.})
    profile.base.save(tmp_path / "sessions.json", [{"session_id": "one", "hardware": {"gpu": "fixture"}, "stopped_epoch": None}])
    profile.base.save(tmp_path / "queue.json", {"runs": [run]})
    profile.base.save(tmp_path / "attempts.json", [{"status": "completed", "run_id": "smoke", "session_id": "one",
        "run_dir": str(directory), "started_epoch": 1., "elapsed_seconds": 20.}])
    row = workflow._timings(tmp_path)[0]
    assert row["setup_seconds"] == 12. and row["epoch_seconds"] == 8.


def test_workflow_report_lists_each_confirmation_unit_once(tmp_path, monkeypatch):
    from test_serial_metal_reporting import confirmation
    confirm, results = confirmation()
    profile.base.save(tmp_path / "queue.json", {"runs": confirm["runs"]})
    profile.base.save(tmp_path / "confirmation_manifest.json", confirm)
    monkeypatch.setattr(profile, "verify_manifest", lambda _: {"runs": []})
    monkeypatch.setattr(workflow, "collect_results", lambda *a, **kw: results)
    monkeypatch.setattr(workflow, "ring_audit_directory", lambda *a: None)
    report = workflow.report(tmp_path)
    assert report["confirmation"]["complete"]
    assert not any("Duplicate planned" in issue for issue in report["diagnostics"])
