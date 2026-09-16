"""CPU evidence gates, including optional checks against immutable local history."""
import copy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from serial_metal_campaign import evidence


def config(**changes):
    return dict(model_architecture="gvp", fusion_mode="late_fusion", use_early_esm=False,
                early_esm_dropout=.2, use_ring_edges=False, shell_role_source="geometry",
                learning_rate=3e-5, **changes)


def test_inactive_compatibility_maps_only_the_two_approved_differences():
    old = config()
    new = {**old, "early_esm_dropout": 0, "shell_role_source": "edge_mode"}
    assert {row["field"] for row in evidence.compatible_config(old, new)} == {
        "early_esm_dropout", "shell_role_source"}
    with pytest.raises(ValueError, match="learning_rate"):
        evidence.compatible_config(old, {**new, "learning_rate": 1e-4})
    with pytest.raises(ValueError, match="early_esm_dropout"):
        evidence.compatible_config({**old, "early_esm_dropout": .1}, new)


@pytest.mark.parametrize("change", [{"fusion_mode": "early_fusion"}, {"use_early_esm": True},
                                      {"use_ring_edges": True}])
def test_inactive_mappings_fail_when_the_intervention_is_active(change):
    old = {**config(), **change}
    new = {**old, "early_esm_dropout": 0, "shell_role_source": "edge_mode"}
    with pytest.raises(ValueError, match="inactive early|RING off"):
        evidence.compatible_config(old, new)


def test_normalization_json_preserves_tensor_values_and_shapes():
    import torch
    assert evidence._jsonable({"means": {"x": torch.tensor([[1., 2.]])}, "clamp_value": 5.}) == {
        "means": {"x": [[1., 2.]]}, "clamp_value": 5.}


def test_epoch_cap_diagnostic_uses_approved_or_rule_without_reselection():
    history = [{"epoch": i, evidence.base.METRIC: .5} for i in range(1, 51)]
    row = evidence.epoch_cap_diagnostics(history, 46, 50)
    assert row["selected_in_final_five"] and row["potentially_training_budget_limited"]
    assert not row["best_checkpoint_at_epoch_cap"]
    for item in history[-5:]:
        item[evidence.base.METRIC] = .503
    row = evidence.epoch_cap_diagnostics(history, 20, 50)
    assert row["late_validation_mean_gain"] == pytest.approx(.003)
    assert row["potentially_training_budget_limited"]
    assert not evidence.epoch_cap_diagnostics(history[:1], 1, 1)["potentially_training_budget_limited"]


def test_scientific_sources_excludes_orchestration_but_not_training_run():
    manifest = {"source_files": {"src/training/run.py": "science", "src/model.py": "model",
        "src/run_serial_metal_campaign.py": "cli", "src/serial_metal_campaign/profile.py": "profile",
        "docs/Plan.md": "docs"}}
    assert evidence.scientific_sources(manifest) == {"src/training/run.py": "science", "src/model.py": "model"}


def test_cache_identity_is_portable_but_content_changes_are_visible():
    old = {"data_root": "/content/data", "external_features_root_dir": "/content/features"}
    new = {"data_root": "/local/data", "external_features_root_dir": "/local/features"}
    def cache(roots):
        return {"files": [{"kind": "esm", "path": roots["data_root"] + "/esm_embeddings/a.pt",
                           "sha256": "tensor", "bytes": 123, "mtime_ns": 99},
                          {"kind": "external", "path": roots["external_features_root_dir"] + "/a.json",
                           "sha256": "feature", "bytes": 10, "mtime_ns": 88}]}
    assert evidence.cache_identity(cache(old), old) == evidence.cache_identity(cache(new), new)
    changed = cache(new)
    changed["files"][0]["sha256"] = "different"
    assert evidence.cache_identity(cache(old), old) != evidence.cache_identity(changed, new)


def test_current_input_rehash_detects_changed_bytes_even_with_preserved_size_and_mtime(tmp_path, monkeypatch):
    import os
    feature = tmp_path / "features/a.json"
    feature.parent.mkdir()
    feature.write_text("original")
    manifest = dict(external_features_root_dir=str(feature.parent), data_root=str(tmp_path),
                    feature_overlay_manifest={"path": "synthetic-overlay"})
    evidence.base.save(tmp_path / "campaign_manifest.json", manifest)
    evidence.base.save(tmp_path / "fold_plan.json", {})
    info = feature.stat()
    cache = {"files": [{"kind": "external", "path": str(feature), "sha256": evidence.base.digest(feature),
                         "bytes": info.st_size, "mtime_ns": info.st_mtime_ns}]}
    evidence.base.save(tmp_path / "training_cache_audit.json", cache)
    evidence.base.save(tmp_path / "input_identity.json", {"files": evidence.cache_identity(cache, manifest)})
    evidence.base.save(tmp_path / "preparation.json", dict(status="passed",
        manifest_sha256=evidence.base.digest(tmp_path / "campaign_manifest.json"),
        folds_sha256=evidence.base.digest(tmp_path / "fold_plan.json"),
        inputs_sha256=evidence.base.digest(tmp_path / "input_identity.json")))
    monkeypatch.setattr(evidence.profile, "verify_manifest", lambda _: manifest)
    monkeypatch.setattr(evidence.base, "validate_feature_overlay", lambda *args: None)
    assert evidence.verify_preparation(tmp_path, hash_contents=True)["status"] == "passed"
    feature.write_text("modified")
    os.utime(feature, ns=(info.st_atime_ns, info.st_mtime_ns))
    with pytest.raises(ValueError, match="Feature contents changed"):
        evidence.verify_preparation(tmp_path, hash_contents=True)


def reuse_fixture(tmp_path, monkeypatch):
    """Small immutable files exercise importer bindings; heavyweight gates isolated."""
    output, source = tmp_path / "current", tmp_path / "historical"
    directory = source / "runs" / "old-run"
    directory.mkdir(parents=True)
    output.mkdir()
    old_config = config()
    target = dict(id="new-run", family="Only-GVP", scheme="four_class", seed=42, epochs=50,
                  config={**old_config, "early_esm_dropout": 0., "shell_role_source": "edge_mode"})
    old_run = {**target, "id": "old-run", "config": old_config}
    common = dict(source_files={"src/model.py": "model-sha"}, dataset_files={"train.csv": "data-sha"},
                  bundle_sha256="bundle", data_root="/data", external_features_root_dir="/features")
    current = {**common, "runs": [target]}
    historical = {**common, "runs": [old_run]}
    evidence.base.save(output / "campaign_manifest.json", current)
    evidence.base.save(source / "campaign_manifest.json", historical)
    artifact_hashes = {}
    for name in evidence.ARTIFACT_NAMES:
        (directory / name).write_text(name)
        artifact_hashes[name] = evidence.base.digest(directory / name)
    provenance = dict(manifest_sha256=evidence.base.digest(source / "campaign_manifest.json"), runs=[
        dict(run_id="old-run", run_dir="/old-host/runs/old-run", files=artifact_hashes,
             normalization_stats_sha256="norm")])
    evidence.base.save(source / "final_capture_receipt.json", provenance)
    cache = dict(status="passed", files=[])
    evidence.base.save(source / "training_cache_audit.json", cache)
    evidence.base.save(output / "input_identity.json", dict(files=[]))
    audit = dict(status="passed", cache_audit_sha256=evidence.base.digest(source / "training_cache_audit.json"),
                 expected_cohort_sha256="cohort", normalization={"off_sha256": "norm"})
    evidence.base.save(source / "ring_input_audit.json", audit)
    monkeypatch.setattr(evidence.profile, "verify_manifest", lambda _: current)
    monkeypatch.setattr(evidence, "verify_preparation", lambda *a, **kw: dict(inputs_sha256="inputs", folds_sha256="folds"))
    monkeypatch.setattr(evidence, "verify_result", lambda *a, **kw: dict(id="new-run", normalization_sha256="norm", cohort_sha256="cohort"))
    return output, source, target, current, directory


def test_reuse_import_writes_only_receipt_and_rechecks_historical_artifacts(tmp_path, monkeypatch):
    output, source, target, _, directory = reuse_fixture(tmp_path, monkeypatch)
    before = {str(path): evidence.base.digest(path) for path in source.rglob("*") if path.is_file()}
    receipt = evidence.import_reuse(output, target, source, "old-run")
    assert receipt["certified"] and receipt["result"]["reuse_certified"]
    assert (output / "reuse/new-run.json").is_file()
    assert before == {str(path): evidence.base.digest(path) for path in source.rglob("*") if path.is_file()}
    assert evidence.verify_reuse_receipt(output, receipt) == receipt
    (directory / "best_model_checkpoint.pt").write_text("changed")
    with pytest.raises(ValueError, match="artifact is missing or changed"):
        evidence.verify_reuse_receipt(output, receipt)


@pytest.mark.parametrize("fault", ["source", "normalization", "target", "receipt"])
def test_reuse_fails_closed_on_binding_changes(tmp_path, monkeypatch, fault):
    output, source, target, current, _ = reuse_fixture(tmp_path, monkeypatch)
    receipt = evidence.import_reuse(output, target, source, "old-run")
    if fault == "source":
        current["source_files"]["src/model.py"] = "changed-science"
    elif fault == "normalization":
        audit = evidence.base.read(source / "ring_input_audit.json")
        audit["normalization"]["off_sha256"] = "changed-normalization"
        evidence.base.save(source / "ring_input_audit.json", audit)
    elif fault == "target":
        manifest = evidence.base.read(output / "campaign_manifest.json")
        manifest["runs"][0]["config"]["learning_rate"] = .5
        evidence.base.save(output / "campaign_manifest.json", manifest)
    else:
        receipt["result"]["balanced_accuracy"] = .99
    with pytest.raises((RuntimeError, ValueError)):
        evidence.verify_reuse_receipt(output, receipt)


HISTORICAL = ROOT / "DeepMzyme_Data/notebook_outputs/campaigns/metal_ring_pilot_v1_20260915"


@pytest.mark.skipif(not HISTORICAL.is_dir(), reason="Optional immutable local historical artifacts are absent")
def test_real_saved_checkpoint_config_normalization_and_parameter_shapes(tmp_path, monkeypatch):
    import torch
    manifest = evidence.base.read(HISTORICAL / "campaign_manifest.json")
    source_run = next(row for row in manifest["runs"] if row["id"] == "GVP_GVP_ring0_lr3e-05_s42")
    directory = HISTORICAL / "runs" / source_run["id"]
    reference = evidence.base.read(directory / "dataset_summary.json")
    evidence.base.save(tmp_path / "fold_plan.json", {"references": {"discovery": reference}})
    evidence.base.save(tmp_path / "input_identity.json", {"files": []})
    monkeypatch.setattr(evidence.profile, "verify_manifest", lambda _: manifest)
    run = {**source_run, "stage": "discovery", "arm": "gvp_four", "parameters": {}, "recipe_id": "actual-history",
           "fold_index": None, "complexity_proxy": 1, "config_sha256": "source-config", "ring": False}
    result = evidence.verify_result(tmp_path, run, directory)
    assert result["parameter_count"] > 1000 and result["selected_epoch"] == 44
    checkpoint = torch.load(directory / "best_model_checkpoint.pt", map_location="cpu", weights_only=True)
    altered = copy.copy(checkpoint)
    altered["model_state_dict"] = dict(checkpoint["model_state_dict"])
    key = next(name for name, tensor in altered["model_state_dict"].items() if tensor.ndim > 0 and tensor.shape[0] > 1)
    altered["model_state_dict"][key] = altered["model_state_dict"][key][:1]
    with pytest.raises((RuntimeError, ValueError)):
        evidence.checkpoint_parameter_count(altered)
    cache = evidence.base.read(HISTORICAL / "training_cache_audit.json")
    portable = evidence.cache_identity(cache, manifest)
    assert portable and all(not Path(row["relative_path"]).is_absolute() for row in portable)
