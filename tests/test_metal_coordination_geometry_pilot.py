"""Geometry campaign orchestration checks without GPU allocation or training."""
from dataclasses import replace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import run_metal_coordination_geometry_pilot as pilot


@pytest.fixture
def planned(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    data, output, parent = (tmp_path / name for name in ("data", "geometry", "original"))
    train = data / pilot.DATASET / "train"
    train.mkdir(parents=True)
    (train / "structure_manifest.csv").write_text("logical_filename,sha256\n")
    (train / "final_data_summarazing_table_transition_metals_only_catalytic.csv").write_text(
        "structure,chain_resi,metaltype,ecnumber\n")
    (data / "esm_embeddings").mkdir()
    (data / "esm_embeddings/fixture_esmc.pt").touch()
    overlay = tmp_path / "overlay/feature_overlay_manifest.json"
    pilot.save(overlay, {"status": "complete"})
    monkeypatch.setattr(pilot.base, "validate_feature_overlay", lambda *args: {})
    pilot.save(parent / "campaign_manifest.json", dict(
        profile=pilot.base.PROFILE, dataset=pilot.DATASET, bundle_sha256=pilot.BUNDLE_SHA,
        external_features_root_dir=str(overlay.parent), feature_overlay_manifest={"path": str(overlay)},
        runs=[dict(id=f"{block}_{i}", block=block) for block in ("A1", "A2") for i in range(4)]))
    pilot.save(parent / "expected_split.json", {"retained_split_identity": {
        "train": {"examples": [dict(structure_id="a", pocket_id="a0", group="a", y_metal=0)]},
        "validation": {"examples": [dict(structure_id="b", pocket_id="b0", group="b", y_metal=1)]}}})
    manifest = pilot.plan(root, data, output, "geometry_source", parent)
    return root, data, output, parent, manifest


def result(run, score=.7, minimum=.4):
    return {**run, "balanced_accuracy": score, "minimum_recall": minimum,
            "normalization_stats_sha256": "no_nodes" if run["arm"] in "ABC" else "metal_nodes"}


def test_separate_immutable_plan_canonical_controls_and_exact_cohort(planned):
    root, data, output, parent, manifest = planned
    assert manifest["profile"] == pilot.PROFILE != pilot.base.PROFILE
    assert manifest["maximum_smokes"] == 5 and manifest["maximum_full_runs"] == 15
    assert len(manifest["runs"]) == 25  # Includes ten alternative templates for five seed repeats.
    assert len(pilot.active_runs(manifest, [])) == 15
    assert len(pilot.active_runs(manifest, [], allow_repeat_placeholders=True)) == 20
    assert pilot.base.digest(output / "expected_split.json") == pilot.base.digest(parent / "expected_split.json")
    for run in manifest["runs"]:
        config = pilot.base.parse_config(run["command"])
        pilot.validate(config, run["arm"], run["epochs"])
        assert (config.site_geometry_features, config.metal_node_mode) == pilot.ARMS[run["arm"]]
        assert config.structural_readout_scope == "residue_only"
        assert config.metal_label_scheme == "merge_fe_class_viii"
        assert config.metal_eligibility_scheme == "six_class"
        assert not config.run_test_eval and config.test_structure_dir is None
        assert run["env"]["DEEPGM_METAL_LABEL_SCHEME"] == "four_class"
    assert pilot.plan(root, data, output, "geometry_source", parent) == manifest
    assert pilot.verify_manifest(root, output) == manifest
    with pytest.raises(ValueError, match="manifest changed"):
        pilot.plan(root, data, output, "changed_source", parent)


def test_configuration_guards_reject_coupled_changes(planned):
    config = pilot.base.parse_config(planned[-1]["runs"][0]["command"])
    for changed in (replace(config, structural_readout_scope="all_nodes"),
                    replace(config, site_geometry_features="legacy"),
                    replace(config, learning_rate=1e-3), replace(config, metal_node_mode="per_metal")):
        with pytest.raises(ValueError):
            pilot.validate(changed, "A", 1)


def test_parent_and_overlay_mutations_rejected(planned):
    root, _, output, parent, manifest = planned
    original = (parent / "campaign_manifest.json").read_bytes()
    pilot.save(parent / "campaign_manifest.json", {"changed": True})
    with pytest.raises(ValueError, match="Parent manifest"):
        pilot.verify_manifest(root, output)
    (parent / "campaign_manifest.json").write_bytes(original)
    pilot.save(Path(manifest["feature_overlay_manifest"]["path"]), {"changed": True})
    with pytest.raises(ValueError, match="overlay manifest"):
        pilot.verify_manifest(root, output)


def test_lr_selection_requires_both_values_and_is_native_then_recall_then_lower_lr(planned):
    manifest = planned[-1]
    results = [result(r) for r in manifest["runs"] if r["block"] in ("G1", "G2")]
    assert pilot.selected_lrs(results[:5]) == {}
    assert pilot.selected_lrs(results) == dict.fromkeys(pilot.ARMS, 3e-5)
    high = next(r for r in results if r["arm"] == "A" and r["lr"] == 1e-4)
    high.update(balanced_accuracy=.71, minimum_recall=.3)
    assert pilot.selected_lrs(results)["A"] == 1e-4
    high.update(balanced_accuracy=.7, minimum_recall=.41)
    assert pilot.selected_lrs(results)["A"] == 1e-4
    active = pilot.active_runs(manifest, results)
    assert len([r for r in active if r["block"] == "GR"]) == 5
    assert next(r for r in active if r["block"] == "GR" and r["arm"] == "A")["lr"] == 1e-4


def test_geometry_forecast_is_per_arm_and_preserves_setup(planned):
    manifest = planned[-1]
    smokes = [r for r in manifest["runs"] if r["block"] == "S"]
    attempts = [dict(run_id=r["id"], status="completed", setup_seconds=100, epoch_seconds=i + 1)
                for i, r in enumerate(smokes)]
    full = next(r for r in manifest["runs"] if r["block"] == "G1" and r["arm"] == "E")
    assert pilot.forecast(full, attempts, manifest) == 350
    with pytest.raises(ValueError, match="timing smoke"):
        pilot.forecast(full, attempts[:-1], manifest)


def test_shared_budget_counts_failed_geometry_setup_smokes_and_retries(planned):
    _, _, output, parent, _ = planned
    pilot.save(parent / "campaign_attempt_ledger.json", [
        dict(block="P", elapsed_seconds=300), dict(block="A1", elapsed_seconds=1000),
        dict(block="A1", elapsed_seconds=None, status="interrupted"),
        dict(block="A1", elapsed_seconds=200, retry_of="lost")])
    pilot.save(output / "campaign_attempt_ledger.json", [
        dict(block="P", elapsed_seconds=100), dict(block="S", elapsed_seconds=120, status="failed"),
        dict(block="G1", elapsed_seconds=300), dict(block="S", elapsed_seconds=90, retry_of="failed")])
    assert pilot.shared_budget_available(output, parent, 5000) == 450 * 60 - 1520
    assert pilot.shared_budget_available(output, parent, 5000, retry=True) == 60 * 60 - 290
    assert pilot.shared_budget_available(output, parent, 569 * 60) == 60
    assert pilot.shared_budget_available(output, parent, 570 * 60) == 0
    pilot.publish_budget_usage(output, parent)
    usage = pilot.read(parent / "coordination_geometry_budget_usage.json")
    assert usage["normal_elapsed_seconds"] == 520 and usage["retry_elapsed_seconds"] == 90


def test_parent_completed_a1_a2_gate_requires_verified_runs(planned, monkeypatch):
    parent = planned[3]
    complete = pilot.read(parent / "campaign_manifest.json")["runs"]
    monkeypatch.setattr(pilot.base, "completed_rows", lambda *args: complete[:-1])
    with pytest.raises(ValueError, match="A1/A2"):
        pilot.require_parent_architecture_screen(parent)
    monkeypatch.setattr(pilot.base, "completed_rows", lambda *args: complete)
    pilot.require_parent_architecture_screen(parent)


def test_same_graph_normalization_checks_accept_node_package_difference(planned):
    results = [result(r) for r in planned[-1]["runs"] if r["block"] == "S"]
    assert pilot.check_normalization_controls(results)["B"] != pilot.check_normalization_controls(results)["D"]
    with pytest.raises(ValueError, match="five geometry smokes"):
        pilot.check_normalization_controls(results[:-1])
    results[2]["normalization_stats_sha256"] = "changed"
    with pytest.raises(ValueError, match="different fitted normalization"):
        pilot.check_normalization_controls(results)


def test_all_fifteen_full_fits_must_fit_before_first_admission(planned, monkeypatch):
    root, _, output, parent, manifest = planned
    smokes = [result(r) for r in manifest["runs"] if r["block"] == "S"]
    attempts = [dict(run_id=r["id"], status="completed", setup_seconds=100, epoch_seconds=20) for r in smokes]
    monkeypatch.setattr(pilot, "_context", lambda *args: (root, output, parent, manifest, {}, 570 * 60 - 10000, attempts))
    monkeypatch.setattr(pilot, "require_parent_architecture_screen", lambda *args: None)
    monkeypatch.setattr(pilot.base, "verify_training_cache", lambda *args: None)
    monkeypatch.setattr(pilot, "rows", lambda *args: smokes)
    monkeypatch.setattr(pilot.base, "_record_attempt", lambda *args: pytest.fail("Training must not launch"))
    pilot.save(output / "readiness.json", dict(status="passed", manifest_sha256=pilot.base.digest(output / "campaign_manifest.json")))
    summary = pilot.execute(root, output, 1000, budget_root=parent)
    assert summary["state"]["status"] == "budget_stopped"
    assert summary["state"]["forecast_seconds"] == 15 * 1100 * 1.25
    assert not (output / "geometry_campaign_admission.json").exists()


def test_summary_excludes_smokes_and_requires_selected_lr_seed_pair(planned, monkeypatch):
    _, _, output, _, manifest = planned
    results = [result(r) for r in manifest["runs"] if r["block"] in ("S", "G1", "G2")]
    repeat = next(r for r in manifest["runs"] if r["block"] == "GR" and r["arm"] == "A" and r["lr"] == 3e-5)
    results.append(result(repeat, .8))
    monkeypatch.setattr(pilot, "rows", lambda *args: results)
    summary = pilot.summarize(output)
    assert summary["completed_smokes"] == 5 and summary["completed_full_runs"] == 11
    assert summary["coverage"]["G1"]["complete"] and not summary["coverage"]["GR"]["complete"]
    assert not summary["promoted"] and not summary["confidence_intervals_computed"]
    assert len(summary["matched_two_seed_repeats"]) == 1
    pair = summary["matched_two_seed_repeats"][0]
    assert pair["mean_native_balanced_accuracy"] == .75
    assert pair["sample_sd_native_balanced_accuracy"] == pytest.approx(.1 / 2**.5)
    assert all(r["block"] != "S" for r in summary["runs"])


def test_completed_output_extracts_selected_checkpoint_and_normalization_hash(planned):
    _, _, output, _, manifest = planned
    run = next(r for r in manifest["runs"] if r["block"] == "G1")
    directory = Path(run["run_dir"])
    history = [dict(epoch=i, val_metal_balanced_acc=.8 if i == 3 else .7,
                    val_metal_per_class_recall={"Mn": .9, "Cu": .6, "Zn": .8, "Class VIII": .9})
               for i in range(1, 51)]
    pilot.save(directory / "run_config.json", dict(config=run["config"], history=history))
    normalization = {"means": {"edge_dist": [1.5]}, "stds": {"edge_dist": [.5]}, "clamp_value": 5}
    metadata = dict(selected_checkpoint_epoch=3, selection_metric=pilot.METRIC, test_report=None,
                    normalization_stats=normalization)
    pilot.save(directory / "run_metadata.json", metadata)
    pilot.save(directory / "dataset_summary.json", pilot.read(output / "expected_split.json"))
    (directory / "epoch_metrics.csv").write_text("epoch\n" + "\n".join(map(str, range(1, 51))))
    for name in ("best_model_checkpoint.pt", "last_model_checkpoint.pt"):
        (directory / name).write_bytes(b"synthetic-checkpoint")
    pilot.save(output / "campaign_attempt_ledger.json", [dict(
        status="completed", run_id=run["id"], run_dir=str(directory), elapsed_seconds=300)])
    rows = pilot.rows(output, manifest)
    assert rows[0]["selected_epoch"] == 3 and rows[0]["balanced_accuracy"] == .8
    assert rows[0]["normalization_stats_sha256"] == pilot.base.fingerprint(normalization)
    metadata.pop("normalization_stats")
    pilot.save(directory / "run_metadata.json", metadata)
    with pytest.raises(ValueError, match="normalization metadata"):
        pilot.rows(output, manifest)
