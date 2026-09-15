"""Scientific controls and budget admission without allocating a GPU."""
from dataclasses import replace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import run_metal_ring_pilot as pilot


@pytest.fixture
def planned(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    data, output, parent, budget = (tmp_path / x for x in ("data", "ring", "parent", "budget"))
    train = data / pilot.base.DATASET / "train"
    train.mkdir(parents=True)
    (train / "structure_manifest.csv").write_text("logical_filename,sha256\n")
    (train / "final_data_summarazing_table_transition_metals_only_catalytic.csv").write_text("structure,chain_resi,metaltype,ecnumber\n")
    (data / "esm_embeddings").mkdir()
    (data / "esm_embeddings/fixture_esmc.pt").touch()
    overlay = tmp_path / "overlay/feature_overlay_manifest.json"
    pilot.save(overlay, {"status": "complete"})
    monkeypatch.setattr(pilot.base, "validate_feature_overlay", lambda *args: {})
    pilot.save(parent / "campaign_manifest.json", dict(profile=pilot.base.PROFILE, dataset=pilot.base.DATASET,
        bundle_sha256=pilot.base.BUNDLE_SHA, external_features_root_dir=str(overlay.parent),
        feature_overlay_manifest={"path": str(overlay)}))
    pilot.save(parent / "expected_split.json", {"retained_split_identity": {
        "train": {"examples": [dict(structure_id="a", pocket_id="a0", group="a")]},
        "validation": {"examples": [dict(structure_id="b", pocket_id="b0", group="b")]}}})
    pilot.save(parent / "expected_ring_files.json", {})
    pilot.save(parent / "expected_feature_files.json", {})
    pilot.save(parent / "expected_cache_inventory_provenance.json", dict(v12_bundle_sha256=pilot.base.BUNDLE_SHA,
        expected_ring_files_sha256=pilot.base.digest(parent / "expected_ring_files.json"),
        expected_feature_files_sha256=pilot.base.digest(parent / "expected_feature_files.json")))
    pilot.save(budget / "budget_handoff.json", dict(profile=pilot.BUDGET_PROFILE, total_cap_seconds=36000,
        training_cap_seconds=34200, main_cap_seconds=27000, prior_main_seconds=12000, prior_retry_seconds=300,
        prior_allocated_seconds=19000, prior_intervals=[dict(started_epoch=1, ended_epoch=4001),
                                                     dict(started_epoch=5001, ended_epoch=20001)]))
    manifest = pilot.plan(root, data, output, "source", parent, budget_root=budget)
    return root, data, output, parent, budget, manifest


def test_fresh_complete_matched_matrix_and_frozen_identity(planned):
    root, data, output, parent, budget, manifest = planned
    assert len(manifest["runs"]) == 20
    assert [sum(r["block"] == b for r in manifest["runs"]) for b in pilot.BLOCKS] == [4, 8, 8]
    for run in manifest["runs"]:
        config = pilot.base.parse_config(run["command"])
        pilot.validate(config, run["ring"], run["family"], run["epochs"])
        assert config.shell_role_source == "geometry"
        assert config.metal_eligibility_scheme == "six_class"
        assert not config.run_test_eval and config.test_structure_dir is None
    assert pilot.plan(root, data, output, "source", parent, budget_root=budget) == manifest
    assert pilot.verify_manifest(root, output) == manifest
    with pytest.raises(ValueError, match="manifest changed"):
        pilot.plan(root, data, output, "other", parent, budget_root=budget)


@pytest.mark.parametrize("field,value", [("shell_role_source", "edge_mode"), ("edge_radius", 8),
    ("metal_node_mode", "per_metal"), ("site_geometry_features", "counts_angles"),
    ("metal_label_scheme", "six_class"), ("prepare_missing_ring_edges", True),
    ("require_ring_edges", True), ("run_test_eval", True), ("use_esm_branch", True)])
def test_reject_coupled_changes(planned, field, value):
    run = planned[-1]["runs"][0]
    config = pilot.base.parse_config(run["command"])
    with pytest.raises(ValueError):
        pilot.validate(replace(config, **{field: value}), False, "Only-GVP", 1)


def test_cumulative_budget_counts_setup_failed_preflight_and_retries(planned):
    _, _, output, _, budget, _ = planned
    pilot.save(budget / "bootstrap_budget_usage.json", dict(normal_elapsed_seconds=400))
    pilot.save(output / "campaign_attempt_ledger.json", [dict(elapsed_seconds=100, block="P", status="failed"),
        dict(elapsed_seconds=200, block="S"), dict(elapsed_seconds=50, block="P", retry_of="failed")])
    assert pilot.available_seconds(output, budget, 20000) == 14200
    assert pilot.available_seconds(output, budget, 21000) == 13200
    assert pilot.available_seconds(output, budget, 20000, retry=True) == 3250
    assert pilot.available_seconds(output, budget, 34200) == 0


def test_budget_handoff_mutation_rejected(planned):
    root, _, output, _, budget, _ = planned
    handoff = pilot.read(budget / "budget_handoff.json")
    handoff["prior_main_seconds"] = 0
    pilot.save(budget / "budget_handoff.json", handoff)
    with pytest.raises(ValueError, match="handoff changed"):
        pilot.verify_manifest(root, output)


def test_timing_uses_family_and_edge_mode_and_maximum(planned):
    manifest = planned[-1]
    smokes = [r for r in manifest["runs"] if r["block"] == "S"]
    attempts = [dict(run_id=r["id"], status="completed", setup_seconds=100, epoch_seconds=i+1)
                for i, r in enumerate(smokes)]
    run = next(r for r in manifest["runs"] if r["block"] == "LATE" and r["ring"])
    assert pilot.forecast(run, attempts, manifest) == 300
    with pytest.raises(ValueError, match="timing smokes"):
        pilot.forecast(run, attempts[:-1], manifest)


def test_partial_block_never_produces_paired_summary(planned, monkeypatch):
    _, _, output, _, _, manifest = planned
    results = [dict(r, balanced_accuracy=.7 + .01*r["ring"], minimum_recall=.4)
               for r in manifest["runs"] if r["block"] in ("S", "GVP")]
    monkeypatch.setattr(pilot, "rows", lambda *args: results[:-1])
    assert pilot.summarize(output)["paired_fixed_recipe_deltas"] == []
    monkeypatch.setattr(pilot, "rows", lambda *args: results)
    summary = pilot.summarize(output)
    assert len(summary["paired_fixed_recipe_deltas"]) == 4
    assert all(p["delta_balanced_accuracy"] == pytest.approx(.01) for p in summary["paired_fixed_recipe_deltas"])
    assert not summary["promoted"] and not summary["held_out_evaluation"]


@pytest.mark.parametrize("declined_block", ["GVP", "LATE"])
def test_execute_declines_whole_family_without_launching_partial_block(planned, monkeypatch, declined_block):
    root, _, output, _, budget, manifest = planned
    completed_blocks = ("S",) if declined_block == "GVP" else ("S", "GVP")
    results = [dict(run, balanced_accuracy=.7, minimum_recall=.4)
               for run in manifest["runs"] if run["block"] in completed_blocks]
    pilot.save(output / "readiness.json", dict(status="passed", manifest_sha256=pilot.base.digest(output / "campaign_manifest.json")))
    pilot.save(output / "campaign_attempt_ledger.json", [])
    # Eight remaining runs × 200 seconds × 1.25 = 2,000 seconds;
    # only 1,999 remain under the original cumulative training deadline.
    monkeypatch.setattr(pilot, "_context", lambda *args: (root, output, budget, manifest,
        {"method": "verified_archive_transfer"}, 32201, []))
    monkeypatch.setattr(pilot.base, "verify_training_cache", lambda *args: None)
    monkeypatch.setattr(pilot, "rows", lambda *args: results)
    monkeypatch.setattr(pilot, "forecast", lambda *args: 200)
    monkeypatch.setattr(pilot.base, "_record_attempt", lambda *args, **kwargs: pytest.fail("No partial family block may launch"))
    summary = pilot.execute(root, output, 1, budget_root=budget)
    assert summary["state"]["status"] == "budget_stopped"
    assert summary["state"]["block"] == declined_block
    assert summary["state"]["forecast_seconds"] == 2000
    assert summary["state"]["available_seconds"] == 1999
    assert not (output / f"{declined_block.lower()}_admission.json").exists()
    if declined_block == "LATE":
        assert summary["coverage"]["GVP"]["complete"]
        assert summary["completed_full_runs"] == 8
        assert len(summary["paired_fixed_recipe_deltas"]) == 4
    else:
        assert summary["completed_full_runs"] == 0
    assert pilot.read(output / "campaign_attempt_ledger.json") == []


def test_execute_failed_retry_uses_retry_budget_and_counts_failure(planned, monkeypatch):
    root, _, output, _, budget, manifest = planned
    results = [dict(run, balanced_accuracy=.7, minimum_recall=.4) for run in manifest["runs"] if run["block"] == "S"]
    pending = next(run for run in manifest["runs"] if run["block"] == "GVP")
    attempts = [dict(attempt_id="attempt_005", run_id=pending["id"], block="GVP", status="failed", elapsed_seconds=60)]
    pilot.save(output / "campaign_attempt_ledger.json", attempts)
    pilot.save(output / "readiness.json", dict(status="passed", manifest_sha256=pilot.base.digest(output / "campaign_manifest.json")))
    monkeypatch.setattr(pilot, "_context", lambda *args: (root, output, budget, manifest,
        {"method": "verified_archive_transfer"}, 20000, attempts))
    monkeypatch.setattr(pilot.base, "verify_training_cache", lambda *args: None)
    monkeypatch.setattr(pilot.base, "allocation_elapsed", lambda *args: 20055)
    monkeypatch.setattr(pilot, "rows", lambda *args: results)
    monkeypatch.setattr(pilot, "forecast", lambda *args: 100)
    normal_before = pilot.available_seconds(output, budget, 20000)
    received = []
    def fail_retry(root_arg, output_arg, run, command, env, available, ledger):
        received.append(available)
        ledger.append(dict(attempt_id="attempt_006", run_id=run["id"], block="GVP", status="failed",
                           retry_of="attempt_005", elapsed_seconds=55))
        pilot.save(output / "campaign_attempt_ledger.json", ledger)
        raise RuntimeError("synthetic retry failure")
    monkeypatch.setattr(pilot.base, "_record_attempt", fail_retry)
    with pytest.raises(RuntimeError, match="synthetic retry failure"):
        pilot.execute(root, output, 1, budget_root=budget)
    assert received == [3300]  # Existing handoff retry cost is 300 seconds.
    assert pilot.available_seconds(output, budget, 20000, retry=True) == 3245
    assert pilot.available_seconds(output, budget, 20000) == normal_before
    assert pilot.read(output / "campaign_state.json")["status"] == "failed"
    assert pilot.read(output / "campaign_attempt_ledger.json")[-1]["elapsed_seconds"] == 55
