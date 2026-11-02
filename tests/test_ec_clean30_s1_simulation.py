"""The characterization must exercise the selector without rewriting calibration evidence."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ec_clean30 import s1_simulation as sim
from ec_clean30 import selection as sel


def test_fixed_spec_covers_declared_cases_and_refuses_alpha_search():
    spec = sim.make_spec()
    sim.validate_spec(spec)
    assert len(spec["scenarios"]) == 50
    assert {point["fold"] for point in spec["scenarios"]} == set(range(5))
    assert {point["effect"] for point in spec["scenarios"]} == set(sim.EFFECTS)
    assert spec["scenarios"][0]["effect"] == "null"
    spec["n_replicates"] = 201
    with pytest.raises(sim.S1SimulationError, match="predeclared"):
        sim.validate_spec(spec)


@pytest.mark.parametrize("gain", [-2, 0, 1, 2, 3])
def test_signed_effects_match_the_existing_expected_ba_model(gain):
    model = sel.NullNoiseModel()
    shift = sim.shift_for_gain(model, gain)
    actual = sim.expected_ba_points(model, shift) - sim.expected_ba_points(model)
    assert actual == pytest.approx(gain, abs=1e-8)


def test_every_replicate_runs_real_lr_selection_for_all_families(monkeypatch):
    choices, screens = [], []
    real_lr = sel.s1_choose_learning_rate
    real_screen = sel.s1_family_screen

    def lr(family, settings):
        assert tuple(settings) == sel.s1_learning_rate_labels(family)
        choices.append(family)
        return real_lr(family, settings)

    def screen(winners, **kwargs):
        assert set(winners) == set(sel.FAMILIES)
        screens.append(tuple(winners))
        return real_screen(winners, **kwargs)

    monkeypatch.setattr(sel, "s1_choose_learning_rate", lr)
    monkeypatch.setattr(sel, "s1_family_screen", screen)
    model = replace(sel.NullNoiseModel(), n_proteins=300)
    point = sim.make_spec()["scenarios"][1]
    first = sim.simulate_scenario(model, point, n_replicates=2, n_draws=50)
    second = sim.simulate_scenario(model, point, n_replicates=2, n_draws=50)
    assert first == second
    assert len(choices) == 4 * len(sel.FAMILIES)
    assert len(screens) == 4
    assert sum(first["winner_counts"].values()) == 2
    assert 0 <= first["bootstrap_aborts"] <= 2


def test_f4_only_simulation_records_gain_and_uses_both_lr_settings(monkeypatch):
    monkeypatch.setattr(sel, "s1_family_screen", lambda *args, **kwargs: SimpleNamespace(winner_family="F4"))
    model = replace(sel.NullNoiseModel(), n_proteins=200)
    result = sim.simulate_scenario(model, sim.make_spec()["scenarios"][1], n_replicates=3, n_draws=50)
    assert result["winner_counts"]["F4"] == 3
    assert result["mean_selected_true_gain_points"] == 2
    assert result["selected_harmful"] == result["selected_nonbeneficial"] == 0
    assert set(result["second_lr_selected"]) == set(sel.FAMILIES)


def test_failed_bootstrap_is_counted_and_cannot_select_a_challenger(monkeypatch):
    def fail(*args, **kwargs):
        raise sim.BootstrapAbort("missing class")
    monkeypatch.setattr(sel, "s1_family_screen", fail)
    result = sim.simulate_scenario(replace(sel.NullNoiseModel(), n_proteins=200),
                                   sim.make_spec()["scenarios"][0], n_replicates=2, n_draws=50)
    assert result["bootstrap_aborts"] == 2
    assert result["winner_counts"]["F2"] == 2


def test_write_once_and_snapshot_tampering_are_refused(tmp_path, monkeypatch):
    path = tmp_path / "record.json"
    sim._new_or_equal(path, {"result": 1})
    sim._new_or_equal(path, {"result": 1})
    with pytest.raises(sim.S1SimulationError, match="never overwrite"):
        sim._new_or_equal(path, {"result": 2})
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "module.py").write_text("x = 1\n")
    monkeypatch.setattr(sim, "SOURCE_FILES", ("module.py",))
    hashes = sim._source_hashes(repo)
    out = tmp_path / "out"
    sim._source_snapshot(out, repo, hashes)
    (out / "source/module.py").write_text("x = 2\n")
    with pytest.raises(sim.S1SimulationError, match="snapshot"):
        sim._source_snapshot(out, repo, hashes)


def test_first_scenario_reuse_and_budget_refusal(tmp_path, monkeypatch):
    spec = sim.make_spec()
    spec_path = tmp_path / "spec.json"
    sim._new_or_equal(spec_path, spec)
    model = sel.NullNoiseModel()
    models = {f"fold{fold}__{corner}": {"model": model, "source_file_sha256": "a" * 64}
              for fold in range(5) for corner in sim.CORNERS}
    monkeypatch.setattr(sim, "load_noise_models", lambda *args: models)
    monkeypatch.setattr(sim, "_source_hashes", lambda *args: {})
    monkeypatch.setattr(sim, "git_state", lambda *args: {"head": "draft"})
    calls = []
    monkeypatch.setattr(sim, "simulate_scenario", lambda *args, **kwargs: calls.append(args[1]["id"]) or {})
    # First scenario takes enough CPU time to make the full forecast exceed 8 h.
    times = iter([0.0, 1000.0])
    monkeypatch.setattr(sim.time, "process_time", lambda: next(times))
    kwargs = dict(spec_path=spec_path, base_freeze=tmp_path / "base.json",
                  campaign_root=tmp_path / "campaign", out=tmp_path / "out")
    code, timed = sim.run_characterization(**kwargs, first_only=True)
    assert code == 0 and timed["status"] == "timed_first_scenario"
    assert len(calls) == 1
    code, stopped = sim.run_characterization(**kwargs)
    assert code == 4 and stopped["status"] == "budget_decision_required"
    assert len(calls) == 1  # the first scenario was reused, not rerun


def test_identity_tree_is_refused_without_reading_it(tmp_path):
    path = tmp_path / "ec_class_vs_clean30_2026-10" / "identity" / "anything.json"
    with pytest.raises(sim.S1SimulationError, match="identity"):
        sim._safe_path(path)


def test_wilson_bounds_include_boundary_rates():
    assert sim.wilson(0, 200)[0] == pytest.approx(0)
    assert sim.wilson(200, 200)[1] == pytest.approx(1)
    assert sim.wilson(100, 200)[0] < .5 < sim.wilson(100, 200)[1]


def test_historical_noise_tampering_fails_before_characterization(tmp_path, monkeypatch):
    campaign = tmp_path / "campaign"
    history = campaign / "simulation/final_v1"
    entries = {}
    result = {"noise_model": sel.NullNoiseModel().to_record(),
              "promotion_config": sel.PromotionConfig().to_record(), "calibration": {}}
    for fold in range(5):
        for corner in sim.CORNERS:
            point_id = f"final-look1__fold{fold}__ordinary-a0.04__looks1__fold{fold}-corner-{corner}"
            relative = f"records/{point_id}.json"
            record = {"result": result, "result_sha256": sel.sha256_canonical(result), "smoke": False}
            digest = sim._new_or_equal(history / relative, record)
            entries[point_id] = {"file": relative, "sha256": digest, "result_sha256": record["result_sha256"]}
    receipt = history / "sweep_receipt.json"
    receipt_hash = sim._new_or_equal(receipt, {"records": entries})
    base = tmp_path / "freeze1.json"
    digest = sim._new_or_equal(base, {"bound_files": {"simulation_sweep_receipt": {
        "path": "simulation/final_v1/sweep_receipt.json", "sha256": receipt_hash}}})
    monkeypatch.setattr(sim, "FROZEN_BASE_SHA256", digest)
    spec = sim.make_spec()
    assert len(sim.load_noise_models(base, campaign, spec)) == 10
    first = history / next(iter(entries.values()))["file"]
    first.write_text(first.read_text() + " ")
    with pytest.raises(sim.S1SimulationError, match="noise record hash"):
        sim.load_noise_models(base, campaign, spec)
