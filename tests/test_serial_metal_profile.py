"""Actual notebook/CLI expansion for the bounded v2 campaign; no GPU work."""
from collections import Counter
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serial_metal_campaign import profile


@pytest.fixture(scope="module")
def manifest(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    temporary = tmp_path_factory.mktemp("serial_profile")
    data, output = temporary / "data", temporary / "output"
    train = data / profile.base.DATASET / "train"
    train.mkdir(parents=True)
    (train / "final_data_summarazing_table_transition_metals_only_catalytic.csv").write_text(
        "structure,chain_resi,metaltype,ecnumber\n")
    (train / "structure_manifest.csv").write_text("logical_filename,sha256\n")
    (data / "esm_embeddings").mkdir()
    (data / "esm_embeddings/fixture_esmc.pt").touch()
    overlay = temporary / "overlay"
    overlay.mkdir()
    templates = profile.base._templates(root, data, output, overlay)
    return {
        "templates": profile.canonical(templates),
        "output_dir": str(output),
        "data_root": str(data),
    }


def test_initial_matrix_uses_real_command_parser_and_all_mandatory_recipes(manifest):
    runs = profile.initial_runs(manifest)
    assert profile.PROFILE == "metal_single_gpu_20h_v2"
    assert len(runs) == len({row["id"] for row in runs}) == 65
    assert Counter(row["block"] for row in runs) == {"screen": 48, "large": 4, "smoke": 13}
    assert Counter(row["arm"] for row in runs if row["block"] == "screen") == dict.fromkeys(profile.ARMS, 6)
    large = [row for row in runs if row["block"] == "large"]
    assert {(r["arm"], r["seed"]) for r in large} == {
        (arm, seed) for arm in ("late_four", "late_six") for seed in (42, 43)}
    for run in runs:
        config = profile.base.parse_config(run["command"])
        profile.validate(config, epochs=run["epochs"], ring=run["ring"])
        assert config.shell_role_source in {"edge_mode", "geometry"}
        assert Path(run["command"][1]).name == "train_serial_metal_profile.py"
        assert "--capacity" not in run["command"]
        assert not config.run_test_eval and config.test_summary_csv is None and config.test_structure_dir is None
        assert config.selection_metric == "val_metal_balanced_acc"
    for run in large:
        config = profile.base.parse_config(run["command"])
        assert (config.hidden_s, config.hidden_v, config.edge_hidden) == (256, 32, 128)
        assert (config.gvp_layers, config.head_mlp_layers, config.esm_fusion_dim) == (4, 2, 128)
        assert config.learning_rate == 3e-5 and config.epochs == 50


def test_confirmation_declares_all_110_units_in_cost_priority_order(manifest):
    initial = profile.initial_runs(manifest)
    candidates = {arm: next(row for row in initial if row["arm"] == arm and row["block"] == "screen")
                  for arm in profile.ARMS}
    runs = profile.confirmation_runs(manifest, candidates)
    assert len(runs) == len({row["id"] for row in runs}) == 110
    assert Counter(row["block"] for row in runs) == {"core": 60, "five": 10, "fusion": 20, "ring": 20}
    assert list(dict.fromkeys(row["block"] for row in runs)) == ["core", "five", "fusion", "ring"]
    for arm in {row["arm"] for row in runs}:
        assert {(r["fold_index"], r["seed"]) for r in runs if r["arm"] == arm} == {
            (fold, seed) for fold in range(5) for seed in (42, 43)}
    for row in runs:
        config = profile.base.parse_config(row["command"])
        profile.validate(config, fold_index=row["fold_index"], ring=row["ring"])
        if row["arm"].startswith("gvp_ring"):
            assert config.shell_role_source == "geometry"
        if row["arm"] == "late_five":
            assert row["parameters"] == profile.reference_parameters()
            assert row["scheme"] == "five_class"


def test_top_two_repeat_count_and_large_remains_separate(manifest):
    screen = [row for row in profile.initial_runs(manifest) if row["block"] == "screen"]
    selected = {arm: [row for row in screen if row["arm"] == arm][:2] for arm in profile.ARMS}
    repeats = profile.top2_repeat_runs(manifest, selected)
    assert len(repeats) == 16
    assert all(row["seed"] == 43 and row["block"] == "repeat" for row in repeats)
    selected["late_four"] = selected["late_four"][:1] * 2
    with pytest.raises(ValueError, match="distinct"):
        profile.top2_repeat_runs(manifest, selected)


def repeated(candidates, zero=False, gap=False):
    return [{**candidate, "seed": seed, "balanced_accuracy": .7,
             "training_balanced_accuracy": .9 if gap else .72,
             "per_class_recall": {"Mn": .8, "Cu": 0 if zero else .6}}
            for candidate in candidates.values() for seed in profile.SEEDS]


def test_mixed_diagnostics_use_two_categories_before_two_boundary_directions(manifest):
    candidates = {arm: profile.make_run(manifest, arm, profile.reference_parameters(
        learning_rate=1e-5 if arm.endswith("four") else 1e-4)) for arm in ("gvp_four", "gvp_six")}
    result = profile.refinement_parameters("gvp", candidates, repeated(candidates, zero=True, gap=True))
    assert result["diagnoses"] == ["boundary_learning_rate", "zero_native_recall"]
    assert result["variants"] == [{"learning_rate": 5e-6}, {"metal_class_weight_mode": "inverse_sqrt_frequency"}]
    assert result["parameters"]["gvp_four"][0]["learning_rate"] == 5e-6
    assert result["parameters"]["gvp_six"][0]["learning_rate"] == 3e-4
    assert result["maximum_fits_per_arm"] == 4
    for arm, variants in result["parameters"].items():
        for params in variants:
            profile.make_run(manifest, arm, params, block="refinement")


def test_opposite_lr_boundary_directions_fill_both_slots_without_another_diagnostic(manifest):
    candidates = {arm: profile.make_run(manifest, arm, profile.reference_parameters(
        learning_rate=1e-5 if arm.endswith("four") else 1e-4)) for arm in ("late_four", "late_six")}
    result = profile.refinement_parameters("late", candidates, repeated(candidates))
    assert result["variants"] == [{"learning_rate": 5e-6}, {"learning_rate": 2e-4}]
    assert [row["learning_rate"] for row in result["parameters"]["late_four"]] == [5e-6, 2e-4]
    assert [row["learning_rate"] for row in result["parameters"]["late_six"]] == [2e-4, 5e-6]


def test_forecast_distinguishes_capacity_edges_and_fold_membership(manifest):
    reference = profile.make_run(manifest, "late_four", profile.reference_parameters())
    smoke = {**reference, "epochs": 1, "elapsed_seconds": 85., "setup_seconds": 80., "epoch_seconds": 5.}
    assert profile.forecast(reference, [smoke]) == 330
    large = profile.make_run(manifest, "late_four", profile.reference_parameters(
        **profile.LARGE_CAPACITIES["large_256"], capacity="large_256"))
    with pytest.raises(ValueError, match="profile it first"):
        profile.forecast(large, [smoke])
    full = {**reference, "elapsed_seconds": 250., "setup_seconds": 80., "epoch_seconds": 3.4}
    assert profile.forecast(reference, [smoke, full]) == 250
    folded = profile.make_run(manifest, "late_four", reference["parameters"], fold_index=0)
    detail = profile.forecast_details(folded, [full])
    assert detail["fold_extrapolation"]
    assert detail["seconds"] == pytest.approx(80 + 50 * 3.4 * 4 / 3)
    on = profile.make_run(manifest, "gvp_ring_on", profile.reference_parameters(), ring=True)
    off = profile.make_run(manifest, "gvp_ring_off", profile.reference_parameters())
    assert profile.base.parse_config(on["command"]).ring_features_dir == str(
        Path(manifest["data_root"]) / "RING_features"
    )
    assert profile.base.parse_config(off["command"]).ring_features_dir is None
    with pytest.raises(ValueError, match="profile it first"):
        profile.forecast(on, [{**off, "elapsed_seconds": 270.}])


def test_extended_widths_and_bottlenecks_parse_without_adding_grid_cells(manifest):
    for name, widths in profile.LARGE_CAPACITIES.items():
        for arm in ("late_four", "late_six"):
            run = profile.make_run(manifest, arm, profile.reference_parameters(**widths, capacity=name), block="adaptive")
            assert profile.base.parse_config(run["command"]).hidden_s == widths["hidden_s"]
    for dimension in (8, 16, 32, 64, 128):
        for arm in ("early_four", "hybrid_four"):
            profile.make_run(manifest, arm, profile.reference_parameters(early_esm_dim=dimension), block="adaptive")
