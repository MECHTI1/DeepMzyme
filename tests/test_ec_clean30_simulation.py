"""Null-adoption sweeps: sweep files, calibration from certified memberships, records (synthetic data only)."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ec_clean30 import identity, simulation  # noqa: E402
from ec_clean30.campaign import CAMPAIGN_ID  # noqa: E402
from ec_clean30.provenance import DirtyTreeError, HeldOutInputError, sha256_file  # noqa: E402
from ec_clean30.selection import NullNoiseModel, _simulated_fit, _simulated_population  # noqa: E402
from ec_clean30.simulation import (  # noqa: E402
    ALLOWED_LOOKS, EXIT_STOPPED, OVERLAY_MANIFEST, STATUS_COMPLETE, STATUS_PARTIAL, STATUS_STOPPED, SUMMARY_COLUMNS,
    SWEEP_FORMAT, FoldCalibration, SimulationError, challenger_shift_for_gain, check_calibration_folder,
    expected_ba_points, fit_shift_sd_for_seed_spread, is_campaign_identity_path, load_fold_calibration,
    parse_calibrate_from, parse_sweep, read_sweep, run_simulate_action, run_sweep, seed_spread_points,
)
import test_ec_clean30_identity as identity_fixture  # noqa: E402  (synthetic identity chain)

RULE = {"n_draws": 200, "stratified": True}
CLEAN_GIT = {"head": "0" * 40, "branch": "main", "clean": True, "status_entries": 0, "status_sample": []}


@pytest.fixture(autouse=True)
def few_spread_populations(monkeypatch):
    monkeypatch.setattr(simulation, "SEED_SPREAD_POPULATIONS", 40)


def synthetic_membership(n=240, seed=3, mean_size=1.6, purity=0.9):
    model = NullNoiseModel(n_proteins=n, mean_component_size=mean_size, component_class_purity=purity)
    true, component, _difficulty = _simulated_population(np.random.default_rng(seed), model)
    rows = [{"accession": f"Q{k:05d}", "group_id": f"c{group}", "ec_class": str(cls + 1)}
            for k, (cls, group) in enumerate(zip(true.tolist(), component.tolist()))]
    return rows, {row["accession"]: row["group_id"] for row in rows}


def synthetic_calibration(fold, **kwargs):
    rows, components = synthetic_membership(**kwargs)
    inputs = {"membership_sha256": f"{fold}" * 64, "certificate_sha256": "c" * 64}
    return FoldCalibration.from_membership(fold, rows, components, source=f"synthetic/fold{fold}", inputs=inputs)


@pytest.fixture
def calibrated(monkeypatch):
    calibrations = {fold: synthetic_calibration(fold, seed=fold + 3) for fold in range(5)}
    monkeypatch.setattr(simulation, "load_fold_calibration", lambda fold, work, smoke=False: calibrations[fold])
    return calibrations


def sweep_document(**changes):
    document = {
        "format": SWEEP_FORMAT,
        "sweep_id": "unit_sweep",
        "n_replicates": 3,
        "rules": {"current": dict(RULE), "alpha_0.05": dict(RULE, alpha=0.05)},
        "noise_base": {"fit_shift_sd": 0.15},
        "noise_points": {"central": {}, "spread_hi": {"fit_shift_sd": 0.3, "fit_noise_sd": 1.2}},
        "groups": [
            {"group": "null", "folds": [2], "rules": ["current", "alpha_0.05"], "looks": [1, 4],
             "noise_points": ["central"]},
            {"group": "power", "folds": [2], "rules": ["current"], "looks": [1], "noise_points": ["central"],
             "true_gain_points": [3.0]},
        ],
    }
    document.update(changes)
    return document


def write_sweep(path, document=None):
    path.write_text(json.dumps(document or sweep_document(), indent=1))
    return path


CALIBRATE = ["2=/nowhere/overlay/identity/fold2"]
POINT_IDS = [
    "null__fold2__current__looks1__central", "null__fold2__current__looks4__central",
    "null__fold2__alpha_0.05__looks1__central", "null__fold2__alpha_0.05__looks4__central",
    "power__fold2__current__looks1__central__gain3",
]


def read_records(out):
    return {path.stem: json.loads(path.read_text()) for path in sorted((out / "records").glob("*.json"))}


# ---------------------------------------------------------------------------
# Sweep files


def test_sweep_expands_groups_in_order_and_accepts_every_look_count():
    spec = parse_sweep(sweep_document())
    assert [point.point_id for point in spec.points] == POINT_IDS
    assert [point.kind for point in spec.points] == ["null"] * 4 + ["power"]
    assert spec.points[-1].true_gain_points == 3.0 and spec.points[0].n_replicates == 3
    assert spec.config("alpha_0.05").alpha == 0.05 and spec.config("current").n_draws == 200
    assert spec.noise["central"] == {"fit_shift_sd": 0.15}
    assert spec.noise["spread_hi"] == {"fit_shift_sd": 0.3, "fit_noise_sd": 1.2}
    assert spec.simulation_seed == 20261024 and spec.spec_sha256 == parse_sweep(sweep_document()).spec_sha256
    assert ALLOWED_LOOKS == (1, 2, 4)
    groups = [{"group": "looks", "folds": [0, 4], "rules": ["current"], "looks": [1, 2, 4], "noise_points": ["central"],
               "n_replicates": 7}]
    spec = parse_sweep(sweep_document(groups=groups))
    assert [(point.fold, point.looks, point.n_replicates) for point in spec.points] == [
        (0, 1, 7), (0, 2, 7), (0, 4, 7), (4, 1, 7), (4, 2, 7), (4, 4, 7)]


def test_noise_points_replace_the_matching_base_settings():
    base = {"seed_spread_points": 1.5, "class_recall_offset": 0.05, "protein_sd": 0.5}
    points = {"central": {}, "direct": {"fit_shift_sd": 0.2, "class_recall": [0.5] * 7}}
    spec = parse_sweep(sweep_document(noise_base=base, noise_points=points,
                                      groups=[dict(sweep_document()["groups"][0], noise_points=["central", "direct"])]))
    assert spec.noise["central"] == base
    assert spec.noise["direct"] == {"protein_sd": 0.5, "fit_shift_sd": 0.2, "class_recall": [0.5] * 7}


def _set_rule(document, value):
    document["rules"]["current"] = value


def _set_noise(document, value):
    document["noise_points"]["central"] = value


def _set_group(document, key, value):
    document["groups"][0][key] = value


BAD_SWEEPS = [
    (lambda d: d.update(format="other"), "format must be"),
    (lambda d: d.update(extra=1), "unknown keys"),
    (lambda d: d.pop("groups"), "missing keys"),
    (lambda d: d.update(groups=[]), "non-empty list"),
    (lambda d: d.update(n_replicates=True), "integer of at least 1"),
    (lambda d: d.update(sweep_id="has__double"), "no '__'"),
    (lambda d: d.update(sweep_id="test"), "names a test split"),
    (lambda d: _set_rule(d, {"alpha2": 0.1}), "unknown PromotionConfig fields"),
    (lambda d: _set_rule(d, {"alpha": 1.5}), "alpha must lie"),
    (lambda d: _set_rule(d, {"stratified": "yes"}), "true or false"),
    (lambda d: _set_rule(d, {"n_draws": 0}), "integer of at least 1"),
    (lambda d: _set_rule(d, {"zero_recall_unit": "per_fold"}), "zero_recall_unit"),
    (lambda d: _set_noise(d, {"n_proteins": 500}), "certified membership"),
    (lambda d: _set_noise(d, {"class_proportions": [1 / 7] * 7}), "certified membership"),
    (lambda d: _set_noise(d, {"challenger_shift": 1.0}), "unknown keys"),
    (lambda d: _set_noise(d, {"fit_shift_sd": 0.2, "seed_spread_points": 1.0}), "sets both"),
    (lambda d: _set_noise(d, {"class_recall": [0.5] * 7, "class_recall_offset": 0.1}), "sets both"),
    (lambda d: _set_noise(d, {"protein_sd": -0.1}), "at least 0"),
    (lambda d: _set_noise(d, {"class_recall_offset": 0.3}), "strictly between"),
    (lambda d: _set_noise(d, {"class_recall": [0.5] * 6}), "one recall per EC class"),
    (lambda d: _set_noise(d, {"seed_spread_points": 0.0}), "must be positive"),
    (lambda d: _set_group(d, "looks", [3]), "looks must be among"),
    (lambda d: _set_group(d, "looks", [1, 1]), "more than once"),
    (lambda d: _set_group(d, "folds", [5]), "folds must lie"),
    (lambda d: _set_rule(d, {"min_gain_points": -5}), "at least 0"),
    (lambda d: _set_rule(d, {"max_skip_fraction": 2.0}), "at most 1"),
    (lambda d: _set_rule(d, {"guard_recall_drop_points": -1.0}), "at least 0"),
    (lambda d: _set_rule(d, {"guard_net_proteins": -0.5}), "at least 0"),
    (lambda d: _set_group(d, "rules", ["missing"]), "unknown rules"),
    (lambda d: _set_group(d, "noise_points", ["missing"]), "unknown noise points"),
    (lambda d: _set_group(d, "rules", [["current"]]), "rules must be names"),
    (lambda d: _set_group(d, "noise_points", [{"central": {}}]), "noise points must be names"),
    (lambda d: _set_group(d, "true_gain_points", [-1.0]), "must be positive"),
    (lambda d: _set_group(d, "seeds", [1]), "unknown keys"),
    (lambda d: d["groups"].append(dict(d["groups"][0])), "defined more than once"),
]


@pytest.mark.parametrize("mutate, message", BAD_SWEEPS)
def test_bad_sweeps_are_refused(mutate, message):
    document = sweep_document()
    mutate(document)
    with pytest.raises(SimulationError, match=message):
        parse_sweep(document)


def test_a_sweep_file_must_be_json(tmp_path):
    path = tmp_path / "sweep.json"
    path.write_text("{not json")
    with pytest.raises(SimulationError, match="not UTF-8 JSON"):
        read_sweep(path)
    spec, data = read_sweep(write_sweep(path))
    assert data == path.read_bytes() and len(spec.points) == 5


# ---------------------------------------------------------------------------
# Conversions on a fold's population


def test_true_gain_conversion_matches_the_model():
    model = synthetic_calibration(0).noise_model
    shift = challenger_shift_for_gain(model, 2.0)
    assert expected_ba_points(model, shift) - expected_ba_points(model) == pytest.approx(2.0, abs=1e-6)
    rng = np.random.default_rng(5)
    scale = np.sqrt(model.component_sd ** 2 + model.protein_sd ** 2 + model.fit_shift_sd ** 2 + model.fit_noise_sd ** 2)
    draws = rng.normal(0.0, scale, 400_000)
    recall = [model.class_recall[c] for c in range(6) if model.class_proportions[c] > 0]
    sampled = np.mean([np.mean(1 / (1 + np.exp(-(np.log(r / (1 - r)) + draws + shift)))) for r in recall])
    assert 100 * sampled == pytest.approx(expected_ba_points(model, shift), abs=0.1)
    # The simulator's own fits on one large population show the same expected gain.
    big = NullNoiseModel(n_proteins=3000, mean_component_size=1.0, fit_shift_sd=0.0)
    rng = np.random.default_rng(6)
    true, _component, difficulty = _simulated_population(rng, big)
    big_shift = challenger_shift_for_gain(big, 2.0)

    def mean_ba(value):
        fits = [_simulated_fit(rng, big, true, difficulty, value) for _ in range(100)]
        return 100 * np.mean([np.mean([np.mean(fit[true == c] == c) for c in range(6)]) for fit in fits])

    assert mean_ba(big_shift) - mean_ba(0.0) == pytest.approx(2.0, abs=0.5)
    with pytest.raises(SimulationError, match="exceeds"):
        challenger_shift_for_gain(model, 80.0)
    with pytest.raises(SimulationError, match="positive"):
        challenger_shift_for_gain(model, 0.0)


def test_seed_spread_conversion_inverts_the_implied_spread():
    model = synthetic_calibration(0).noise_model
    target = seed_spread_points(replace(model, fit_shift_sd=0.4), n_populations=200)
    solved = fit_shift_sd_for_seed_spread(model, target, n_populations=200)
    assert solved["fit_shift_sd"] == pytest.approx(0.4, abs=0.05)
    assert solved["achieved_points"] == pytest.approx(target, rel=0.02)
    assert solved["floor_points"] < target and solved["n_populations"] == 200
    with pytest.raises(SimulationError, match="per-protein noise"):
        fit_shift_sd_for_seed_spread(model, solved["floor_points"] / 2, n_populations=200)
    with pytest.raises(SimulationError, match="exceeds"):
        fit_shift_sd_for_seed_spread(model, 1000.0, n_populations=200)


# ---------------------------------------------------------------------------
# Calibration folders


def test_calibrate_from_maps_each_outer_fold_to_one_folder():
    assert parse_calibrate_from(["2=/a/b", "0=rel/c"]) == {2: Path("/a/b"), 0: Path("rel/c")}
    assert parse_calibrate_from({4: "/x"}) == {4: Path("/x")}
    for values, message in ((["2"], "K=PATH"), (["x=/p"], "K=PATH"), (["2="], "K=PATH"), (["-1=/p"], "K=PATH"),
                            (["5=/p"], "0..4"), (["2=/a", "2=/b"], "more than once"), ({True: "/p"}, "0..4")):
        with pytest.raises(SimulationError, match=message):
            parse_calibrate_from(values)


def _overlay_tree(tmp_path):
    root = tmp_path / CAMPAIGN_ID
    overlay = root / "overlay" / "ecc30-overlay-v1"
    (overlay / "identity").mkdir(parents=True)
    (overlay / OVERLAY_MANIFEST).write_text("{}\n")
    return root, overlay


def test_campaign_identity_folder_is_refused(tmp_path):
    root, overlay = _overlay_tree(tmp_path)
    for name in ("fold2", "fold2_cap1-3"):
        (root / "identity" / name).mkdir(parents=True)
        with pytest.raises(SimulationError, match="campaign's identity folder"):
            check_calibration_folder(2, root / "identity" / name, require_overlay=False)
    with pytest.raises(SimulationError, match="campaign's identity folder"):
        check_calibration_folder(2, overlay.parent / ".." / "identity" / "fold2", require_overlay=False)
    (overlay / "identity" / "fold3").symlink_to(root / "identity" / "fold2", target_is_directory=True)
    assert is_campaign_identity_path(overlay / "identity" / "fold3")
    with pytest.raises(SimulationError, match="campaign's identity folder"):
        check_calibration_folder(3, overlay / "identity" / "fold3", require_overlay=False)
    linked = overlay / "identity" / "fold4"
    linked.mkdir()
    (tmp_path / "elsewhere.json").write_text("{}")
    (linked / identity.INNER_CERTIFICATE).symlink_to(tmp_path / "elsewhere.json")
    with pytest.raises(SimulationError, match="is a link"):
        check_calibration_folder(4, linked, require_overlay=False)
    copy = overlay / "identity" / "fold2"
    copy.mkdir()
    assert not is_campaign_identity_path(copy)
    assert check_calibration_folder(2, copy, require_overlay=True) == overlay / OVERLAY_MANIFEST
    plain = tmp_path / "plain" / "fold2"
    plain.mkdir(parents=True)
    assert check_calibration_folder(2, plain, require_overlay=False) is None
    with pytest.raises(SimulationError, match="a real run reads the data overlay's copy"):
        check_calibration_folder(2, plain, require_overlay=True)
    with pytest.raises(HeldOutInputError):
        check_calibration_folder(2, tmp_path / "plain" / "fold2_test", require_overlay=False)


def test_simulate_refuses_the_campaign_identity_folder_before_writing(tmp_path):
    root, _overlay = _overlay_tree(tmp_path)
    (root / "identity" / "fold2").mkdir(parents=True)
    out = tmp_path / "out"
    with pytest.raises(SimulationError, match="campaign's identity folder"):
        run_sweep(write_sweep(tmp_path / "sweep.json"), out, [f"2={root / 'identity' / 'fold2'}"], smoke=True)
    assert not out.exists()


@pytest.fixture
def certified_chain(tmp_path, monkeypatch):
    """A pinned synthetic identity chain of fold 0 (24 eligible proteins in 18 components), copied into an overlay."""
    binary = identity_fixture.fake_binary(tmp_path / "fakebin")
    monkeypatch.setattr(identity, "MMSEQS_BINARY_SHA256", sha256_file(binary))
    dataset = identity_fixture.write_inner_dataset(tmp_path / "chain_data")
    work, _fake = identity_fixture.run_metallo_pipeline(tmp_path, "work", dataset, hits=identity_fixture.INNER_HITS,
                                                        allow_unpinned=False)
    identity.assign_inner(work, 0, metallo_folds_dir=dataset[1])
    identity.certify_inner(work, 0, metallo_folds_dir=dataset[1])
    _root, overlay = _overlay_tree(tmp_path)
    copy = overlay / "identity" / "fold0"
    shutil.copytree(work, copy)
    return work, copy, overlay


def test_calibration_from_a_certified_identity_chain(certified_chain):
    work, copy, overlay = certified_chain
    calibration = load_fold_calibration(0, copy)
    assert calibration.statistics["n_proteins"] == 24 and calibration.statistics["n_components"] == 18
    assert calibration.noise_model.mean_component_size == pytest.approx(24 / 18)
    assert calibration.inputs["certificate_sha256"] == sha256_file(work / identity.INNER_CERTIFICATE)
    receipt = json.loads((work / identity.INNER_ASSIGNMENT_RECEIPT).read_text())
    assert calibration.inputs["membership_sha256"] == receipt["outputs"]["membership"]["sha256"]
    assert calibration.inputs["overlay_manifest_sha256"] == sha256_file(overlay / OVERLAY_MANIFEST)
    assert calibration.public_record() == {"fold": 0, "n_proteins": 24,
                                           "membership_sha256": calibration.inputs["membership_sha256"],
                                           "certificate_sha256": calibration.inputs["certificate_sha256"]}
    with pytest.raises(SimulationError, match="fold"):
        load_fold_calibration(1, copy)
    uncertified = overlay / "identity" / "fold0_uncertified"
    shutil.copytree(work, uncertified)
    (uncertified / identity.INNER_CERTIFICATE).unlink()
    with pytest.raises(SimulationError, match="certify-inner"):
        load_fold_calibration(0, uncertified)


def _tiny_document(fold=0):
    return sweep_document(n_replicates=2, rules={"current": {"n_draws": 50, "stratified": True}},
                          groups=[{"group": "null", "folds": [fold], "rules": ["current"], "looks": [1, 2],
                                   "noise_points": ["central"]}])


def test_end_to_end_on_the_overlay_copy_prints_no_class_proportions(tmp_path, certified_chain):
    work, copy, _overlay = certified_chain
    sweep = write_sweep(tmp_path / "sweep.json", _tiny_document())
    # A real run refuses the certified folder outside the overlay before writing anything.
    with pytest.raises(SimulationError, match="a real run reads the data overlay's copy"):
        run_sweep(sweep, tmp_path / "real", [f"0={work}"], git=CLEAN_GIT)
    assert not (tmp_path / "real").exists()
    payload = run_sweep(sweep, tmp_path / "real", [f"0={copy}"], git=CLEAN_GIT)
    assert payload["status"] == STATUS_COMPLETE and payload["calibrations"]["0"]["n_proteins"] == 24
    proportions = load_fold_calibration(0, copy).statistics["class_proportions"]
    printed = json.dumps(payload)
    public = [printed, (tmp_path / "real" / "summary.tsv").read_text(), (tmp_path / "real" / "sweep_receipt.json").read_text()]
    for text in public:
        assert "class_proportions" not in text
        assert all(repr(value) not in text for value in proportions if value)
    record = next(iter(read_records(tmp_path / "real").values()))
    assert record["smoke"] is False and record["result"]["noise_model"]["class_proportions"] == list(proportions)
    with pytest.raises(SimulationError, match="smoke and real records mixed"):
        run_sweep(sweep, tmp_path / "real", [f"0={copy}"], smoke=True)


# ---------------------------------------------------------------------------
# Records, summary and continuation


def test_sweep_records_are_deterministic_and_written_once(tmp_path, calibrated):
    sweep = write_sweep(tmp_path / "sweep.json")
    first = run_sweep(sweep, tmp_path / "out1", CALIBRATE, smoke=True)
    second = run_sweep(sweep, tmp_path / "out2", CALIBRATE, smoke=True)
    assert first["status"] == second["status"] == STATUS_COMPLETE
    assert [item["point_id"] for item in first["points"]] == POINT_IDS
    assert (tmp_path / "out1" / "summary.tsv").read_bytes() == (tmp_path / "out2" / "summary.tsv").read_bytes()
    assert first["summary"] == second["summary"]
    records_1, records_2 = read_records(tmp_path / "out1"), read_records(tmp_path / "out2")
    assert sorted(records_1) == sorted(POINT_IDS)
    assert {key: value["result_sha256"] for key, value in records_1.items()} == \
        {key: value["result_sha256"] for key, value in records_2.items()}
    record = records_1[POINT_IDS[1]]
    assert record["result"]["looks"] == 4 and record["result"]["promotion_config"]["n_draws"] == 200
    # The simulator itself ran four looks (its own record, not the copy of the point).
    assert record["result"]["operating_characteristics"]["looks"] == 4
    assert records_1[POINT_IDS[0]]["result"]["operating_characteristics"]["looks"] == 1
    assert record["result"]["calibration"]["inputs"]["membership_sha256"] == "2" * 64
    assert record["result"]["noise_model"]["n_proteins"] == 240 and record["smoke"] is True
    assert "src/ec_clean30/simulation.py" in record["provenance"]["code_sha256"]
    assert set(record["timing"]) == {"started_utc", "cpu_seconds", "wall_seconds"}
    assert (tmp_path / "out1" / "sweep.json").read_bytes() == sweep.read_bytes()
    header = (tmp_path / "out1" / "summary.tsv").read_text().splitlines()[0].split("\t")
    assert tuple(header) == SUMMARY_COLUMNS
    receipt = json.loads((tmp_path / "out1" / "sweep_receipt.json").read_text())
    assert sorted(receipt["records"]) == sorted(POINT_IDS) and receipt["summary"] == first["summary"]
    again = run_sweep(sweep, tmp_path / "out1", CALIBRATE, smoke=True)
    assert again["status"] == STATUS_COMPLETE and {item["status"] for item in again["points"]} == {"existing"}
    with pytest.raises(FileExistsError):
        simulation.write_new_json(tmp_path / "out1" / "records" / f"{POINT_IDS[0]}.json", {})


def test_the_sweep_seed_reaches_the_simulator(tmp_path, calibrated):
    groups = [{"group": "null", "folds": [2], "rules": ["current"], "looks": [1], "noise_points": ["central"]}]
    results = {}
    for seed in (20261024, 7):
        sweep = write_sweep(tmp_path / f"sweep{seed}.json", sweep_document(groups=groups, simulation_seed=seed))
        run_sweep(sweep, tmp_path / f"out{seed}", CALIBRATE, smoke=True)
        record = read_records(tmp_path / f"out{seed}")["null__fold2__current__looks1__central"]
        assert record["result"]["operating_characteristics"]["seed"] == seed == record["result"]["simulation_seed"]
        assert record["result"]["implied"]["seed_spread_seed"][0] == seed
        results[seed] = record
    assert results[7]["result_sha256"] != results[20261024]["result_sha256"]
    gains = [results[seed]["result"]["operating_characteristics"]["gain_points_mean"] for seed in results]
    assert gains[0] != gains[1]


def test_every_call_names_every_sweep_fold_and_the_receipt_lists_them(tmp_path, calibrated):
    groups = [{"group": "null", "folds": [0, 2], "rules": ["current"], "looks": [1], "noise_points": ["central"]}]
    sweep = write_sweep(tmp_path / "sweep.json", sweep_document(groups=groups))
    both = ["0=/nowhere/overlay/identity/fold0", *CALIBRATE]
    out = tmp_path / "out"
    with pytest.raises(SimulationError, match=r"uses folds \[0\] without a --calibrate-from"):
        run_sweep(sweep, out, CALIBRATE, points=["null__fold2__current__looks1__central"], smoke=True)
    with pytest.raises(SimulationError, match=r"names folds \[3\] that no point of the sweep uses"):
        run_sweep(sweep, out, [*both, "3=/nowhere/overlay/identity/fold3"], smoke=True)
    assert not out.exists()
    first = run_sweep(sweep, out, both, points=["null__fold0__current__looks1__central"], smoke=True)
    assert first["status"] == STATUS_PARTIAL and sorted(first["calibrations"]) == ["0", "2"]
    second = run_sweep(sweep, out, both, smoke=True)
    assert second["status"] == STATUS_COMPLETE
    receipt = json.loads((out / "sweep_receipt.json").read_text())
    assert sorted(receipt["calibrations"]) == ["0", "2"]
    assert receipt["calibrations"]["0"]["membership_sha256"] == "0" * 64
    # A fold-0 record is checked against fold 0's calibration even when only a fold-2 point is selected.
    calibrated[0] = synthetic_calibration(0, seed=99)
    with pytest.raises(SimulationError, match="another calibration"):
        run_sweep(sweep, out, both, points=["null__fold2__current__looks1__central"], smoke=True)


@pytest.mark.parametrize("bad", ["seed_spread", "true_gain"])
def test_every_point_is_resolved_before_anything_is_written(tmp_path, calibrated, bad):
    floor = seed_spread_points(replace(calibrated[2].noise_model, fit_shift_sd=0.0), n_populations=40)
    noise = {"central": {}, "too_low": {"seed_spread_points": floor / 4}}
    groups = [{"group": "null", "folds": [2], "rules": ["current"], "looks": [1], "noise_points": ["central"]}]
    if bad == "seed_spread":
        groups.append(dict(groups[0], group="low", noise_points=["too_low"]))
        message = "per-protein noise gives alone"
    else:
        groups.append(dict(groups[0], group="power", true_gain_points=[90.0]))
        message = "exceeds the"
    sweep = write_sweep(tmp_path / "sweep.json", sweep_document(noise_points=noise, groups=groups))
    out = tmp_path / "out"
    with pytest.raises(SimulationError, match=message):
        run_sweep(sweep, out, CALIBRATE, points=["null__fold2__current__looks1__central"], smoke=True)
    assert not out.exists()


def test_out_may_not_lie_in_the_identity_folder_a_calibration_folder_or_an_overlay(tmp_path, calibrated):
    root, overlay = _overlay_tree(tmp_path)
    sweep = write_sweep(tmp_path / "sweep.json")
    (tmp_path / "linked").symlink_to(overlay, target_is_directory=True)
    (tmp_path / "linked_out").symlink_to(overlay / "identity", target_is_directory=True)
    calibration = overlay / "identity" / "fold2"
    cases = [
        (root / "identity" / "sim", CALIBRATE, "campaign's identity folder"),
        (overlay / "sim", CALIBRATE, "inside a data overlay"),
        (overlay, CALIBRATE, "inside a data overlay"),
        (tmp_path / "linked" / "sim", CALIBRATE, "inside a data overlay"),
        (tmp_path / "linked_out", CALIBRATE, "inside a data overlay"),  # --out itself links into the overlay
        (calibration / "sim", [f"2={calibration}"], "inside the calibration folder of fold 2"),
    ]
    for out, calibrate, message in cases:
        with pytest.raises(SimulationError, match=message):
            run_sweep(sweep, out, calibrate, smoke=True)
    assert not (root / "identity").exists()
    assert sorted(path.relative_to(overlay).as_posix() for path in overlay.rglob("*")) == ["identity", OVERLAY_MANIFEST]


def test_a_damaged_record_is_refused_not_a_traceback(tmp_path, calibrated):
    sweep = write_sweep(tmp_path / "sweep.json")
    out = tmp_path / "out"
    run_sweep(sweep, out, CALIBRATE, points=[POINT_IDS[0]], smoke=True)
    path = out / "records" / f"{POINT_IDS[0]}.json"
    record = json.loads(path.read_text())
    no_timing = {key: value for key, value in record.items() if key != "timing"}
    other_point = dict(record, point=dict(record["point"], looks=2))
    for content, message in ((b"{broken", "not a readable record"), (b"\xff\xfe{}", "not a readable record"),
                             (b"[1]\n", "not a simulation record"),
                             (json.dumps(no_timing).encode(), "not a simulation record"),
                             (json.dumps(dict(record, result=[])).encode(), "not a simulation record"),
                             (json.dumps(other_point).encode(), "another point definition")):
        path.write_bytes(content)
        with pytest.raises(SimulationError, match=message):
            run_sweep(sweep, out, CALIBRATE, smoke=True)
    assert sorted(read_records(out)) == [POINT_IDS[0]]


def test_power_point_adopts_more_than_its_null_twin(tmp_path, calibrated):
    groups = [{"group": "null", "folds": [2], "rules": ["current"], "looks": [1], "noise_points": ["central"]},
              {"group": "power", "folds": [2], "rules": ["current"], "looks": [1], "noise_points": ["central"],
               "true_gain_points": [6.0]}]
    sweep = write_sweep(tmp_path / "sweep.json", sweep_document(n_replicates=8, groups=groups))
    run_sweep(sweep, tmp_path / "out", CALIBRATE, smoke=True)
    records = read_records(tmp_path / "out")
    null = records["null__fold2__current__looks1__central"]["result"]
    power = records["power__fold2__current__looks1__central__gain6"]["result"]
    assert null["noise_model"]["challenger_shift"] == 0.0 < power["noise_model"]["challenger_shift"]
    assert power["implied"]["expected_gain_points"] == pytest.approx(6.0, abs=1e-6)
    assert power["implied"]["seed_spread_points"] == null["implied"]["seed_spread_points"]
    assert power["operating_characteristics"]["adoption_rate"] > null["operating_characteristics"]["adoption_rate"]
    assert power["operating_characteristics"]["gain_points_mean"] > null["operating_characteristics"]["gain_points_mean"]


def test_seed_spread_noise_points_are_solved_on_the_fold_population(tmp_path, calibrated):
    target = 1.2 * seed_spread_points(replace(calibrated[2].noise_model, fit_shift_sd=0.0), n_populations=40)
    document = sweep_document(noise_points={"central": {"seed_spread_points": target}},
                              groups=[{"group": "null", "folds": [2], "rules": ["current"], "looks": [1],
                                       "noise_points": ["central"]}])
    run_sweep(write_sweep(tmp_path / "sweep.json", document), tmp_path / "out", CALIBRATE, smoke=True)
    result = next(iter(read_records(tmp_path / "out").values()))["result"]
    solved = result["resolution"]["fit_shift_sd_from_seed_spread"]
    assert solved["target_points"] == target and result["noise_model"]["fit_shift_sd"] == solved["fit_shift_sd"] > 0
    assert result["implied"]["seed_spread_points"] == pytest.approx(target, rel=0.05)
    row = (tmp_path / "out" / "summary.tsv").read_text().splitlines()[1].split("\t")
    assert float(row[SUMMARY_COLUMNS.index("seed_spread_target_points")]) == pytest.approx(target)


def test_partial_runs_continue_and_changed_inputs_are_refused(tmp_path, calibrated, monkeypatch):
    sweep = write_sweep(tmp_path / "sweep.json")
    out = tmp_path / "out"
    first = run_sweep(sweep, out, CALIBRATE, points=[POINT_IDS[0]], smoke=True)
    assert first["status"] == STATUS_PARTIAL and first["points_recorded"] == 1 and first["summary"] is None
    assert first["resolution_cpu_seconds"] >= 0.0
    assert not (out / "summary.tsv").exists()
    assert first["forecast"]["remaining_points"] == 4 and first["forecast"]["cpu_seconds_per_replicate"] > 0
    assert first["forecast"]["forecast_total_cpu_hours"] < 8.0
    second = run_sweep(sweep, out, CALIBRATE, smoke=True)
    assert second["status"] == STATUS_COMPLETE
    assert [item["status"] for item in second["points"]] == ["existing"] + ["written"] * 4
    with pytest.raises(SimulationError, match="does not define"):
        run_sweep(sweep, out, CALIBRATE, points=["nope"], smoke=True)
    changed = write_sweep(tmp_path / "changed.json", sweep_document(n_replicates=4))
    with pytest.raises(SimulationError, match="holds another sweep file"):
        run_sweep(changed, out, CALIBRATE, smoke=True)
    original = simulation.provenance_record

    def other_code(**kwargs):
        record = original(**kwargs)
        return dict(record, code_sha256=dict(record["code_sha256"], **{"src/ec_clean30/selection.py": "f" * 64}))

    monkeypatch.setattr(simulation, "provenance_record", other_code)
    with pytest.raises(SimulationError, match="other campaign code"):
        run_sweep(sweep, out, CALIBRATE, smoke=True)
    monkeypatch.setattr(simulation, "provenance_record", original)
    path = out / "records" / f"{POINT_IDS[2]}.json"
    record = json.loads(path.read_text())
    record["result"]["operating_characteristics"]["n_adopted"] += 1
    path.write_text(json.dumps(record))
    with pytest.raises(SimulationError, match="no longer matches its hash"):
        run_sweep(sweep, out, CALIBRATE, smoke=True)
    assert run_sweep(sweep, tmp_path / "other", CALIBRATE, smoke=True)["status"] == STATUS_COMPLETE
    calibrated[2] = synthetic_calibration(2, seed=99)
    with pytest.raises(SimulationError, match="another calibration"):
        run_sweep(sweep, tmp_path / "other", CALIBRATE, smoke=True)


def test_forecast_limit_stops_before_the_next_point(tmp_path, calibrated):
    sweep = write_sweep(tmp_path / "sweep.json")
    args = SimpleNamespace(sweep=str(sweep), out=str(tmp_path / "out"), calibrate_from=CALIBRATE, points=None,
                           smoke=True, max_forecast_cpu_hours=1e-9)
    code, payload = run_simulate_action(args)
    assert (code, payload["status"], payload["points_recorded"]) == (EXIT_STOPPED, STATUS_STOPPED, 1)
    assert payload["forecast"]["over_limit"] is True
    code, payload = run_simulate_action(SimpleNamespace(**{**vars(args), "max_forecast_cpu_hours": 8.0}))
    assert (code, payload["status"]) == (0, STATUS_COMPLETE)
    assert [item["status"] for item in payload["points"]] == ["existing"] + ["written"] * 4
    code, payload = run_simulate_action(SimpleNamespace(**{**vars(args), "out": str(tmp_path / "one"),
                                                           "points": f" {POINT_IDS[4]} ,", "max_forecast_cpu_hours": 8.0}))
    assert (code, payload["status"], payload["points_selected"]) == (0, STATUS_PARTIAL, 1)
    with pytest.raises(SimulationError, match="must be positive"):
        run_sweep(sweep, tmp_path / "zero", CALIBRATE, smoke=True, max_forecast_cpu_hours=0.0)


def test_real_runs_need_a_clean_tree_and_untested_paths(tmp_path, calibrated, monkeypatch):
    sweep = write_sweep(tmp_path / "sweep.json")

    def dirty(*, smoke):
        if not smoke:
            raise DirtyTreeError("uncommitted entries")
        return dict(CLEAN_GIT, clean=False)

    monkeypatch.setattr(simulation, "real_action_git_state", dirty)
    with pytest.raises(DirtyTreeError):
        run_sweep(sweep, tmp_path / "out", CALIBRATE)
    assert not (tmp_path / "out").exists()
    with pytest.raises(HeldOutInputError):
        run_sweep(sweep, tmp_path / "out_test", CALIBRATE, smoke=True)
    test_named = write_sweep(tmp_path / "sweep_test.json")
    with pytest.raises(HeldOutInputError):
        run_sweep(test_named, tmp_path / "out", CALIBRATE, smoke=True)
    with pytest.raises(SimulationError, match="without a --calibrate-from"):
        run_sweep(sweep, tmp_path / "out", ["1=/nowhere/overlay/identity/fold1"], smoke=True)
    with pytest.raises(SimulationError, match="inside the calibration folder"):
        run_sweep(sweep, tmp_path / "cal" / "out", [f"2={tmp_path / 'cal'}"], smoke=True)


def test_simulation_module_does_not_import_torch():
    code = (f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r}); import ec_clean30.simulation; "
            "print('torch' in sys.modules)")
    output = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert output.strip() == "False"
