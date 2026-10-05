"""Independent-replay probability tolerance: strict default, explicit v3 policy (pmm-core-replay-v1 value)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from training import campaign_runtime as runtime  # noqa: E402
from training.campaign_runtime import CampaignContractError  # noqa: E402

V3_POLICY = {"policy_id": "pmm-v3-replay-1", "probability_atol": 1e-5,
             "probability_atol_source": "pmm-core-replay-v1"}


def rows(changes=None):
    base = {"u1": {"source_uid": "u1", "physical_ion_id": "i1", "y_native": "1", "pred_native": "1",
                   "y_common4": "1", "pred_common4": "1", "p_native_Mn": "0.10000000", "p_common4_Cu": "0.90000000"},
            "u2": {"source_uid": "u2", "physical_ion_id": "i2", "y_native": "0", "pred_native": "0",
                   "y_common4": "0", "pred_common4": "0", "p_native_Mn": "0.80000000", "p_common4_Cu": "0.20000000"}}
    for (uid, key), value in (changes or {}).items():
        base[uid][key] = value
    return base


def _change(uid, key, value):
    return {(uid, key): value}


@pytest.mark.parametrize("delta, strict_ok, v3_ok", [
    (0.0, True, True), (9e-7, True, True), (5e-6, False, True), (1e-5, False, True), (1.001e-5, False, False),
    (1.2e-5, False, False)])
def test_probability_tolerance_strict_versus_v3(delta, strict_ok, v3_ok):
    saved = rows()
    replayed = rows(_change("u2", "p_common4_Cu", f"{0.2 + delta:.8f}"))
    for kwargs, ok in (({"probability_atol": runtime.STRICT_REPLAY_PROBABILITY_ATOL}, strict_ok),
                       ({"probability_atol": V3_POLICY["probability_atol"], "exact_decimal": True}, v3_ok)):
        if ok:
            largest = runtime.compare_replayed_predictions(saved, replayed, **kwargs)
            assert largest == pytest.approx(delta, abs=1e-12)
        else:
            with pytest.raises(CampaignContractError, match="u2/p_common4_Cu"):
                runtime.compare_replayed_predictions(saved, replayed, **kwargs)


def test_exact_decimal_accepts_the_boundary_that_binary_floats_reject():
    saved = rows(_change("u2", "p_common4_Cu", "0.00003000"))
    replayed = rows(_change("u2", "p_common4_Cu", "0.00004000"))
    assert abs(float("0.00004000") - float("0.00003000")) > 1e-5  # binary rounding crosses the boundary
    with pytest.raises(CampaignContractError):
        runtime.compare_replayed_predictions(saved, replayed, probability_atol=1e-5)
    assert runtime.compare_replayed_predictions(saved, replayed, probability_atol=1e-5, exact_decimal=True) == 1e-5


@pytest.mark.parametrize("key, value", [("pred_common4", "0"), ("pred_native", "0"), ("y_native", "0"),
                                        ("y_common4", "0"), ("physical_ion_id", "other")])
def test_discrete_and_identity_fields_stay_exact_under_the_v3_policy(key, value):
    with pytest.raises(CampaignContractError, match=f"u1/{key}"):
        runtime.compare_replayed_predictions(rows(), rows(_change("u1", key, value)), probability_atol=1e-5,
                                             exact_decimal=True)


def test_uid_set_and_columns_must_match():
    replayed = rows()
    replayed.pop("u2")
    with pytest.raises(CampaignContractError, match="UID set"):
        runtime.compare_replayed_predictions(rows(), replayed, probability_atol=1e-5)
    extra = rows()
    extra["u1"]["p_native_Zn"] = "0.0"
    saved = rows()
    with pytest.raises(CampaignContractError, match="columns differ"):
        runtime.compare_replayed_predictions(saved, extra, probability_atol=1e-5, require_same_columns=True)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "", "x"])
@pytest.mark.parametrize("exact", [False, True])
def test_nonfinite_or_malformed_probability_never_matches(value, exact):
    with pytest.raises((CampaignContractError, ValueError)):
        runtime.compare_replayed_predictions(rows(), rows(_change("u1", "p_native_Mn", value)), probability_atol=1e-5,
                                             exact_decimal=exact)


@pytest.mark.parametrize("policy", [
    {"probability_atol": 1e-5}, {"policy_id": "", "probability_atol": 1e-5}, {"policy_id": "x"},
    {"policy_id": "x", "probability_atol": 2e-5}, {"policy_id": "x", "probability_atol": 0.0},
    {"policy_id": "x", "probability_atol": -1e-6}, {"policy_id": "x", "probability_atol": float("nan")},
    {"policy_id": "x", "probability_atol": True}, {"policy_id": "x", "probability_atol": "1e-5"},
    {"policy_id": "x", "probability_atol": 1e-5, "nested": {"a": 1}}, ["policy_id", 1e-5]])
def test_replay_policy_validation_refuses_loose_or_malformed_policies(policy):
    with pytest.raises(CampaignContractError):
        runtime.validate_replay_policy(policy)


def test_v3_policy_is_valid_and_bounded_by_pmm_core_replay_v1():
    import pmm_core_replay
    import pmm_v3_campaign as v3

    assert runtime.validate_replay_policy(v3.REPLAY_POLICY) == v3.REPLAY_POLICY
    assert v3.REPLAY_POLICY["probability_atol"] == float(pmm_core_replay.POLICY["probability_atol"])
    assert v3.REPLAY_POLICY["probability_atol"] == runtime.MAX_REPLAY_PROBABILITY_ATOL
    assert float(pmm_core_replay.POLICY["legacy_probability_atol"]) == runtime.STRICT_REPLAY_PROBABILITY_ATOL


# ---------------------------------------------------------------------------
# End to end on a tiny CPU fit (synthetic training-side fixture)
# ---------------------------------------------------------------------------

def _perturb_saved(run_dir: Path, key: str, value_fn) -> None:
    from training.source_cohort import sha256_file

    receipt_path = run_dir / "selected_checkpoint.json"
    receipt = json.loads(receipt_path.read_text())
    path = run_dir / receipt["validation_predictions"]["path"]
    with path.open(encoding="utf-8", newline="") as handle:
        table = list(csv.DictReader(handle))
    table[0][key] = value_fn(table[0][key])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    receipt["validation_predictions"]["sha256"] = sha256_file(path)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def test_policy_end_to_end_records_tolerance_and_keeps_strict_default(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_v3_training_options import tiny_argv
    from training import run as run_module
    from training.config import parse_args

    run_dir = run_module.run_training(parse_args(tiny_argv(tmp_path, monkeypatch, "replay_policy")))
    strict = runtime.replay_campaign_run(run_dir, output_dir=tmp_path / "strict")
    assert "replay_policy" not in strict and "max_probability_abs_difference" not in strict
    header = next(csv.reader((run_dir / "val_predictions.csv").open()))
    p_key = next(key for key in header if key.startswith("p_native_"))
    _perturb_saved(run_dir, p_key, lambda value: f"{float(value) + 5e-6:.8f}" if float(value) < 0.5
                   else f"{float(value) - 5e-6:.8f}")
    with pytest.raises(CampaignContractError, match=p_key):
        runtime.replay_campaign_run(run_dir, output_dir=tmp_path / "strict_after")
    relaxed = runtime.replay_campaign_run(run_dir, output_dir=tmp_path / "v3", replay_policy=V3_POLICY)
    assert relaxed["replay_policy"] == V3_POLICY and relaxed["prediction_rows_verified"] is True
    assert relaxed["reconciliation_status"] == "match"
    assert 4e-6 < relaxed["max_probability_abs_difference"] <= 1e-5
    on_disk = json.loads((tmp_path / "v3" / "replay_receipt.json").read_text())
    assert on_disk["replay_policy"] == V3_POLICY
    _perturb_saved(run_dir, "pred_common4", lambda value: str((int(value) + 1) % 4))
    with pytest.raises(CampaignContractError, match="pred_common4"):
        runtime.replay_campaign_run(run_dir, output_dir=tmp_path / "v3_class", replay_policy=V3_POLICY)
