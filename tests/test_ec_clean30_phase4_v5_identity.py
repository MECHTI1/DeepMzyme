"""CPU-only amendment-2 identity and independently checked replay boundaries."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ec_clean30 import campaign, phase3, phase4


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return path


def test_s1_trainer_command_scrubs_inherited_git_environment(tmp_path, monkeypatch):
    unit = phase4.s1_units()[0]
    original = phase4.commands.UnitCommand(unit_id=unit.unit_id, kind=unit.kind, step=unit.step,
        seed=unit.seed, recipe=campaign.recipe_for_unit(unit), argv=(), env={"LOCAL_SETTING": "kept"},
        identity={}, unset_env=("EXISTING_EXCLUSION",))
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "foreign.git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "foreign"))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.worktree")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", str(tmp_path / "foreign"))
    monkeypatch.setenv("EXISTING_EXCLUSION", "removed")
    monkeypatch.setenv("RETAIN_SETTING", "kept")
    monkeypatch.setattr(phase4, "_unit", lambda *args: unit)
    monkeypatch.setattr(phase4.commands, "build_unit_command", lambda *args, **kwargs: original)
    source = {"freeze1_sha256": "a" * 64, "freeze1b_sha256": "b" * 64,
              "amendment_chain": ["c" * 64], "git_head": "d" * 40, "source_tree_sha256": "e" * 64}
    command = phase4.build_s1_command({"unit_id": unit.unit_id, "execution_root": str(tmp_path), "device": "cuda"},
                                      source, object())
    environment = command.environment()
    assert not any(key.startswith("GIT_") for key in environment)
    assert "EXISTING_EXCLUSION" not in environment
    assert environment["LOCAL_SETTING"] == environment["RETAIN_SETTING"] == "kept"
    assert original.unset_env == ("EXISTING_EXCLUSION",)


def test_replay_cpu_logits_matches_the_science_contract():
    path = Path(__file__).parent / "fixtures/ec_clean30_amendment2_contract.json"
    if not path.exists():
        pytest.skip("The science contract joins this checkout in the amendment-2 integration.")
    policy = json.loads(path.read_text())["execution_api"]["cpu_replay"]
    result = phase4.replay_cpu_logits([[0, 1, 0]], [[0, 1, 0]])
    for key in ("absolute_tolerance", "near_tie_tolerance", "argmax_agreement"):
        assert result[key] == policy[key]
    assert policy["qualname"] == phase4.replay_cpu_logits.__name__


def test_replay_cpu_logits_counts_near_ties_but_requires_argmax_agreement():
    accepted = phase4.replay_cpu_logits([[1, 1, 0]], [[1.000001, 1, 0]])
    assert accepted["status"] == "passed" and accepted["near_tie_rows"] == 1
    with pytest.raises(ValueError, match="argmax changes 1"):
        phase4.replay_cpu_logits([[1, .999999, 0]], [[.999999, 1, 0]])


@pytest.mark.parametrize("saved,replayed,device", [
    ([[1, 0]], [[1.00002, 0]], "cpu"),
    ([[1, 0]], [[1, 0]], "cuda"),
    ([[1, 0]], [[float("nan"), 0]], "cpu"),
    ([[1, 0]], [[float("inf"), 0]], "cpu"),
    ([[1, 0]], [[1, 0, 0]], "cpu"),
    ([[1, 0]], [], "cpu"),
    ([], [], "cpu"),
    ([[1]], [[1]], "cpu"),
])
def test_replay_cpu_logits_refuses_drift_device_and_malformed_matrices(saved, replayed, device):
    with pytest.raises(ValueError):
        phase4.replay_cpu_logits(saved, replayed, device=device)


@pytest.fixture
def replay_run(tmp_path):
    from training.ec_prediction_export import LOGIT_COLUMNS

    run = tmp_path / "run"
    replay = run / "ec_prediction_replay"
    replay.mkdir(parents=True)
    original = {"checkpoint": {"sha256": "c" * 64}}
    record = {"device": "cpu", "logit_atol": 1e-5, "replay_status": "match", "n_problems": 0,
              "checkpoint_sha256": "c" * 64, "replay_files": {}}
    for key in ("pocket_predictions", "protein_predictions"):
        row = {"identity": "training-only", "pred": "0", **dict.fromkeys(LOGIT_COLUMNS, "0")}
        row[LOGIT_COLUMNS[0]] = "1"
        for directory, bindings in ((run, original), (replay, record["replay_files"])):
            path = directory / f"{key}.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            bindings[key] = {"path": path.name, "sha256": phase4._sha(path)}
    original_path = _json(run / "ec_prediction_receipt.json", original)
    record["source_receipt_sha256"] = phase4._sha(original_path)
    receipt = _json(replay / "ec_prediction_replay_receipt.json", record)
    return run, receipt, record


def test_verify_cpu_replay_checks_both_hash_bound_tables(replay_run):
    run, _, _ = replay_run
    result = phase4.verify_cpu_replay(run)
    assert result["status"] == "passed"
    assert set(result["comparisons"]) == {"pocket_predictions", "protein_predictions"}


@pytest.mark.parametrize("mutation", ["receipt_device", "receipt_source", "bytes", "argmax"])
def test_verify_cpu_replay_refuses_tampered_or_near_tie_flip_evidence(replay_run, mutation):
    run, receipt_path, receipt = replay_run
    if mutation == "receipt_device":
        receipt["device"] = "cuda"
    elif mutation == "receipt_source":
        receipt["source_receipt_sha256"] = "0" * 64
    else:
        path = run / "ec_prediction_replay/protein_predictions.csv"
        if mutation == "bytes":
            path.write_text(path.read_text() + "\n")
        else:
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["pred"] = "1"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            receipt["replay_files"]["protein_predictions"]["sha256"] = phase4._sha(path)
    _json(receipt_path, receipt)
    with pytest.raises(phase4.Phase4Error):
        phase4.verify_cpu_replay(run)


def test_phase4_source_passes_session_chain_receipt_and_expected_chain(tmp_path, monkeypatch):
    receipt = tmp_path / "chain-receipt.json"
    expected = ["a" * 64, "b" * 64]
    source = {"source_sha256": {"src/ec_clean30/phase4.py": "1" * 64}}
    observed = {}
    monkeypatch.setattr(phase3, "verify_execution_source", lambda path, **kwargs: observed.update(kwargs) or source)
    monkeypatch.setattr(phase3, "_bind_view", lambda view, value: None)
    monkeypatch.setattr(phase4.views.View, "load", lambda path: SimpleNamespace(fold=0, smoke=False))
    phase4._source_view({"execution_manifest": str(tmp_path / "manifest.json"), "view": str(tmp_path),
                         "amendment_chain_receipt": str(receipt), "expected_chain": expected})
    assert observed == {"require_published": True, "chain_receipt_path": receipt, "expected_chain": expected}


@pytest.fixture
def v5_authority(tmp_path, monkeypatch):
    identity = {"scientific_identity_sha256": "a" * 64, "fit_chain": ["b" * 64, "c" * 64]}
    source = {**identity, "freeze1_sha256": "d" * 64, "execution_manifest_sha256": "e" * 64}
    evidence = []
    for index in range(10):
        path = _json(tmp_path / f"measurement{index}.json", identity)
        evidence.append({"path": str(path), "sha256": phase4._sha(path)})
    recipes = [(f"recipe{index}", None) for index in range(7)]
    monkeypatch.setattr(campaign, "timing_recipes_v2", lambda: recipes, raising=False)
    monkeypatch.setattr(campaign, "build_fold_plan_v2", lambda fold: {"plan_sha256": "2" * 64}, raising=False)
    pretrain = _json(tmp_path / "pretrain.json", {"status": "passed"})
    monkeypatch.setattr(phase3, "verify_pretrain_evidence", lambda *args, **kwargs: {}, raising=False)
    measured = _json(tmp_path / "freeze1c.json", {
        **identity, "execution_manifest_sha256": "f" * 64,
        "format": "deepmzyme_ec_clean30_freeze1c_v2", "campaign_id": campaign.CAMPAIGN_ID,
        "freeze1_sha256": source["freeze1_sha256"], "lanes": 1, "evidence": evidence,
        "timing_recipes": [name for name, _ in recipes],
        "pretrain_evidence": {"path": str(pretrain), "sha256": phase4._sha(pretrain)},
    })
    authority = {**identity, "execution_manifest_sha256": "f" * 64, "campaign_id": campaign.CAMPAIGN_ID,
                 "status": "approved", "scope": phase4.V5_SCOPE, "freeze1c_sha256": phase4._sha(measured),
                 "allowed_folds": [0], "plan_sha256_by_fold": {"0": "2" * 64},
                 "allowed_steps": {"0": ["S1", "S4"]}, "clean_unit_ids": [],
                 "ceilings": {"gpu_seconds": 7200, "cpu_seconds": 7200, "cost_usd": 3},
                 "allocations": {"session": {"resource": "gpu", "allocation_started_unix": 1000,
                                             "deadline_unix": 8000, "max_total_seconds": 7200,
                                             "cost_ceiling_usd": 3}}}
    path = _json(tmp_path / "g3.json", authority)
    calls = []
    monkeypatch.setattr(phase3, "require_record_precedence", lambda record, source: calls.append(record), raising=False)
    return {"freeze1c": str(measured), "g3_authority": str(path), "g3_sha256": phase4._sha(path)}, source, authority, calls


def test_g3_and_freeze1c_reuse_same_science_under_new_execution_manifest(v5_authority):
    request, source, authority, precedence = v5_authority
    assert phase4.verify_authority(request, source)[0] == authority
    assert len(precedence) == 12  # G3, Freeze 1c and every G2 measurement.


@pytest.mark.parametrize("field,value", [("scientific_identity_sha256", "0" * 64), ("fit_chain", ["b" * 64])])
def test_g3_rejects_scientific_identity_or_prefix_chain_drift(v5_authority, field, value):
    request, source, authority, _ = v5_authority
    authority[field] = value
    path = _json(Path(request["g3_authority"]), authority)
    request["g3_sha256"] = phase4._sha(path)
    with pytest.raises(phase4.Phase4Error):
        phase4.verify_authority(request, source)
