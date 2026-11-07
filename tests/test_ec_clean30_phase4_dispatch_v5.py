"""Inert dispatch, scope, persistence ordering and scientific-stop checks."""
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

from ec_clean30 import campaign, phase3, phase4


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")
    return path


def _v2_plan(fold):
    if not hasattr(campaign, "build_fold_plan_v2"):
        pytest.skip("The science-side plan-v2 implementation joins at integration.")
    return campaign.build_fold_plan_v2(fold)


@pytest.mark.parametrize("fold", range(5))
def test_v2_fit_queues_are_complete_with_five_refit_templates(fold):
    plan = _v2_plan(fold)
    s1, s4 = phase4.s1_units(plan), phase4.fold_units(plan, "S4")
    assert len(s1) == (112 if fold == 0 else 96)
    assert len(s4) == 5 and len(s1) + len(s4) == (117 if fold == 0 else 101)
    assert {unit.fold for unit in s1 + s4} == {fold}
    assert {unit.step for unit in s1} == {"S1"} and {unit.step for unit in s4} == {"S4"}
    assert ("F1" in {unit.family for unit in s1}) is (fold == 0)
    plan["units"].pop()
    with pytest.raises(phase4.Phase4Error, match="complete canonical"):
        phase4.fold_units(plan, "S4")


def test_final_clean_unit_counts_and_seeds_come_from_frozen_record():
    frozen = json.loads((phase4.REPO_ROOT / phase3.freeze.CAMPAIGN_DOCS_DIR / "freeze1.json").read_text())
    seeds = frozen["constants"]["scoring"]["FROZEN_SETTINGS"]
    assert sum(len(phase4.clean_final_units(fold)) for fold in range(5)) + len(phase4.clean_units()) == 63
    for fold in range(5):
        units = phase4.clean_final_units(fold)
        for arm in ("r1", "r2", "c_esmc"):
            assert [unit["seed"] for unit in units if unit["arm"] == arm] == seeds[f"{arm}_seeds"]
        assert all(unit["fold"] == fold and unit["inner_fold"] is None for unit in units)


@pytest.fixture
def futility_guard(tmp_path, monkeypatch):
    identity = {"scientific_identity_sha256": "a" * 64, "fit_chain": ["b" * 64, "c" * 64]}
    source = {**identity, "amendment_chain": ["b" * 64, "c" * 64, "d" * 64],
              "amendment_path": "amendment3/amendment.json", "amendment_chain_receipt": str(tmp_path / "receipt.json"),
              "freeze1_sha256": "e" * 64}
    state = {"pointers": {}, "decision": "continue", "calls": []}
    def verify(path, **kwargs):
        state["calls"].append((path, kwargs))
        return {**identity, "decision": state["decision"]}
    fake = SimpleNamespace(read_pointers=lambda root, tip: state["pointers"], verify_futility_decision=verify)
    monkeypatch.setitem(sys.modules, "ec_clean30.futility", fake)
    import ec_clean30
    monkeypatch.setattr(ec_clean30, "futility", fake, raising=False)
    monkeypatch.setattr(phase3, "authenticated_chain", lambda source: {"tip_sequence": 3})
    request = {"campaign_root": str(tmp_path), "fold0_step_chain": str(tmp_path / "selection.jsonl")}
    return request, source, state


def test_fold0_initial_s1_needs_no_futility_record(futility_guard):
    request, source, state = futility_guard
    assert phase4.require_futility(request, source, 0) is None and not state["calls"]


@pytest.mark.parametrize("fold", range(5))
def test_required_continue_uses_full_chain_and_session_receipt(futility_guard, fold):
    request, source, state = futility_guard
    assert phase4.require_futility(request, source, fold, require_continue=True)["decision"] == "continue"
    _, kwargs = state["calls"][0]
    assert kwargs["expected_chain"] == source["amendment_chain"]
    assert kwargs["chain_receipt"] == Path(source["amendment_chain_receipt"])


def test_any_stop_pointer_blocks_resumed_fold0_inner_fits(futility_guard):
    request, source, state = futility_guard
    state.update(pointers={2: {"decision": "stop_futility"}, 3: {"decision": "continue"}}, decision="stop_futility")
    with pytest.raises(phase4.Phase4Error, match="stop_futility"):
        phase4.require_futility(request, source, 0)
    assert state["calls"]


def test_missing_pointer_after_preserved_futility_cannot_restart_inner_fits(futility_guard, tmp_path):
    request, source, state = futility_guard
    _write(tmp_path / "runs/fold0/phase4/runs/ecc30_science_fold0_futility/complete.json", {})
    phase4.require_futility(request, source, 0)
    assert state["calls"]


@pytest.mark.parametrize("kind,fold", [("s1", 0), ("refit", 0), ("s1", 2), ("clean", 0)])
def test_runtime_rechecks_stop_before_any_fit_or_resume_enters_execution(tmp_path, monkeypatch, kind, fold):
    source = {"scientific_identity_sha256": "a" * 64}
    view = SimpleNamespace(fold=fold, campaign_root=tmp_path)
    plan = {"fold": fold, "plan_sha256": "p"}
    path = _write(tmp_path / "plan.json", plan)
    request = {"campaign_root": str(tmp_path), "plan": str(path), "unit_id": "unit", "clean_scope": "final",
               "provider": "runpod"}
    authority = {"allowed_folds": [fold], "allowed_steps": {str(fold): ["S1", "S4"]}, "clean_unit_ids": ["unit"]}
    monkeypatch.setattr(phase4, "_source_view", lambda request: (source, view))
    monkeypatch.setattr(phase4, "verify_authority", lambda *args, **kwargs: (authority, {}))
    monkeypatch.setattr(campaign, "build_fold_plan_v2", lambda fold: plan, raising=False)
    calls = []
    def stop(*args, **kwargs):
        calls.append(kwargs["require_continue"])
        raise phase4.Phase4Error("stop_futility")
    monkeypatch.setattr(phase4, "require_futility", stop)
    monkeypatch.setattr(phase4, "_execution", lambda *args: pytest.fail("No execution/admission after stop"))
    with pytest.raises(phase4.Phase4Error, match="stop_futility"):
        phase4.run_unit(request, kind=kind)
    assert calls == [kind in ("refit", "clean") or fold != 0]


def test_scientific_request_accepts_paths_and_refuses_policy_override(tmp_path):
    request = {"execution_root": str(tmp_path), "step_chain": str(tmp_path / "records/selection.jsonl"),
               "scientific_inputs": {"components": str(tmp_path / "components.tsv"),
                                     "futility_out": str(tmp_path / "records/futility.json")}}
    assert phase4._science_paths(request, None) == request["scientific_inputs"]
    request["scientific_inputs"]["n_replicates"] = 1
    with pytest.raises(phase4.Phase4Error, match="paths only"):
        phase4._science_paths(request, None)


def test_sanity_runs_sample_export_check_in_order_before_worker_can_return(tmp_path, monkeypatch):
    attempt, work = tmp_path / "attempt", tmp_path / "attempt/work"
    work.mkdir(parents=True)
    events = []
    def run(argv, **kwargs):
        action = argv[argv.index("sample")] if "sample" in argv else "in-sample"
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert not any(key.startswith("GIT_") for key in kwargs["env"])
        events.append(action)
        Path(argv[argv.index("--out") + 1]).write_text("{}")
    def export(**kwargs):
        assert kwargs["allow_in_sample"] is True and kwargs["device"] == "cpu" and kwargs["mode"] == "validation"
        events.append("export")
    monkeypatch.setenv("GIT_DIR", "/foreign")
    monkeypatch.setattr(phase4.subprocess, "run", run)
    monkeypatch.setattr(phase4.clean, "run_export", export)
    fake = SimpleNamespace(verify_record=lambda path: events.append("verify") or {})
    monkeypatch.setitem(sys.modules, "ec_clean30.clean_sanity", fake)
    import ec_clean30
    monkeypatch.setattr(ec_clean30, "clean_sanity", fake, raising=False)
    request = {"python": sys.executable, "cap_prefix": ["cap"], "clean_clone": str(tmp_path / "clone"),
               "export_script": str(tmp_path / "export.py"), "threads": 2}
    phase4._clean_sanity_commands(request, work, attempt, [], arm="r1_inner", clean_python=tmp_path / "python")
    assert events == ["sample", "export", "in-sample", "verify"]


def test_scientific_step_returns_persisted_binding_stop_to_coordinator(tmp_path, monkeypatch):
    record = {"status": "passed", "unit_id": "ecc30_science_fold0_futility", "decision": "stop_futility"}
    _write(tmp_path / "runs" / record["unit_id"] / "complete.json", record)
    calls = []
    monkeypatch.setattr(phase4, "_run_remote_queue", lambda request, units, **kwargs: calls.append((units, kwargs))
                        or {"status": "passed"})
    request = {"fold": 0, "durable_root": str(tmp_path), "scientific_forecasts": {"futility": 90},
               "scientific_timeouts": {"futility": 60}}
    assert phase4.run_scientific_step(request, step="futility", transport="owned", check_session="check") == record
    assert calls[0][1]["action"] == "scientific-unit" and calls[0][1]["close_session"] is False


def test_refit_queue_uses_exact_five_templates(tmp_path, monkeypatch):
    plan = _v2_plan(1)
    path = _write(tmp_path / "plan.json", plan)
    calls = []
    monkeypatch.setattr(phase4, "_run_remote_queue", lambda request, units, **kwargs: calls.append((units, kwargs)) or {})
    phase4.run_refit_queue({"local_plan": str(path), "refit_forecast_seconds": 100, "refit_timeout_seconds": 80})
    assert [unit["unit_id"] for unit in calls[0][0]] == [unit.unit_id for unit in campaign.fold_units_v2(plan, "S4")]
    assert calls[0][1]["action"] == "refit-unit"


@pytest.fixture
def pending_scientific_outputs(tmp_path, monkeypatch):
    fake = SimpleNamespace(pointer_path=lambda root, sequence: root / "amendment2/futility" / f"fut_seq{sequence}.json")
    monkeypatch.setitem(sys.modules, "ec_clean30.futility", fake)
    import ec_clean30
    monkeypatch.setattr(ec_clean30, "futility", fake, raising=False)
    campaign_root = tmp_path / "campaign"
    root = campaign_root / "runs/fold0/phase4"
    request = {"execution_root": str(root), "campaign_root": str(campaign_root),
        "unit_id": "ecc30_science_fold0_futility", "scientific_step": "futility", "attempt": 1,
        "expected_chain": ["b" * 64, "c" * 64, "d" * 64], "step_chain": str(root / "records/selection.jsonl"),
        "scientific_inputs": {"futility_out": str(root / "records/futility.json")}}
    source = {"scientific_identity_sha256": "a" * 64, "fit_chain": ["b" * 64, "c" * 64]}
    command = _write(root / "commands/request.json", request)
    phase4._prepare_scientific_outputs(request, source, command)
    return request, phase4._attempt(request)


def test_pending_scientific_output_marker_blocks_automatic_recovery(pending_scientific_outputs):
    request, _ = pending_scientific_outputs
    with pytest.raises(phase4.Phase4Error, match="Pending scientific outputs"):
        phase4.require_restored_scientific_outputs(Path(request["execution_root"]))


def test_parent_restores_all_known_pointers_even_partial_bytes(pending_scientific_outputs):
    request, attempt = pending_scientific_outputs
    paths = phase4._science_preservation_paths(request)
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'{"interrupted":')
    restored = phase4._restore_scientific_outputs(attempt)
    assert restored["status"] == "restored" and len(restored["files"]) == 4
    assert all((attempt / item["snapshot"]).read_bytes() == b'{"interrupted":' for item in restored["files"])
    phase4.require_restored_scientific_outputs(Path(request["execution_root"]))


def test_scientific_preservation_verifies_offline_after_original_paths_disappear(pending_scientific_outputs, tmp_path):
    request, attempt = pending_scientific_outputs
    path = Path(request["step_chain"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("partial step chain\n")
    phase4._restore_scientific_outputs(attempt)
    copy = tmp_path / "independent-copy"
    shutil.copytree(attempt, copy)
    shutil.rmtree(request["campaign_root"])
    record = phase4.verify_preserved_scientific_outputs(copy / phase4.SCIENTIFIC_PENDING, original_attempt=attempt)
    assert sum(item["exists"] for item in record["files"]) == 1
    item = next(item for item in record["files"] if item["exists"])
    (copy / item["snapshot"]).write_text("changed")
    with pytest.raises(phase4.Phase4Error, match="output bytes differ"):
        phase4.verify_preserved_scientific_outputs(copy / phase4.SCIENTIFIC_PENDING, original_attempt=attempt)
