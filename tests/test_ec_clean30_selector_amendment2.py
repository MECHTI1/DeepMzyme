"""Amendment-2 selector: fit-chain dispatch, plan v2, S1 (best inner, C-GF from F6 control files), S2 (L9 only), the
FUT record and S4, and the refit recipe (synthetic run folders)."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ec_clean30 import futility, selection, selector  # noqa: E402
from ec_clean30.campaign import CAMPAIGN_ID, build_fold_plan_v2, plan_units  # noqa: E402
from ec_clean30.selection import read_step_chain, verify_step_chain  # noqa: E402
from ec_clean30.selector import SelectionInputError  # noqa: E402
from ec_clean30_amendment2_fixtures import (  # noqa: E402
    A1, A2, CHAIN, FREEZE1, IDS, TABLES, amendment2, append_fut_record, install_family_table, sanity_stub, select,
    world, write_control_files, write_plan_v1, write_plan_v2,
)

A3 = "a3" * 32


@pytest.fixture(autouse=True)
def family_table(monkeypatch):
    install_family_table(monkeypatch)


# ---------------------------------------------------------------------------
# Policy dispatch on the fit chain (contract X13)


def test_policy_dispatch_by_fit_chain(monkeypatch, tmp_path):
    from ec_clean30 import freeze

    path, calls = amendment2(monkeypatch, tmp_path)
    policy = selector._selection_policy(path, FREEZE1, smoke=False)
    assert policy == {"freeze1_amendment_chain": CHAIN, "s1_rule": "best_inner_harm_guarded_v1",
                      "selection_pipeline": "amendment2_selection_v1",
                      "selection_policy_sha256": selection.selection_policy_sha256_v2(),
                      "amendment_tip_chain": CHAIN, "scientific_identity_sha256": "5c" * 32,
                      "requires_futility_continue": False}
    assert calls == [False] and selector.is_pipeline_v2(policy)
    one = tmp_path / "s1_amendment.json"
    one.write_text("{}\n")
    monkeypatch.setattr(freeze, "verify_s1_amendment", lambda given, *, expected_freeze1_sha256, require_published: {
        "document_sha256": A1, "amendment_chain": [A1], "s1_rule": selection.AMENDED_S1_RULE, "published": True})
    assert selector._selection_policy(one, FREEZE1, smoke=False) == {
        "freeze1_amendment_chain": [A1], "s1_rule": "fixed_anchor_holm_v1"}
    assert selector._selection_policy(None, FREEZE1, smoke=False)["s1_rule"] == "family_escalation_v1"


def test_a_tip_with_later_analysis_links_binds_the_fit_chain(monkeypatch, tmp_path):
    path, _ = amendment2(monkeypatch, tmp_path, chain=[A1, A2, A3], fit_chain=CHAIN)
    policy = selector._selection_policy(path, FREEZE1, smoke=False)
    assert policy["freeze1_amendment_chain"] == CHAIN and policy["amendment_tip_chain"] == [A1, A2, A3]
    assert policy["requires_futility_continue"] is True
    # Fits bound to the fit chain [A1, A2] verify under the [A1, A2, A3] tip.
    selector._check_policy_record({"freeze1_amendment_chain": CHAIN}, policy, "fit", fit=True)
    with pytest.raises(SelectionInputError, match="amendment chain"):
        selector._check_policy_record({"freeze1_amendment_chain": [A1, A2, A3]}, policy, "fit", fit=True)


@pytest.mark.parametrize("options, message", [
    ({"chain": [A1]}, "fit chain \\[amendment 1, amendment 2\\]"),
    ({"chain": [A2, A1, A3], "fit_chain": [A2, A1, A3]}, "fit chain \\[amendment 1, amendment 2\\]"),
    ({"chain": [A1, A1]}, "malformed chain or fit chain"),
    ({"tip_sequence": 3}, "fit chain \\[amendment 1, amendment 2\\]"),
    ({"chain": [A1, A2, A3], "fit_chain": [A1, A3]}, "fit chain \\[amendment 1, amendment 2\\]"),
    ({"policies": {}}, "does not carry the implemented amendment-2 selection policy"),
    ({"policies": {"AM2-S1": {"policy": {**selection.selection_policy_record_v2(), "s3": {"run": True}}}}},
     "does not carry"),
    ({"published": False}, "published amendment"),
    ({"identity": "not a sha"}, "malformed scientific identity"),
    ({"identity": None}, "missing or malformed scientific identity"),
])
def test_other_chains_are_refused(monkeypatch, tmp_path, options, message):
    path, _ = amendment2(monkeypatch, tmp_path, **options)
    with pytest.raises(SelectionInputError, match=message):
        selector._selection_policy(path, FREEZE1, smoke=False)


def test_unpublished_chain_serves_smoke_only_and_a_missing_verifier_refuses(monkeypatch, tmp_path):
    path, calls = amendment2(monkeypatch, tmp_path, published=False)
    assert selector.is_pipeline_v2(selector._selection_policy(path, FREEZE1, smoke=True)) and calls == [True]
    monkeypatch.undo()
    install_family_table(monkeypatch)
    import builtins

    real_import = builtins.__import__

    def no_chain_module(name, *args, **kwargs):
        if name == "ec_clean30" and args and args[2] and "amendment_chain" in args[2]:
            raise ImportError("absent")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_chain_module)
    with pytest.raises(SelectionInputError, match="general amendment-chain verifier"):
        selector._selection_policy(path, FREEZE1, smoke=False)


def test_the_verifier_is_called_with_the_contract_signature(monkeypatch, tmp_path):
    import types

    import ec_clean30
    from ec_clean30.amendment_rules import FREEZE1B_SHA256

    seen = {}

    def verify_chain(tip, **kwargs):
        seen.update(tip=tip, **kwargs)
        return {}

    module = types.SimpleNamespace(verify_chain=verify_chain)
    monkeypatch.setitem(sys.modules, "ec_clean30.amendment_chain", module)
    monkeypatch.setattr(ec_clean30, "amendment_chain", module, raising=False)
    selector._verify_amendment_chain(tmp_path / "amendment.json", FREEZE1, smoke=True)
    assert seen["expected_freeze1_sha256"] == FREEZE1 and seen["expected_freeze1b_sha256"] == FREEZE1B_SHA256
    assert seen["require_published"] is False and Path(seen["repo"]).is_dir()


# ---------------------------------------------------------------------------
# Plan v2 and record formats


def test_plan_versions_are_refused_under_the_other_policy(monkeypatch, tmp_path):
    path, _ = amendment2(monkeypatch, tmp_path)
    root = tmp_path / CAMPAIGN_ID
    v1, v2 = write_plan_v1(root), write_plan_v2(root)
    assert selector.load_frozen_plan(v2, 0)["format"] == "deepmzyme_ec_clean30_fold_plan_v2"
    assert selector.load_frozen_plan(v1, 0)["format"] == "deepmzyme_ec_clean30_fold_plan_v1"
    with pytest.raises(SelectionInputError, match="needs a deepmzyme_ec_clean30_fold_plan_v2 plan"):
        selector.run_select(fold=0, step="S1", plan_path=v1, runs_dirs=[tmp_path], tables=TABLES,
                            freeze1_sha256=FREEZE1, out=tmp_path / "out.jsonl", freeze1_amendment_path=path)
    with pytest.raises(SelectionInputError, match="needs a deepmzyme_ec_clean30_fold_plan_v1 plan"):
        selector.run_select(fold=0, step="S1", plan_path=v2, runs_dirs=[tmp_path], tables=TABLES,
                            freeze1_sha256=FREEZE1, out=tmp_path / "out.jsonl")
    wrong = root / "plans_v2" / "fold1.json"
    wrong.write_text(v1.read_text().replace('"fold": 0', '"fold": 1'))
    with pytest.raises(SelectionInputError, match="holds fold|must have format"):
        selector.load_frozen_plan(wrong, 1)
    (root / "plans" / "fold0.json").write_text(v2.read_text())
    with pytest.raises(SelectionInputError, match="must have format deepmzyme_ec_clean30_fold_plan_v1"):
        selector.load_frozen_plan(root / "plans" / "fold0.json", 0)


def test_steps_outside_the_active_pipeline_are_refused(monkeypatch, tmp_path):
    path, _ = amendment2(monkeypatch, tmp_path)
    with pytest.raises(SelectionInputError, match="S3 is not a step of the amendment2_selection_v1"):
        selector.run_select(fold=0, step="S3", plan_path=tmp_path / "plans_v2" / "fold0.json", runs_dirs=[tmp_path],
                            tables=TABLES, freeze1_sha256=FREEZE1, out=tmp_path / "o.jsonl",
                            freeze1_amendment_path=path)
    with pytest.raises(SelectionInputError, match="S4 is not a step of the family_escalation_v1"):
        selector.run_select(fold=0, step="S4", plan_path=tmp_path / "plans" / "fold0.json", runs_dirs=[tmp_path],
                            tables=TABLES, freeze1_sha256=FREEZE1, out=tmp_path / "o.jsonl")
    with pytest.raises(SelectionInputError, match="L3 cost inputs are refused"):
        selector.run_select(fold=0, step="S2", plan_path=tmp_path / "plans_v2" / "fold0.json", runs_dirs=[tmp_path],
                            tables=TABLES, freeze1_sha256=FREEZE1, out=tmp_path / "o.jsonl",
                            freeze1_amendment_path=path, freeze1c_path=tmp_path / "c.json",
                            l3_cost_disposition_path=tmp_path / "d.json")
    with pytest.raises(SelectionInputError, match="belong to the amendment-2 pipeline"):
        selector.run_select(fold=0, step="S1", plan_path=tmp_path / "plans" / "fold0.json", runs_dirs=[tmp_path],
                            tables=TABLES, freeze1_sha256=FREEZE1, out=tmp_path / "o.jsonl",
                            clean_sanity_record_path=tmp_path / "s.json")


# ---------------------------------------------------------------------------
# S1 -> S2 -> FUT -> S4


@pytest.fixture
def v2_fold(monkeypatch, tmp_path):
    path, _ = amendment2(monkeypatch, tmp_path)
    world_ = world(tmp_path, accuracies={"F5": 0.9, "C-G": 0.95})
    world_["sanity"] = sanity_stub(monkeypatch, tmp_path)
    out = tmp_path / "records" / "fold0_steps.jsonl"
    out.parent.mkdir()
    return world_, out, path


def test_s1_s2_fut_s4_chain_completes(v2_fold):
    world_, out, amendment = v2_fold
    s1 = select(world_, "S1", out, amendment)
    payload = s1["payload"]
    assert payload["format"] == "deepmzyme_ec_clean30_step_record_v2" and payload["schema_version"] == 2
    assert payload["selection_pipeline"] == "amendment2_selection_v1" and payload["freeze1_amendment_chain"] == CHAIN
    assert payload["plan_sha256"] == build_fold_plan_v2(0)["plan_sha256"]
    screen = payload["best_inner_screen"]
    assert screen["rule_id"] == "best_inner_harm_guarded_v1" and screen["bootstrap"] is None
    assert payload["s1_winner"]["family"] == screen["winner_family"] == "F5"
    assert set(screen["controls"]) == {"F1", "C-G", "C-GF"}  # C-G scores highest and is never selected
    assert screen["control_scores_points"]["C-G"] > screen["candidate_scores_points"]["F5"]
    assert "F6 and C-GF pretraining may use non-metalloenzyme" in screen["f6_holdout_asymmetry"]
    assert screen["clean_sanity_record_sha256"] == payload["clean_sanity"]["sha256"]
    control = payload["whole_protein_control"]
    assert control["source_family"] == "F6" and len(control["unit_ids"]) == 8
    assert all("lr1e-4" in unit_id for unit_id in control["unit_ids"])  # the first F6 setting
    assert control["second_setting_determinism"]["n_content_equal"] == 8
    assert len(payload["configurations"]["C-GF"]["fits"]) == 8
    assert len(payload["configurations"]) == 7 * 2 + 1
    assert all(choice["metric"] == "ec_classes_1_6_accuracy" for choice in payload["learning_rate_choices"].values())
    s2 = select(world_, "S2", out, amendment)["payload"]
    assert s2["screen_scope"] == "L9_only" and s2["dropped_levers"] == ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"]
    assert s2["l9"]["details"]["rule_id"] == "l9_tau_promote_primary_v1"
    assert s2["l9"]["decision"]["config"]["alpha"] == 0.01 and s2["adopted_tau"] == 0.0
    assert s2["selected"]["step"] == "S2" and s2["selected"]["config"] == payload["selected"]["config"]
    with pytest.raises(SelectionInputError, match="S4 needs the FUT record"):
        select(world_, "S4", out, amendment)
    append_fut_record(world_, out)
    s4 = select(world_, "S4", out, amendment)
    spec = s4["payload"]["refit_specification"]
    assert spec["rule_id"] == "refit_spec_v2" and spec["family"] == "F5" and spec["epochs"] == 30
    assert spec["rbf_distances"] == "normalized" and spec["fold"] == 0 and spec["futility_record_sha256"] == "fe" * 32
    records = read_step_chain(out, anchor_sha256=FREEZE1)
    assert [record["step"] for record in records] == ["S1", "S2", "FUT", "S4"]
    assert spec["s1_record_sha256"] == selection.record_sha256(records[0])
    assert spec["s2_record_sha256"] == selection.record_sha256(records[1])
    assert s4["payload"]["fut_step_record_sha256"] == selection.record_sha256(records[2])
    assert s4["payload"]["refit_units"] == [f"ecc30__fold0__S4__selected__refit__seed{seed}" for seed in range(42, 47)]
    assert s4["payload"]["futility"]["decision"] == "continue" and s4["payload"]["n_runner_records"] == 0
    assert verify_step_chain(records, anchor_sha256=FREEZE1)
    assert not any(pid in out.read_text() for pid in IDS)
    with pytest.raises(SelectionInputError, match="already holds the S4 record"):
        select(world_, "S4", out, amendment)
    refit = selector.verified_refit_recipe(out, 0, freeze1_sha256=FREEZE1, amendment_tip=amendment,
                                           campaign_root=world_["root"], plan_path=world_["plan_path"])
    assert refit.recipe.to_dict() == payload["s1_winner"] and refit.tau == 0.0 and refit.seeds == (42, 43, 44, 45, 46)
    assert refit.f6_pretraining is None and refit.s4_record_sha256 == selection.record_sha256(records[3])


def test_fold0_s1_needs_the_held_out_clean_sanity_record(monkeypatch, v2_fold, tmp_path):
    world_, out, amendment = v2_fold
    with pytest.raises(SelectionInputError, match="needs --clean-sanity-record"):
        select(world_, "S1", out, amendment, clean_sanity_record_path=None)
    failed = sanity_stub(monkeypatch, tmp_path, status="comparator_diagnosis_required", name="failed.json")
    with pytest.raises(SelectionInputError, match="comparator diagnosis is required"):
        select(world_, "S1", out, amendment, clean_sanity_record_path=failed)
    assert not out.exists()
    disposed = sanity_stub(monkeypatch, tmp_path, status="comparator_diagnosis_required",
                           disposition="proceed_disclosed", name="disposed.json")
    payload = select(world_, "S1", out, amendment, clean_sanity_record_path=disposed)["payload"]
    assert payload["clean_sanity"]["disposition"] == "proceed_disclosed"
    with pytest.raises(SelectionInputError, match="bound by the fold-0 S1 record only"):
        select(world_, "S2", out, amendment, clean_sanity_record_path=disposed)


def test_futility_precedence(v2_fold, tmp_path):
    world_, out, amendment = v2_fold
    select(world_, "S1", out, amendment)
    select(world_, "S2", out, amendment)
    head = out.read_text()
    append_fut_record(world_, out, decision="stop_futility")
    with pytest.raises(SelectionInputError, match="stop_futility"):
        select(world_, "S4", out, amendment)
    policy = selector._selection_policy(amendment, FREEZE1, smoke=False)
    # A rerun FUT = continue in the same chain is refused, and S1 or S2 after FUT is refused.
    rerun = tmp_path / "rerun.jsonl"
    rerun.write_text(out.read_text())
    append_fut_record(world_, rerun, decision="continue", pointer=False)
    with pytest.raises(SelectionInputError, match="second futility record"):
        selector._check_policy_chain(read_step_chain(rerun, anchor_sha256=FREEZE1), policy)
    records = read_step_chain(out, anchor_sha256=FREEZE1)
    after = tmp_path / "after.jsonl"
    after.write_text(out.read_text())
    selection.append_step_record(after, "S2", records[1]["payload"], anchor_sha256=FREEZE1)
    with pytest.raises(SelectionInputError, match="after the fold-0 futility record"):
        selector._check_policy_chain(read_step_chain(after, anchor_sha256=FREEZE1), policy)
    # Any stop pointer wins, even against a fresh chain whose FUT record says continue.
    other = tmp_path / "other.jsonl"
    other.write_text(head)
    append_fut_record(world_, other, decision="continue", pointer=False)
    decision = futility.verify_futility_decision(other, freeze1_sha256=FREEZE1, amendment_tip=amendment,
                                                 campaign_root=world_["root"])
    assert decision["decision"] == "stop_futility" and decision["pointer_sequence"] == 2


def test_a_non_binding_futility_record_never_continues(v2_fold):
    world_, out, amendment = v2_fold
    select(world_, "S1", out, amendment)
    select(world_, "S2", out, amendment)
    append_fut_record(world_, out, binding=False)
    with pytest.raises(SelectionInputError, match="not binding"):
        select(world_, "S4", out, amendment)


def test_later_folds_need_the_verified_fold0_continue(v2_fold, tmp_path):
    world_, out, amendment = v2_fold
    plan1 = write_plan_v2(world_["root"], fold=1)
    options = {"fold": 1, "step": "S1", "plan_path": plan1, "runs_dirs": [world_["runs"]], "tables": TABLES,
               "freeze1_sha256": FREEZE1, "out": tmp_path / "fold1.jsonl", "freeze1_amendment_path": amendment}
    with pytest.raises(SelectionInputError, match="fold-0 step chain and --campaign-root"):
        selector.run_select(**options)
    select(world_, "S1", out, amendment)
    select(world_, "S2", out, amendment)
    with pytest.raises(SelectionInputError, match="holds no futility record"):
        selector.run_select(**options, fold0_step_chain_path=out, campaign_root=world_["root"])
    append_fut_record(world_, out, decision="stop_futility")
    with pytest.raises(SelectionInputError, match="stop_futility"):
        selector.run_select(**options, fold0_step_chain_path=out, campaign_root=world_["root"])


def test_f6_winner_pretrains_inside_its_refits(monkeypatch, tmp_path):
    path, _ = amendment2(monkeypatch, tmp_path)
    world_ = world(tmp_path, accuracies={"F6": 0.95})
    world_["sanity"] = sanity_stub(monkeypatch, tmp_path)
    out = tmp_path / "steps.jsonl"
    assert select(world_, "S1", out, path)["payload"]["s1_winner"]["family"] == "F6"
    select(world_, "S2", out, path)
    append_fut_record(world_, out)
    s4 = select(world_, "S4", out, path)["payload"]
    pretraining = s4["refit_specification"]["whole_protein_pretraining"]
    assert "inside each refit" in pretraining["location"] and "whole corpus" in pretraining["corpus"]
    assert "pretraining_units" not in s4
    refit = selector.verified_refit_recipe(out, 0, freeze1_sha256=FREEZE1, amendment_tip=path,
                                           campaign_root=world_["root"], plan_path=world_["plan_path"])
    assert refit.recipe.family == "F6" and refit.recipe.pretrains_global_branch and refit.f6_pretraining


def test_a_raw_rbf_winner_keeps_its_encoding(monkeypatch, tmp_path):
    path, _ = amendment2(monkeypatch, tmp_path)
    world_ = world(tmp_path, accuracies={"F3": 0.95})
    world_["sanity"] = sanity_stub(monkeypatch, tmp_path)
    out = tmp_path / "steps.jsonl"
    winner = select(world_, "S1", out, path)["payload"]["s1_winner"]
    assert winner["family"] == "F3" and winner["rbf_distances"] == "raw"
    select(world_, "S2", out, path)
    append_fut_record(world_, out)
    assert select(world_, "S4", out, path)["payload"]["refit_specification"]["rbf_distances"] == "raw"
    refit = selector.verified_refit_recipe(out, 0, freeze1_sha256=FREEZE1, amendment_tip=path,
                                           campaign_root=world_["root"], plan_path=world_["plan_path"])
    assert refit.recipe.rbf_distances == "raw"


def test_s2_refuses_any_s2_fit(v2_fold):
    world_, out, amendment = v2_fold
    select(world_, "S1", out, amendment)
    unit = next(unit for unit in plan_units(build_fold_plan_v2(0)) if unit.step == "S1")
    stray_id = unit.unit_id.replace("__S1__", "__S2__")
    recipe = {"family": "F2", "learning_rate": {"learning_rate": 1e-4}, "epochs": 30, "lever_configs": ["L4"]}
    identity = {"campaign_id": CAMPAIGN_ID, "unit_id": stray_id, "step": "S2", "kind": "inner", "fold": 0,
                "inner_fold": 0, "seed": 42, "freeze1_amendment_chain": CHAIN, "recipe": recipe}
    (world_["runs"] / f"{stray_id}.runner.json").write_text(json.dumps({"unit_id": stray_id, "unit": {
        "unit_id": stray_id, "kind": "inner", "step": "S2", "seed": 42, "inner_fold": 0, "recipe": recipe,
        "identity": identity}}))
    with pytest.raises(SelectionInputError, match="S2 or S3 fits are refused"):
        select(world_, "S2", out, amendment)


def test_missing_candidate_and_missing_control_files_refuse_s1(monkeypatch, tmp_path):
    path, _ = amendment2(monkeypatch, tmp_path)
    sanity = sanity_stub(monkeypatch, tmp_path)
    no_f6 = world(tmp_path / "a", skip=lambda unit: unit.family == "F6")
    no_f6["sanity"] = sanity
    out = tmp_path / "a.jsonl"
    with pytest.raises(SelectionInputError, match="S1 fits: 16 required fits are missing"):
        select(no_f6, "S1", out, path)
    assert not out.exists()
    f6 = next(unit for unit in plan_units(build_fold_plan_v2(0)) if unit.family == "F6")
    no_control = world(tmp_path / "b")
    no_control["sanity"] = sanity
    (no_control["runs"] / f6.unit_id / selector.WP_CONTROL_PREDICTIONS_NAME).unlink()
    with pytest.raises(SelectionInputError, match="C-GF control file"):
        select(no_control, "S1", out, path)
    changed = world(tmp_path / "c")
    changed["sanity"] = sanity
    control = changed["runs"] / f6.unit_id / selector.WP_CONTROL_PREDICTIONS_NAME
    control.write_text(control.read_text() + "\n")
    with pytest.raises(SelectionInputError, match="differs from its receipt"):
        select(changed, "S1", out, path)
    foreign = world(tmp_path / "d", identity_changes={f6.unit_id: {"freeze1_amendment_chain": [A1]}})
    foreign["sanity"] = sanity
    with pytest.raises(SelectionInputError, match="amendment chain"):
        select(foreign, "S1", out, path)
    assert not out.exists()


def test_second_setting_control_differences_are_recorded_not_refused(monkeypatch, tmp_path):
    path, _ = amendment2(monkeypatch, tmp_path)
    world_ = world(tmp_path)
    world_["sanity"] = sanity_stub(monkeypatch, tmp_path)
    second = [unit for unit in plan_units(build_fold_plan_v2(0)) if unit.family == "F6" and "lr3e-4" in unit.unit_id]
    for name in (selector.WP_CONTROL_PREDICTIONS_NAME, selector.WP_PRETRAIN_RECEIPT_NAME):
        (world_["runs"] / second[0].unit_id / name).unlink()
    write_control_files(world_["runs"] / second[0].unit_id, second[0], accuracy=0.3, branch_digest="b2" * 32)
    payload = select(world_, "S1", tmp_path / "steps.jsonl", path)["payload"]
    determinism = payload["whole_protein_control"]["second_setting_determinism"]
    assert determinism["n_positions"] == 8 and determinism["n_content_equal"] == 7
    assert determinism["n_branch_digest_equal"] == 7


def _host(model_name):
    return {"runtime": {"cpu_identity": {"machine": "x86_64", "processor": "x86_64", "model_name": model_name},
                        "torch_version": "2.x"},
            "hyperparameters": {"num_threads": 1}}


def test_a_branch_digest_difference_on_the_same_recorded_host_is_refused(monkeypatch, tmp_path):
    path, _ = amendment2(monkeypatch, tmp_path)
    world_ = world(tmp_path)
    world_["sanity"] = sanity_stub(monkeypatch, tmp_path)
    f6 = [unit for unit in plan_units(build_fold_plan_v2(0)) if unit.family == "F6"]
    second = [unit for unit in f6 if "lr3e-4" in unit.unit_id]
    first = next(unit for unit in f6 if unit not in second and unit.inner_fold == second[0].inner_fold
                 and unit.seed == second[0].seed)

    def rewrite(unit, digest, changes):
        for name in (selector.WP_CONTROL_PREDICTIONS_NAME, selector.WP_PRETRAIN_RECEIPT_NAME):
            (world_["runs"] / unit.unit_id / name).unlink()
        write_control_files(world_["runs"] / unit.unit_id, unit, branch_digest=digest, changes=changes)

    rewrite(first, "b1" * 32, _host("cpu-a"))
    rewrite(second[0], "b2" * 32, _host("cpu-a"))
    out = tmp_path / "same_host.jsonl"
    with pytest.raises(SelectionInputError, match="same recorded CPU identity and thread count"):
        select(world_, "S1", out, path)
    assert not out.exists()
    # Another recorded CPU identity: the difference is recorded, not refused.
    rewrite(second[0], "b2" * 32, _host("cpu-b"))
    payload = select(world_, "S1", tmp_path / "other_host.jsonl", path)["payload"]
    determinism = payload["whole_protein_control"]["second_setting_determinism"]
    assert determinism["n_branch_digest_equal"] == 7 and determinism["n_same_recorded_host"] == 0
    # Equal digests on the same recorded host pass and are counted.
    rewrite(second[0], "b1" * 32, _host("cpu-a"))
    payload = select(world_, "S1", tmp_path / "equal.jsonl", path)["payload"]
    determinism = payload["whole_protein_control"]["second_setting_determinism"]
    assert determinism["n_branch_digest_equal"] == 8 and determinism["n_same_recorded_host"] == 1


def test_mixed_formats_and_historical_records_are_refused(monkeypatch, tmp_path, v2_fold):
    world_, out, amendment = v2_fold
    s1 = select(world_, "S1", out, amendment)
    historical = dict(s1["payload"], format=selector.STEP_RECORD_FORMAT)
    mixed = tmp_path / "mixed.jsonl"
    selector.append_step_record(mixed, "S1", historical, anchor_sha256=FREEZE1)
    with pytest.raises(SelectionInputError, match="mixed formats are refused"):
        select(world_, "S2", mixed, amendment)
    relabelled = dict(s1["payload"])
    relabelled["best_inner_screen"] = dict(relabelled["best_inner_screen"], rule_id=selection.AMENDED_S1_RULE)
    other = tmp_path / "relabelled.jsonl"
    selector.append_step_record(other, "S1", relabelled, anchor_sha256=FREEZE1)
    with pytest.raises(SelectionInputError, match="best-inner rule"):
        select(world_, "S2", other, amendment)
    # A historical policy refuses the amendment-2 chain before reading any fit.
    with pytest.raises(SelectionInputError, match="amendment chain|selection_pipeline"):
        selector._check_policy_chain(read_step_chain(out, anchor_sha256=FREEZE1),
                                     {"freeze1_amendment_chain": [], "s1_rule": selection.HISTORICAL_S1_RULE})


def test_gap_gate_is_refused_under_amendment2(v2_fold, tmp_path):
    world_, out, amendment = v2_fold
    with pytest.raises(SelectionInputError, match="gap gate is superseded"):
        selector.run_gap_gate(fold=0, after="S1", step_chain=out, freeze1_sha256=FREEZE1,
                              plan_path=world_["plan_path"], runs_dirs=[world_["runs"]], clean_r1_inner=[],
                              clean_r2_inner=[], tables=TABLES, out=tmp_path / "gap.json",
                              freeze1_amendment_path=amendment)
    assert not (tmp_path / "gap.json").exists()


def test_recipe_key_compares_amendment2_fields():
    base = {"family": "F3", "learning_rate": {"learning_rate": 1e-4}, "epochs": 30, "lever_configs": []}
    assert selector._recipe_key(base) == {"family": "F3", "learning_rate": [["learning_rate", 1e-4]], "epochs": 30,
                                          "lever_configs": []}
    assert selector._recipe_key({**base, "rbf_distances": "raw"}) != selector._recipe_key(base)
    assert selector._recipe_key({**base, "rbf_distances": "normalized"}) == selector._recipe_key(base)
    assert selector._recipe_key({**base, "other": 1}) != selector._recipe_key(base)
    record = {"family": "F3", "learning_rate": {"gvp_learning_rate": 3e-4, "learning_rate": 1e-4}, "epochs": 30,
              "lever_configs": [], "rbf_distances": "raw"}
    assert selector.recipe_from_record(record).rbf_distances == "raw"


def test_the_selector_and_futility_paths_load_no_post_fit_analysis_file():
    # Rule A2-GOV-2: the comparator is loaded through clean_sanity.load_comparator, never through the scorer.
    code = ("import sys; sys.path.insert(0, sys.argv[1]); "
            "from ec_clean30 import selector, futility, clean_sanity; clean_sanity.load_comparator(); "
            "import json; print(json.dumps(sorted(name for name in sys.modules if name.startswith('ec_clean30.'))))")
    loaded = json.loads(subprocess.run([sys.executable, "-c", code, str(ROOT / "src")], capture_output=True, text=True,
                                       check=True).stdout.strip().splitlines()[-1])
    assert "ec_clean30.scoring" not in loaded and "ec_clean30.baseline" not in loaded
    assert "ec_clean30.selector" in loaded and "ec_clean30.futility" in loaded


def test_a_real_selection_verifies_once_per_process_and_checks_a_receipt_instead_when_given(monkeypatch, tmp_path):
    from ec_clean30 import amendment_chain
    from ec_clean30.provenance import REPO_ROOT

    tip = Path(REPO_ROOT) / "docs/campaigns/ec_class_vs_clean30_2026-10/amendment2/amendment.json"
    result = {"chain": CHAIN, "links": [{"path": "docs/campaigns/ec_class_vs_clean30_2026-10/amendment2/amendment.json"}]}
    calls = []

    def receipt_record(path, **kwargs):
        calls.append(("verify", kwargs.get("expected_chain")))
        return {"result": dict(result)}

    def recheck(receipt, *, repo, expected_chain=None):
        calls.append(("recheck", expected_chain))
        if calls[-1] == ("recheck", ["f" * 64]):
            raise amendment_chain.AmendmentChainError("[C3] other chain")
        return dict(receipt["result"])

    def check(path, *, repo, expected_chain=None):
        calls.append(("receipt", Path(path).name, expected_chain))
        return dict(result)

    monkeypatch.setattr(selector, "_VERIFIED_CHAINS", {})
    monkeypatch.setattr(amendment_chain, "chain_receipt_record", receipt_record)
    monkeypatch.setattr(amendment_chain, "check_chain_receipt_record", recheck)
    monkeypatch.setattr(amendment_chain, "check_chain_receipt", check)
    assert selector._verify_amendment_chain(tip, FREEZE1, smoke=False)["chain"] == CHAIN
    assert selector._verify_amendment_chain(tip, FREEZE1, smoke=False, expected_chain=CHAIN)["chain"] == CHAIN
    assert calls == [("verify", None), ("recheck", CHAIN)]
    # A failed re-check verifies again (and that verification compares the expected chain itself).
    selector._verify_amendment_chain(tip, FREEZE1, smoke=False, expected_chain=["f" * 64])
    assert calls[-2:] == [("recheck", ["f" * 64]), ("verify", ["f" * 64])]
    receipt = tmp_path / "chain_receipt.json"
    with pytest.raises(SelectionInputError, match="another Freeze 1"):
        selector._verify_amendment_chain(tip, FREEZE1, smoke=False, chain_receipt=receipt)
    from ec_clean30.amendment_rules import FREEZE1_SHA256

    assert selector._verify_amendment_chain(tip, FREEZE1_SHA256, smoke=False, chain_receipt=receipt)["chain"] == CHAIN
    assert calls[-1] == ("receipt", "chain_receipt.json", None)
    with pytest.raises(SelectionInputError, match="authenticates another tip"):
        selector._verify_amendment_chain(tip.with_name("other.json"), FREEZE1_SHA256, smoke=False,
                                         chain_receipt=receipt)
    with pytest.raises(SelectionInputError, match="serves real use only"):
        selector._verify_amendment_chain(tip, FREEZE1_SHA256, smoke=True, chain_receipt=receipt)
