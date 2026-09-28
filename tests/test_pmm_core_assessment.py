"""Core assessment guards use synthetic development predictions, never test data."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pmm_core_assessment as core

FROZEN = Path("/media/mechti/Data1/DeepMzyme_Data/campaigns/_code/pmm_core_scope_v2")


@pytest.fixture(scope="module")
def modules():
    if not FROZEN.is_dir():
        pytest.skip("Frozen PMM scientific checkout is required")
    return core.bind_source(FROZEN)


def make_rows(fold, target):
    labels = core.NATIVE[target]
    rows = []
    for i, element in enumerate(("MN", "CU", "ZN", "FE", "CO", "NI")):
        y4 = min(i, 3)
        yn = min(i, len(labels) - 1)
        row = {"source_uid": f"uid:{fold}:{i}", "physical_ion_id": f"ion:{fold}:{i}",
               "group_id": f"pdb:{fold}:{i}", "native_element": element, "fold": str(fold),
               "model_seed": "42", "checkpoint_sha256": "c" * 64,
               "y_common4": str(y4), "pred_common4": str(y4), "y_native": str(yn), "pred_native": str(yn)}
        for prefix, names, truth in (("p_common4", core.COMMON4, y4), ("p_native", labels, yn)):
            row.update({f"{prefix}_{name.replace(' ', '_')}": str(float(j == truth)) for j, name in enumerate(names)})
        rows.append(row)
    return rows


def make_grid(analysis):
    grid = {}
    for config_id in core.config_ids():
        target = config_id.split("__")[1]
        folds = {}
        for fold in range(5):
            rows = make_rows(fold, target)
            folds[fold] = {"run_name": config_id + f"__fold{fold}__seed42", "rows": rows,
                           "metrics": analysis.prediction_metrics(rows, native_labels=core.NATIVE[target]),
                           "replay_policy_id": "synthetic_strict", "receipt": {"selected_checkpoint_sha256": "c" * 64}}
        grid[config_id] = {"folds": folds}
    return grid


def scores(entry, value=.6, recalls=None):
    for fold in entry["folds"].values():
        fold["metrics"]["common4"].update(balanced_accuracy=value, min_recall=min((recalls or {}).values(), default=value),
                                            recall=recalls or dict.fromkeys(core.COMMON4, value))


def contrasts(grid, analysis):
    return [core.compare_pair(f"{f}__four_class__none", f"{f}__{t}__none", grid, analysis)
            for f in core.FAMILIES for t in ("five_class", "six_class")]


def test_full_grid_has_six_target_contrasts_and_nine_denominator(modules):
    analysis, _ = modules
    grid = make_grid(analysis)
    for entry in grid.values():
        scores(entry)
    scores(grid["only_gvp__five_class__none"], .7)
    rows = contrasts(grid, analysis)
    assert len(rows) == 6
    assert all(row["multiplicity_denominator"] == 9 and row["resamples"] == 10000 and row["seed"] == 42 for row in rows)
    assert all(row["bonferroni_confidence"] == pytest.approx(1 - .05 / 9) for row in rows)
    assert next(row for row in rows if row["challenger"] == "only_gvp__five_class__none")["promoted"]


def test_bootstrap_uses_paired_folds_and_fixed_seed(modules):
    analysis, _ = modules
    grid = make_grid(analysis)
    scores(grid[core.CONTROL], .6)
    challenger = "only_gvp__six_class__none"
    delta = [.04, -.01, .08, .01, -.02]
    for fold, difference in enumerate(delta):
        grid[challenger]["folds"][fold]["metrics"]["common4"].update(balanced_accuracy=.6 + difference, recall=dict.fromkeys(core.COMMON4, .6))
    actual = core.compare_pair(core.CONTROL, challenger, grid, analysis)
    expected = analysis.fold_bootstrap(np.array(actual["fold_differences"]), resamples=10000, seed=42, n_contrasts=9)
    assert all(actual[k] == value for k, value in expected.items())
    assert actual["fold_differences"] == pytest.approx(delta)


@pytest.mark.parametrize("recalls", [dict(zip(core.COMMON4, [.9, .9, .9, .56])),
                                    dict(zip(core.COMMON4, [.9, .9, .9, 0])),
                                    dict(zip(core.COMMON4, [.9, .9, .9, None]))])
def test_rare_missing_and_zero_class_recall_block_promotion(modules, recalls):
    analysis, _ = modules
    grid = make_grid(analysis)
    scores(grid[core.CONTROL], .6)
    candidate = "only_gvp__five_class__none"
    for unit in grid[candidate]["folds"].values():
        unit["metrics"]["common4"].update(balanced_accuracy=.8, recall=recalls)
    assert not core.compare_pair(core.CONTROL, candidate, grid, analysis)["promoted"]


@pytest.mark.parametrize("mutation", ["missing_fold", "missing_config", "awareness_config"])
def test_incomplete_or_wrong_grid_cannot_select(modules, mutation):
    analysis, _ = modules
    grid = make_grid(analysis)
    if mutation == "missing_fold":
        del grid[core.CONTROL]["folds"][4]
    elif mutation == "missing_config":
        del grid[core.CONTROL]
    else:
        grid["only_gvp__four_class__first_shell_bias"] = grid.pop(core.CONTROL)
    with pytest.raises(core.CoreAssessmentError, match="exact nine|every planned fold"):
        core.select_core_configuration(grid, [], analysis, tie_epsilon=.002)


def test_native_ranking_never_overrides_common_four_and_matched_gate(modules):
    analysis, _ = modules
    grid = make_grid(analysis)
    for entry in grid.values():
        scores(entry)
    candidate = "only_gvp__five_class__none"
    for unit in grid[candidate]["folds"].values():
        unit["metrics"]["native"]["balanced_accuracy"] = .99
    assert core.select_core_configuration(grid, contrasts(grid, analysis), analysis, tie_epsilon=.002)["selected_config_id"] == core.CONTROL
    scores(grid[candidate], .7)
    assert core.select_core_configuration(grid, [], analysis, tie_epsilon=.002)["selected_config_id"] == core.CONTROL
    assert core.select_core_configuration(grid, contrasts(grid, analysis), analysis, tie_epsilon=.002)["selected_config_id"] == candidate


def test_direct_four_can_replace_control_and_frozen_tie_rule_is_explicit(modules):
    analysis, _ = modules
    from benchmarking.pmm_final_report import TIE_EPSILON
    grid = make_grid(analysis)
    for entry in grid.values():
        scores(entry)
    scores(grid["only_esm__four_class__none"], .7)
    scores(grid["gvp_late_fusion__four_class__none"], .701,
           {"Mn": .702, "Cu": .702, "Zn": .702, "Class VIII": .698})
    selected = core.select_core_configuration(grid, contrasts(grid, analysis), analysis, tie_epsilon=TIE_EPSILON)
    assert selected["selected_config_id"] == "only_esm__four_class__none"
    assert selected["tie_epsilon"] == .002 and "frozen" in selected["tie_policy_source"]


@pytest.mark.parametrize("kind", ["missing_pmm", "duplicate_uid", "wrong_fold", "missing_uid"])
def test_oof_and_pmm_coverage_required(modules, kind):
    analysis, _ = modules
    grid = make_grid(analysis)
    pmm = copy.deepcopy(grid[core.CONTROL]["folds"])
    cohort = {row["source_uid"]: row for unit in pmm.values() for row in unit["rows"]}
    if kind == "missing_pmm":
        del pmm[4]
    elif kind == "duplicate_uid":
        grid[core.CONTROL]["folds"][1]["rows"][0]["source_uid"] = grid[core.CONTROL]["folds"][0]["rows"][0]["source_uid"]
    elif kind == "wrong_fold":
        grid[core.CONTROL]["folds"][1]["rows"][0]["fold"] = "0"
    else:
        pmm[0]["rows"].pop()
    with pytest.raises(ValueError, match="PMM folds|predicted in folds|row fold|OOF predictions"):
        core.evaluate_core(grid, pmm, cohort, analysis)


def test_report_preserves_native_five_meaning_and_replay_provenance(tmp_path, modules, monkeypatch):
    analysis, campaign = modules
    grid = make_grid(analysis)
    grid["only_gvp__five_class__none"]["folds"][0].update(replay_policy_id="retrospective_fixture",
        replay_qualification={"post_observation": True, "limitations": ["Original strict failure retained"]})
    pmm = copy.deepcopy(grid[core.CONTROL]["folds"])
    for unit in pmm.values():
        unit["metrics"].pop("native")
    cohort = {row["source_uid"]: row for unit in pmm.values() for row in unit["rows"]}
    monkeypatch.setattr(core, "collect_core", lambda *a, **k: (grid, pmm, cohort, {}, analysis, campaign))
    out = tmp_path / "report"
    result = core.assess_core(tmp_path, tmp_path / "train", source_root=FROZEN, out_dir=out)
    assert result["qualified_core_fits"] == 45 and result["pmm_folds"] == 5
    assert result["summaries"]["only_gvp__five_class__none"]["native_label_meaning"] == "Class VIII = Co+Ni"
    assert any(q["replay_policy_id"] == "retrospective_fixture" and q["replay_qualification"]["post_observation"] for q in result["replay_qualifications"])
    assert all(core.sha256(out / name) == value for name, value in result["output_files"].items())
    with (out / "core_cv_oof_predictions.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 300
    assert not (tmp_path / "validation_decision.json").exists()
    with pytest.raises(core.CoreAssessmentError, match="already exists"):
        core.assess_core(tmp_path, tmp_path / "train", source_root=FROZEN, out_dir=out)


def test_failed_collection_creates_no_complete_decision(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("unqualified replay")
    monkeypatch.setattr(core, "collect_core", fail)
    with pytest.raises(ValueError, match="unqualified"):
        core.assess_core(tmp_path, tmp_path / "train", source_root=FROZEN, out_dir=tmp_path / "report")
    assert not (tmp_path / "report").exists()


@pytest.fixture
def collected_fixture(tmp_path, modules, monkeypatch):
    analysis, campaign = modules
    import pmm_core_replay
    from benchmarking import pmm_comparator as comparator
    root, train = tmp_path / "campaign", tmp_path / "train"
    root.mkdir()
    train.mkdir()
    (root / "pmm_comparator").mkdir()
    for name in ("campaign_manifest.json", "train_cohort.csv", "fold_membership.csv", "fold_class_weights.json",
                 "feature_inventory.json", "pmm_comparator/pmm_comparator_manifest.json"):
        (root / name).write_text("synthetic frozen evidence\n")
    pmm_source = tmp_path / "classmodel_train_set"
    pmm_source.write_text("synthetic training-side source\n")
    monkeypatch.setattr(comparator, "PMM_SOURCE_DEFAULT", pmm_source)
    membership = {row["source_uid"]: row for fold in range(5) for row in make_rows(fold, "four_class")}
    monkeypatch.setattr(comparator, "read_campaign_contract", lambda *a: (
        {"campaign_id": "pmm_ion_metal_v2_context", "source_side": "train"}, membership, membership))
    monkeypatch.setattr(comparator, "verify_comparator_outputs", lambda *a: {})
    units = {}
    for config_id in core.config_ids():
        family, target, _ = config_id.split("__")
        for fold in range(5):
            rows = make_rows(fold, target)
            identity = {"campaign_id": "pmm_ion_metal_v2_context", "family": family, "target_scheme": target,
                        "readout": "none", "fold": fold, "model_seed": 42, "epochs": 50,
                        "source_tree_sha256": core.SOURCE_SHA256, "cohort_sha256": core.sha256(root / "train_cohort.csv"),
                        "fold_membership_sha256": core.sha256(root / "fold_membership.csv")}
            units[family, target, fold] = {"identity": identity, "run_name": config_id + f"__fold{fold}__seed42",
                "rows": rows, "metrics": analysis.prediction_metrics(rows, native_labels=core.NATIVE[target]),
                "receipt": {"selected_checkpoint_sha256": "c" * 64}, "replay_policy_id": "synthetic",
                "evidence_files": {str(pmm_source): core.sha256(pmm_source)}}
    monkeypatch.setattr(pmm_core_replay, "validate_core_unit", lambda c, t, family, target, fold, **kw: units[family, target, fold])
    for fold in range(5):
        analysis.write_rows(root / "pmm_comparator" / f"fold{fold}_predictions.csv", make_rows(fold, "four_class"))
    return root, train, units


def test_collection_accepts_exact_45_normalized_units(collected_fixture):
    root, train, _units = collected_fixture
    grid, pmm, cohort, evidence, _a, _c = core.collect_core(root, train, source_root=FROZEN)
    assert len(grid) == 9 and sum(len(entry["folds"]) for entry in grid.values()) == 45
    assert len(pmm) == 5 and len(cohort) == 30 and evidence


@pytest.mark.parametrize("mutation", ["identity", "source", "metrics", "repeated_uid", "qualification"])
def test_collection_refuses_wrong_normalized_unit(collected_fixture, mutation):
    root, train, units = collected_fixture
    unit = units["only_esm", "four_class", 0]
    if mutation == "identity":
        unit["identity"]["model_seed"] = 43
    elif mutation == "source":
        unit["identity"]["source_tree_sha256"] = "a" * 64
    elif mutation == "metrics":
        unit["metrics"]["common4"]["balanced_accuracy"] = .3
    elif mutation == "repeated_uid":
        unit["rows"][1]["source_uid"] = unit["rows"][0]["source_uid"]
    else:
        unit["evidence_files"] = {}
    with pytest.raises(ValueError, match="identity differs|metrics differ|exactly once|qualification lacks"):
        core.collect_core(root, train, source_root=FROZEN)


def test_wrong_source_is_rejected_before_import(tmp_path):
    with pytest.raises(core.CoreAssessmentError, match="source hash"):
        core.bind_source(tmp_path)


@pytest.mark.parametrize("field,value", [("targets", ["four_class", "six_class"]), ("model_seeds", [43]), ("epochs", 49)])
def test_scope_identity_changes_are_rejected(tmp_path, field, value):
    scope = core.read_json(core.SCOPE_PATH)
    scope[field] = value
    path = tmp_path / "scope.json"
    core.write_json(path, scope)
    with pytest.raises(core.CoreAssessmentError, match="scope field"):
        core.load_scope(path)


def test_scope_cannot_reduce_multiplicity_for_paused_arms(tmp_path):
    scope = core.read_json(core.SCOPE_PATH)
    scope["assessment"]["simultaneous_correction_denominator"] = 6
    path = tmp_path / "scope.json"
    core.write_json(path, scope)
    with pytest.raises(core.CoreAssessmentError, match="simultaneous_correction_denominator"):
        core.load_scope(path)


@pytest.fixture
def refit_campaign(tmp_path, modules, monkeypatch):
    _, campaign = modules
    train = tmp_path / "train"
    train.mkdir()
    root = tmp_path / "campaign"
    root.mkdir()
    (root / "train_cohort.csv").write_text("synthetic cohort,read via stub\n")
    (root / "fold_membership.csv").write_text("synthetic folds\n")
    core.write_json(root / "campaign_manifest.json", {"campaign_id": "pmm_ion_metal_v2_context", "cohort": {"sha256": core.sha256(root / "train_cohort.csv")}})
    core.write_json(root / "feature_inventory.json", {"certified": True, "certified_scope": "structure_and_esmc600m", "esm": {"embeddings_dir": str(tmp_path / "esm")}})
    core.write_json(root / "fold_class_weights.json", {"folds": {"0": {
        "runnable": True, "four_class_multipliers": dict.fromkeys(("mn", "cu", "zn", "class_viii"), 1.),
        "six_class_multipliers": dict.fromkeys(("mn", "cu", "zn", "fe", "co", "ni"), 1.)}}})
    bindings = [SimpleNamespace(native_element=e, structure_stem=f"pdb{i}__chain_A", pdbid=f"pdb{i}",
                                example_id=lambda i=i: f"pdb{i}__chain_A__SRC_{i}")
                for i, e in enumerate(("MN", "CU", "ZN", "FE", "CO", "NI"))]
    monkeypatch.setattr(campaign, "read_cohort_csv", lambda *a: bindings)
    return root, train, campaign


@pytest.mark.parametrize("target", core.TARGETS)
def test_full_refit_preview_weights_and_no_fold_no_test(refit_campaign, target):
    root, train, campaign = refit_campaign
    from training.config import parse_args
    command, env, identity = core._build_refit(campaign, root, train, f"only_gvp__{target}__none", "d" * 64,
                                              python_bin=sys.executable, device="cpu")
    cfg = parse_args(command[3:])
    assert cfg.epochs == 50 and cfg.seed == 42 and cfg.val_fraction == 0 and cfg.n_folds is None
    assert cfg.fold_membership_csv is None and not cfg.export_validation_predictions and cfg.selection_metric == "train_loss"
    assert Path(cfg.source_cohort_csv) == root / "train_cohort.csv"
    assert not cfg.run_test_eval and cfg.test_structure_dir is None and cfg.test_summary_csv is None
    assert identity["checkpoint_rule"] == "terminal_epoch_50" and identity["core_validation_decision_sha256"] == "d" * 64
    assert cfg.mn_loss_multiplier == 1.5 and cfg.cu_loss_multiplier == 1.5 and cfg.zn_loss_multiplier == 1.5
    if target == "five_class":
        assert cfg.fe_loss_multiplier == cfg.class_viii_loss_multiplier == .5
    elif target == "six_class":
        assert cfg.fe_loss_multiplier == cfg.co_loss_multiplier == cfg.ni_loss_multiplier == .5
    else:
        assert cfg.class_viii_loss_multiplier == .5
    assert any("FORBIDDEN" in key for key in env)


def test_refit_preview_requires_supported_route_before_reading_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "verify_decision", lambda *a, **k: pytest.fail("Decision opened before route gate"))
    with pytest.raises(core.CoreAssessmentError, match="route"):
        core.preview_core_refit(tmp_path, tmp_path / "train", source_root=FROZEN, decision_path=tmp_path / "absent.json",
                                out_dir=tmp_path / "out", route="primary_pristine_test")
    assert not (tmp_path / "out").exists()


def test_premature_decision_refuses_refit_before_training_files(tmp_path):
    path = tmp_path / "core_validation_decision.json"
    core.write_json(path, {"status": "complete", "qualified_core_fits": 44})
    with pytest.raises(core.CoreAssessmentError, match="complete, frozen"):
        core.verify_decision(path, tmp_path, tmp_path / "train", source_root=FROZEN)


def _completed_refit_fixture(refit_campaign, monkeypatch):
    import torch
    from training.config import config_to_payload, parse_args
    root, train, campaign = refit_campaign
    selected = "only_gvp__five_class__none"
    decision_path = root / "core_validation_decision.json"
    core.write_json(decision_path, {"selection": {"selected_config_id": selected}})
    command, env, identity = core._build_refit(campaign, root, train, selected, core.sha256(decision_path), python_bin=sys.executable, device="cpu")
    preview = {"status": "preview_only", "launch_final_refit": False, "held_out_access_authorized": False,
               "route": core.ROUTE, "identity": identity, "decision_path": str(decision_path),
               "validation_decision_sha256": core.sha256(decision_path)}
    preview_path = root / "core_stage6b_decision.json"
    core.write_json(preview_path, preview)
    monkeypatch.setattr(core, "verify_decision", lambda *a, **k: (core.read_json(decision_path), campaign))
    run = root / "core_final_refit" / f"core_stage6b__{selected}__seed42"
    run.mkdir(parents=True)
    cfg = config_to_payload(parse_args(command[3:]))
    cfg.update(model_seed=42, git_dirty=False, git_commit="synthetic_augmented_metadata")
    history = [{"epoch": k} for k in range(1, 51)]
    normalization = {"means": {"feature": 0.}, "stds": {"feature": 1.}, "clamp_value": 5.}
    examples = [{"pocket_id": binding.example_id(), "structure_id": binding.structure_stem,
                 "group": binding.pdbid, "y_metal": min(i, 4)}
                for i, binding in enumerate(campaign.read_cohort_csv(root / "train_cohort.csv"))]
    dataset = {"n_train_pockets": 6, "n_val_pockets": 0,
               "retained_split_identity": {"train": {"examples": examples}, "validation": {"examples": []}}}
    payload = {"config": cfg, "normalization_stats": normalization, "test_report": None, "dataset_summary": dataset}
    core.write_json(run / "run_metadata.json", {**payload, "campaign_run_identity": identity, "fit_status": "completed"})
    core.write_json(run / "run_config.json", {**payload, "history": history})
    torch.save({**payload, "history": history, "model_state_dict": {}}, run / "last_model_checkpoint.pt")
    return root, train, run, preview_path


def test_completed_refit_verification_does_not_authorize_test_or_pmm(refit_campaign, monkeypatch):
    root, train, run, preview = _completed_refit_fixture(refit_campaign, monkeypatch)
    result = core.verify_core_refit(root, train, source_root=FROZEN, preview_path=preview, run_dir=run)
    assert result["status"] == "completed" and result["terminal_checkpoint"]
    assert result["pmm_refit_verified"] is False and result["final_reporting_authorized"] is False
    assert not (root / "stage6b_selected_final_refit_candidate.json").exists()


@pytest.mark.parametrize("mutation", ["partial", "test_report", "normalization", "configuration", "decision", "cohort", "target"])
def test_refit_drift_and_wrong_checkpoint_block(refit_campaign, monkeypatch, mutation):
    import torch
    root, train, run, preview = _completed_refit_fixture(refit_campaign, monkeypatch)
    ckpt_path = run / "last_model_checkpoint.pt"
    checkpoint = torch.load(ckpt_path, weights_only=False)
    if mutation == "partial":
        checkpoint["history"].pop()
    elif mutation == "test_report":
        (run / "test_report.json").write_text("{}")
    elif mutation == "normalization":
        checkpoint["normalization_stats"]["means"]["feature"] = 8.
    elif mutation == "configuration":
        checkpoint["config"]["epochs"] = 49
    elif mutation == "decision":
        (root / "core_validation_decision.json").write_text("{}")
    else:
        summary = checkpoint["dataset_summary"]
        if mutation == "cohort":
            summary["retained_split_identity"]["train"]["examples"].pop()
        else:
            summary["retained_split_identity"]["train"]["examples"][0]["y_metal"] = 3
        for name in ("run_config", "run_metadata"):
            payload = core.read_json(run / f"{name}.json")
            payload["dataset_summary"] = summary
            core.write_json(run / f"{name}.json", payload)
    torch.save(checkpoint, ckpt_path)
    with pytest.raises(core.CoreAssessmentError, match="epoch 50|test report|normalization|configuration differs|decision changed|complete frozen cohort|structure/group/target"):
        core.verify_core_refit(root, train, source_root=FROZEN, preview_path=preview, run_dir=run)
