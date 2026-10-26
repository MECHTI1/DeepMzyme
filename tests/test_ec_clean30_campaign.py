"""Unit matrix, command builder, smoke units, verification and Freeze manifests of the EC class vs CLEAN30 runner.

Synthetic data only: every view is built from the fixture tree of tests/ec_clean30_fixtures.py, and the
command-line tests of the training-side input actions (cohort-audit, r2-lists, overlay, simulate) use the
overlay tests' synthetic five-fold world (tests/test_ec_clean30_overlay.py).
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import replace
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from ec_clean30_fixtures import (  # noqa: E402
    build_chain_means, build_smoke_view, make_writable, unlock, write_clean30_data_root, write_inner_membership,
)
from ec_clean30 import campaign, provenance  # noqa: E402
from ec_clean30.campaign import (  # noqa: E402
    KIND_INNER, KIND_REFIT, LEVER_CONFIGS, MAX_FITS_PER_FOLD, CampaignPlanError, Recipe, build_fold_plan,
    default_smoke_recipe, plan_units, recipe_with_lever, smoke_unit_id,
)
from ec_clean30.commands import (  # noqa: E402
    CommandError, build_unit_command, check_command, collect_refit_checkpoints, dry_run, refit_coverage_problems,
    run_unit, verify_unit, verify_units, view_unit_commands,
)
from ec_clean30.views import SourceLayout, View, ViewError, build_view, freeze1_view_record  # noqa: E402
from ec_clean30 import cohort, identity, overlay, r2_lists, simulation, views  # noqa: E402
from test_ec_clean30_overlay import build_world  # noqa: E402  (synthetic world: data root, cohort, identity, R2)

PROVENANCE = {"note": "synthetic test provenance"}
PYTHON = sys.executable
SCRIPT = ROOT / "scripts" / "run_ec_clean30_campaign.py"


def _fold_dir(base: Path, fold: int = 0) -> Path:
    """Where the fold-k view of a campaign rooted below ``base`` lives (the location rule)."""
    return Path(base) / campaign.CAMPAIGN_ID / views.VIEWS_DIR / f"fold{fold}"


def _smoke_dir(base: Path, name: str = "view") -> Path:
    return Path(base) / campaign.CAMPAIGN_ID / views.SMOKE_VIEWS_DIR / name


def _script():
    spec = importlib.util.spec_from_file_location("run_ec_clean30_campaign_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    paths = write_clean30_data_root(tmp_path_factory.mktemp("clean30_source"))
    yield paths
    unlock(paths)


@pytest.fixture(scope="module")
def smoke_view(source, tmp_path_factory):
    return build_smoke_view(source, _smoke_dir(tmp_path_factory.mktemp("g1")), provenance=PROVENANCE)


# ---------------------------------------------------------------------------
# Plan


def test_plan_counts_match_the_campaign_fit_budget():
    plan = build_fold_plan(0)
    assert plan["counts"] == {"S1": 96, "S2": 72, "S3": 8, "S4": 5, "total": 181}
    assert plan["counts"]["total"] <= MAX_FITS_PER_FOLD
    units = plan_units(plan)
    s1 = [unit for unit in units if unit.step == "S1"]
    assert {unit.family for unit in s1} == {"F1", "F2", "F3", "F4", "F5", "C-G"}
    assert {(unit.inner_fold, unit.seed) for unit in s1} == {(inner, seed) for inner in range(4) for seed in (42, 43)}
    assert sorted({campaign.learning_rate_tag(unit.learning_rate) for unit in s1 if unit.family == "F4"}) == \
        ["lr1e-4-1e-4", "lr3e-4-1e-4"]
    s2_configs = sorted({unit.lever_configs[0] for unit in units if unit.step == "S2"})
    assert s2_configs == sorted(LEVER_CONFIGS) and len(s2_configs) == 9
    assert [unit.seed for unit in units if unit.step == "S4"] == [42, 43, 44, 45, 46]
    assert all(unit.kind == KIND_REFIT and unit.inner_fold is None for unit in units if unit.step == "S4")
    ids = [unit.unit_id for unit in units]
    assert len(ids) == len(set(ids)) and all("/" not in value and value.startswith("ecc30__fold0__") for value in ids)
    assert "ecc30__fold0__S1__F3_lr3e-4-1e-4_e30__inner2__seed43" in ids
    assert "ecc30__fold0__S4__selected__refit__seed44" in ids
    assert next(unit for unit in units if unit.unit_id.startswith("ecc30__fold0__S2__L7e12")).epochs == 12
    # L3 runs last within S2, charged to its own sub-ceiling.
    s2 = [unit for unit in units if unit.step == "S2"]
    order = list(dict.fromkeys(unit.lever_configs[0] for unit in s2))
    assert order == list(campaign.S2_EXECUTION_ORDER) and order[-1] == "L3"
    assert {unit.sub_ceiling for unit in s2 if unit.lever_configs == ("L3",)} == {"L3"}
    assert {unit.sub_ceiling for unit in s2 if unit.lever_configs != ("L3",)} == {None}
    assert plan["constants"]["s2_execution_order"][-1] == "L3"
    assert plan["constants"]["s2_sub_ceilings"]["L3"]["max_fits_per_fold"] == 8
    assert [unit.step for unit in units] == sorted((unit.step for unit in units), key=campaign.STEPS.index)


def test_plan_is_deterministic_and_hash_bound():
    first, second = build_fold_plan(3), build_fold_plan(3)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["plan_sha256"] != build_fold_plan(2)["plan_sha256"]
    tampered = json.loads(json.dumps(first))
    tampered["units"][0]["seed"] = 99
    with pytest.raises(CampaignPlanError, match="plan_sha256"):
        plan_units(tampered)


def test_plan_restricts_levers_to_the_s1_winner():
    f2 = build_fold_plan(0, s1_winner="F2")
    assert f2["counts"]["S2"] == 6 * 8 and set(f2["levers_not_applicable"]) == {"L1", "L2", "L8"}
    assert build_fold_plan(0, s1_winner="F4")["counts"]["S2"] == 72
    assert build_fold_plan(0, s1_winner="F5")["counts"]["S2"] == 7 * 8
    only = build_fold_plan(0, families=["F2", "F5"], levers=["L7", "L9"])
    assert only["counts"] == {"S1": 32, "S2": 16, "S3": 8, "S4": 5, "total": 61}


@pytest.mark.parametrize("kwargs, message", [
    ({"families": ["F9"]}, "Unknown family"),
    ({"levers": ["L10"]}, "Unknown lever"),
    ({"levers": ["L1", "L1"]}, "repeat"),
    ({"s1_winner": "C-G"}, "never selected"),
    ({"s1_winner": "F1"}, "never selected"),
])
def test_plan_refuses_unknown_families_and_levers(kwargs, message):
    with pytest.raises(CampaignPlanError, match=message):
        build_fold_plan(0, **kwargs)


@pytest.mark.parametrize("fold", [-1, 5, True])
def test_plan_refuses_folds_outside_the_benchmark(fold):
    with pytest.raises(CampaignPlanError, match="Outer fold"):
        build_fold_plan(fold)


def test_recipes_refuse_inapplicable_or_unknown_levers():
    lr = tuple(campaign.S1_LEARNING_RATE_GRID["F2"][0])
    with pytest.raises(CampaignPlanError, match="does not apply"):
        Recipe("F2", lr, lever_configs=("L1",))
    with pytest.raises(CampaignPlanError, match="Unknown lever configuration"):
        Recipe("F2", lr, lever_configs=("L9",))
    with pytest.raises(CampaignPlanError, match="appears twice"):
        Recipe("F2", lr, lever_configs=("L7e12", "L7e20"))
    with pytest.raises(CampaignPlanError, match="not in the S1 grid"):
        Recipe("F2", (("learning_rate", 1e-3),))
    assert "L9" in campaign.LEVERS_WITHOUT_FITS


# ---------------------------------------------------------------------------
# Commands


def _command(view, family, *, refit=False, runs_dir=None, lr_index=1, levers=()):
    """A smoke unit command; its runs folder defaults to the view's own (``<campaign root>/smoke_runs/<name>``)."""
    recipe = default_smoke_recipe(family, lr_index=lr_index)
    for config_id in levers:
        recipe = recipe_with_lever(recipe, config_id)
    runs_dir = view.default_runs_dir if runs_dir is None else runs_dir
    unit_id = smoke_unit_id(view.fold, recipe, refit=refit, seed=42)
    return build_unit_command(view, recipe, unit_id=unit_id, kind=KIND_REFIT if refit else KIND_INNER, seed=42,
                              runs_dir=runs_dir, step="smoke", inner_fold=None if refit else 0)


def test_f4_inner_command_matches_the_reference_argv(smoke_view):
    view = smoke_view
    command = _command(view, "F4")
    root = str(view.root)
    reference = [
        "--task", "ec", "--ec-label-depth", "1", "--ec-class-vocabulary", "fixed_seven", "--ec-conflicting-labels",
        "fail", "--site-extraction", "selected_sites", "--model-architecture", "gvp", "--fusion-mode", "late_fusion",
        "--gvp-learning-rate", "3e-4", "--learning-rate", "1e-4", "--use-global-esm-branch",
        "--global-esm-cache-dir", f"{root}/chain_means_final", "--global-esm-manifest-sha256",
        view.chain_means_manifest_sha256("final"),
        "--ec-group-weighting", "structure_id", "--ec-class-weight-unit", "group", "--batch-size", "4", "--epochs", "1",
        "--lr-schedule", "cosine", "--checkpoint-rule", "terminal", "--edge-radius", "6.0", "--ec-contrastive-weight",
        "0", "--no-prepare-missing-esm-embeddings", "--no-prepare-missing-ring-edges",
        "--no-prepare-missing-external-features", "--fold-split-source", "membership", "--fold-membership-csv",
        f"{root}/smoke_membership.csv", "--fold-membership-sha256", view.membership["sha256"], "--n-folds", "2",
        "--fold-index", "0", "--export-ec-predictions", "--structure-dir", f"{root}/structures", "--summary-csv",
        f"{root}/site_summary_train.csv", "--esm-embeddings-dir", f"{root}/esm", "--external-features-root-dir",
        f"{root}/external", "--dataset-bundle-id", "g1-smoke-view", "--dataset-bundle-sha256", view.manifest_sha256,
        "--device", "cpu", "--load-workers", "1", "--num-workers", "0", "--seed", "42",
    ]
    assert list(command.argv[: len(reference)]) == reference
    extras = list(command.argv[len(reference):])
    assert extras[:6] == ["--ring-features-dir", f"{root}/ring", "--runs-dir", str(view.default_runs_dir),
                          "--run-name", command.unit_id]
    assert extras[6] == "--campaign-run-identity" and len(extras) == 8
    identity = json.loads(extras[7])
    assert identity["unit_id"] == command.unit_id and identity["view_manifest_sha256"] == view.manifest_sha256
    assert identity["membership_sha256"] == view.membership["sha256"] and identity["smoke"] is True
    assert command.env["DEEPMZYME_PARSE_CACHE_DIR"] == f"{root}/caches/parse"
    assert command.env["DEEPMZYME_GRAPH_CACHE_DIR"] == f"{root}/caches/graph"
    assert command.env["DEEPMZYME_FORBIDDEN_READ_ROOTS"].split(os.pathsep) == view.forbidden_read_roots
    assert command.env["MKL_THREADING_LAYER"] == "GNU"


@pytest.mark.parametrize("refit", [False, True], ids=["inner", "refit"])
@pytest.mark.parametrize("family", ["F1", "F2", "F3", "F4", "F5", "C-G"])
def test_every_family_builds_a_contract_command(smoke_view, family, refit):
    from training.config import parse_args

    command = _command(smoke_view, family, refit=refit)
    assert check_command(command, smoke_view) == []
    config = parse_args(list(command.argv))
    spec = campaign.FAMILY_SPECS[family]
    assert config.model_architecture == spec.model_architecture
    assert config.use_global_esm_branch is spec.global_branch
    if spec.fusion_mode:
        assert config.fusion_mode == spec.fusion_mode
    if family in {"F3", "F4"}:
        assert (config.gvp_learning_rate, config.learning_rate) == (3e-4, 1e-4)
    else:
        assert config.gvp_learning_rate is None and config.learning_rate == 3e-4
    assert (config.task, config.ec_class_vocabulary, config.site_extraction) == ("ec", "fixed_seven", "selected_sites")
    assert (config.lr_schedule, config.checkpoint_rule, config.batch_size, config.edge_radius) == \
        ("cosine", "terminal", 4, 6.0)
    assert not (config.prepare_missing_esm_embeddings or config.prepare_missing_external_features
                or config.prepare_missing_ring_edges or config.run_test_eval)
    assert config.export_ec_predictions and config.dataset_bundle_sha256 == smoke_view.manifest_sha256
    if refit:
        assert config.val_fraction == 0.0 and config.n_folds is None
        assert Path(config.ec_probe_structure_ids) == smoke_view.probe_file
        assert "--fold-split-source" not in command.argv
    else:
        assert (config.fold_split_source, config.n_folds, config.fold_index) == ("membership", 2, 0)
        assert config.ec_probe_structure_ids is None


@pytest.mark.parametrize("family, config_id, expected", [
    ("F4", "L1", [("--gvp-target-shrink", "0.059"), ("--gvp-target-shrink-scope", "trunk")]),
    ("F4", "L2", [("--rbf-use-raw-distances", None), ("--edge-radius", "8.0")]),
    ("F2", "L3", [("--position-noise-std", "0.1"), ("--outer-residue-dropout", "0.1"), ("--augmentation-rng", "per_fit")]),
    ("F2", "L4", [("--ec-aux-levels", "2,3"), ("--ec-aux-weight", "0.3")]),
    ("F5", "L5", [("--ec-supcon-weight", "0.1"), ("--ec-supcon-memory", "256")]),
    ("F2", "L6", [("--head-mlp-dropout", "0.3"), ("--esm-graph-encoder-dropout", "0.2")]),
    ("F1", "L6", [("--head-mlp-dropout", "0.3")]),
])
def test_lever_configurations_add_their_flags(smoke_view, family, config_id, expected):
    from training.config import parse_args

    command = _command(smoke_view, family, levers=(config_id,))
    argv = list(command.argv)
    for flag, value in expected:
        assert argv.count(flag) == 1
        if value is not None:
            assert argv[argv.index(flag) + 1] == value
    assert check_command(command, smoke_view) == []
    parse_args(argv)
    if family == "F1" and config_id == "L6":
        assert "--esm-graph-encoder-dropout" not in argv


def test_epoch_lever_and_missing_layer_cache(smoke_view):
    command = _command(smoke_view, "F2", levers=("L7e20",))
    assert command.argv[list(command.argv).index("--epochs") + 1] == "20"
    with pytest.raises(CommandError, match="block26 chain-mean cache"):
        _command(smoke_view, "F4", levers=("L8",))


def _set_flag(argv: list[str], flag: str, value: str) -> list[str]:
    argv = list(argv)
    argv[argv.index(flag) + 1] = value
    return argv


@pytest.mark.parametrize("edit, message", [
    (lambda argv: argv + ["--run-test-eval"], "held-out test set"),
    (lambda argv: argv + ["--test-structure-dir", "/somewhere"], "held-out test set"),
    (lambda argv: argv + ["--allow-missing-external-features"], "refused flag --allow-missing-external-features"),
    (lambda argv: [token for token in argv if token != "--no-prepare-missing-ring-edges"],
     "--no-prepare-missing-ring-edges is missing"),
    (lambda argv: _set_flag(argv, "--dataset-bundle-sha256", "0" * 64), "--dataset-bundle-sha256 differs"),
    (lambda argv: _set_flag(argv, "--checkpoint-rule", "best_validation"), "the trainer refuses the arguments"),
    (lambda argv: _set_flag(argv, "--lr-schedule", "fixed"), "resolved lr_schedule is 'fixed'"),
])
def test_static_check_refuses_test_flags_generation_and_bundle_mismatch(smoke_view, edit, message):
    command = _command(smoke_view, "F2")
    edited = replace(command, argv=tuple(edit(list(command.argv))))
    problems = check_command(edited, smoke_view)
    assert any(message in problem for problem in problems), problems


def test_static_check_follows_the_resolved_configuration(smoke_view, source, tmp_path):
    command = _command(smoke_view, "F4")
    argv = list(command.argv)
    # Abbreviated flags resolve to real options in the trainer's parser; they are refused, and the
    # resolved paths are checked as well.
    abbreviated = argv + ["--esm-embeddings", str(source["esm_dir"]), "--structure-d", str(tmp_path)]
    problems = check_command(replace(command, argv=tuple(abbreviated)), smoke_view)
    assert any("--esm-embeddings is not an exact trainer option" in problem for problem in problems)
    assert any("resolved esm_embeddings_dir" in problem and "outside the view" in problem for problem in problems)
    assert any("resolved structure_dir" in problem for problem in problems)
    inline = argv + [f"--ring-features-dir={tmp_path}"]
    assert any("resolved ring_features_dir" in problem
               for problem in check_command(replace(command, argv=tuple(inline)), smoke_view))
    # Settings that differ from the recipe are refused even when they are valid trainer options.
    for flag, value in (("--learning-rate", "3e-4"), ("--model-architecture", "only_esm"),
                        ("--fusion-mode", "early_fusion")):
        edited = replace(command, argv=tuple(_set_flag(argv, flag, value)))
        assert any("differs from the recipe's command" in problem or "trainer refuses" in problem
                   for problem in check_command(edited, smoke_view)), flag
    no_branch = [token for token in argv if token != "--use-global-esm-branch"]
    assert check_command(replace(command, argv=tuple(no_branch)), smoke_view)
    layered = argv + ["--global-esm-layer", "block26"]
    assert any("differs from the recipe's command" in problem
               for problem in check_command(replace(command, argv=tuple(layered)), smoke_view))
    other_path = argv + ["--source-cohort-csv", str(tmp_path / "cohort.csv")]
    assert check_command(replace(command, argv=tuple(other_path)), smoke_view)
    assert any(problem.startswith("-x is refused")
               for problem in check_command(replace(command, argv=tuple(argv + ["-x"])), smoke_view))


def test_fold_view_plan_commands_cover_the_matrix(source, tmp_path):
    membership, digest, _certificate = write_inner_membership(source, tmp_path / "identity")
    root = _fold_dir(tmp_path)
    build_view(fold=0, layout=SourceLayout.from_data_root(source["data_root"]), out_dir=root,
               inner_membership=(membership, digest), provenance=PROVENANCE)
    build_chain_means(root)
    view = View.load(root)
    planned = view_unit_commands(view, runs_dir=tmp_path / "runs")
    s1 = [item for item in planned if item.unit_id.split("__")[2] == "S1"]
    assert len(s1) == 96 and all(item.command is not None for item in s1)
    l8 = [item for item in planned if item.template_of and "__S2__L8__" in item.unit_id]
    assert l8 and all(item.problem and "block26" in item.problem for item in l8)
    others = [item for item in planned if item not in l8]
    assert all(item.problem is None for item in others), [item.problem for item in others if item.problem][:3]
    assert {item.family for item in planned if "__S4__" in item.unit_id} == {"F2", "F3", "F4", "F5"}
    without_l8 = view_unit_commands(view, runs_dir=tmp_path / "runs", exclude_levers=("L8",))
    assert all(item.problem is None for item in without_l8)
    inner = next(item.command for item in s1 if item.unit_id == "ecc30__fold0__S1__F3_lr3e-4-1e-4_e30__inner2__seed43")
    argv = list(inner.argv)
    assert argv[argv.index("--fold-index") + 1] == "2" and argv[argv.index("--n-folds") + 1] == "4"
    assert argv[argv.index("--epochs") + 1] == "30" and argv[argv.index("--seed") + 1] == "43"
    assert argv[argv.index("--dataset-bundle-id") + 1] == "ecc30-fold0-view"


# ---------------------------------------------------------------------------
# Provenance


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid", *args], cwd=repo,
                   check=True, capture_output=True)


def test_real_actions_refuse_a_dirty_tree(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "tracked.txt").write_text("one\n")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-q", "-m", "fixture")
    state = provenance.require_clean_tree(repo)
    assert state["clean"] and len(state["head"]) == 40
    (repo / "tracked.txt").write_text("two\n")
    with pytest.raises(provenance.DirtyTreeError, match="uncommitted or untracked"):
        provenance.require_clean_tree(repo)
    _git(repo, "checkout", "-q", "--", "tracked.txt")
    (repo / "untracked.txt").write_text("new\n")
    with pytest.raises(provenance.DirtyTreeError):
        provenance.require_clean_tree(repo)
    assert provenance.real_action_git_state(smoke=True, repo=repo)["clean"] is False
    record = provenance.provenance_record(smoke=True, repo=repo, argv=["plan"])
    assert record["git"]["status_entries"] == 1 and record["argv"] == ["plan"] and record["smoke"] is True
    assert set(record["environment"]["packages"]) >= {"torch", "numpy"}
    with pytest.raises(provenance.DirtyTreeError, match="not a git checkout"):
        provenance.require_clean_tree(tmp_path)


@pytest.mark.parametrize("name, expected", [
    ("test", True), ("tests", True), ("metadata", True), ("pinmymetal_files", True),
    ("CLEAN_30_train_test_split_0_test.csv", True), ("classmodel_test_set", True), ("decoy_test.pdb", True),
    ("CLEAN_30_train_test_split_0", False), ("CLEAN_30_train_test_split_0_train.csv", False), ("train", False),
    ("P0000__chain_A__EC_1.1.1.1.pdb", False), ("latest_runs", False), ("contest", False),
])
def test_test_named_path_components(name, expected):
    assert provenance.names_test_split(name) is expected


def test_test_flags_are_refused():
    for flag in ("--run-test-eval", "--test-summary-csv=/x", "--final-test-checkpoint-paths",
                 "--allow-final-refit-test-eval", "--enable-final-test-calibration", "--disable-final-test-bootstrap-ci",
                 "--no-final-test-temperature-scaling", "--allow-train-loss-test-eval-debug"):
        with pytest.raises(provenance.HeldOutInputError):
            provenance.refuse_test_arguments(["--task", "ec", flag])
    provenance.refuse_test_arguments(["--task", "ec", "--val-fraction", "0", "--latest-epochs", "contest_test"])


# ---------------------------------------------------------------------------
# Smoke units end to end


@pytest.fixture(scope="module")
def smoke_runs(smoke_view):
    runs = smoke_view.default_runs_dir
    receipt = dry_run(smoke_view, cap_prefix=(), provenance=PROVENANCE, timeout=600)
    assert receipt["status"] == "passed", receipt["probes"]
    receipts = {}
    for family, refit in (("F4", False), ("F4", True), ("F2", False)):
        command = _command(smoke_view, family, refit=refit)
        receipts[command.unit_id] = run_unit(command, view=smoke_view, cap_prefix=(), provenance=PROVENANCE,
                                             timeout=900)
    return runs, receipts


G1_CHECKS = {"exit_code", "terminal_checkpoint", "selected_epoch", "history_length", "no_test_report",
             "optimizer_groups", "no_generated_features", "ec_label_report", "site_extraction_summary",
             "ec_predictions", "prediction_checkpoint", "prediction_files", "prediction_structures", "peak_memory",
             "campaign_identity", "data_paths_in_view", "overlay_identity", "smoke_matches_view", "capped"}


def test_smoke_units_run_and_verify(smoke_runs, smoke_view):
    from training.ec_prediction_export import PREDICTION_FILE_NAMES

    from ec_clean30 import commands

    assert {mode: tuple(names) for mode, names in PREDICTION_FILE_NAMES.items()} == dict(commands.PREDICTION_FILE_NAMES)
    runs, receipts = smoke_runs
    assert all(receipt["status"] == "completed" and receipt["returncode"] == 0 for receipt in receipts.values())
    assert all(receipt["view_clearance"]["view_manifest_sha256"] == smoke_view.manifest_sha256
               for receipt in receipts.values())
    assert all(receipt["read_guard"]["expanded_roots"] == smoke_view.forbidden_read_roots
               for receipt in receipts.values())
    report = verify_units(runs, provenance=PROVENANCE)
    assert report["status"] == "passed", [(unit["unit_id"], {k: v for k, v in unit["checks"].items() if not v["ok"]})
                                          for unit in report["units"]]
    by_unit = {(unit["family"], unit["kind"]): unit for unit in report["units"]}
    assert set(by_unit) == {("F4", KIND_INNER), ("F4", KIND_REFIT), ("F2", KIND_INNER)}
    assert G1_CHECKS | {"split_type", "global_attach"} <= set(by_unit["F4", KIND_INNER]["checks"])
    assert G1_CHECKS | {"refit_coverage", "global_attach"} <= set(by_unit["F4", KIND_REFIT]["checks"])
    assert G1_CHECKS | {"split_type", "no_global_branch"} <= set(by_unit["F2", KIND_INNER]["checks"])
    assert "global_attach" not in by_unit["F2", KIND_INNER]["checks"]
    assert by_unit["F4", KIND_INNER]["checks"]["global_attach"]["detail"] == {"attached": 8, "loaded": 8}
    assert by_unit["F4", KIND_REFIT]["checks"]["refit_coverage"]["detail"] == {"trained": 8, "eligible": 8}
    refit_files = by_unit["F4", KIND_REFIT]["checks"]["prediction_files"]["detail"]["files"]
    assert {item["path"] for item in refit_files.values()} == set(PREDICTION_FILE_NAMES["probe"])
    assert (runs / "verify_unit_receipt.json").is_file()
    with pytest.raises(CommandError, match="already exists"):
        run_unit(_command(smoke_view, "F4"), view=smoke_view, cap_prefix=(), provenance=PROVENANCE)


def test_run_unit_needs_a_verified_view_with_a_passed_dry_run(source, tmp_path):
    view = build_smoke_view(source, _smoke_dir(tmp_path), chain_means=False, provenance=PROVENANCE)
    # The runs folder must be the view's own (the static command check alone does not test the location).
    for runs in (tmp_path / "elsewhere", view.campaign_root / views.SMOKE_RUNS_DIR / "another_view"):
        loose = _command(view, "F2", runs_dir=runs)
        assert check_command(loose, view) == []
        with pytest.raises(CommandError, match="must be <campaign root>/smoke_runs/view or a folder inside it"):
            run_unit(loose, view=view, cap_prefix=(), provenance=PROVENANCE)
        assert not runs.exists()
    command = _command(view, "F2")
    with pytest.raises(CommandError, match="run dry-run first"):
        run_unit(command, view=view, cap_prefix=(), provenance=PROVENANCE)
    receipt = {"status": "failed", "view_manifest_sha256": view.manifest_sha256, "verified_view": True}
    (view.receipts / "dry_run_receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(CommandError, match="did not pass"):
        run_unit(command, view=view, cap_prefix=(), provenance=PROVENANCE)
    receipt.update({"status": "passed", "view_manifest_sha256": "0" * 64})
    (view.receipts / "dry_run_receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(CommandError, match="another view manifest"):
        run_unit(command, view=view, cap_prefix=(), provenance=PROVENANCE)
    receipt["view_manifest_sha256"] = view.manifest_sha256
    (view.receipts / "dry_run_receipt.json").write_text(json.dumps(receipt))
    make_writable(view.root)
    (view.esm / "extra_chain_A_esmc.pt").write_bytes(b"x")
    with pytest.raises(CommandError, match="does not verify"):
        run_unit(command, view=view, cap_prefix=(), provenance=PROVENANCE)
    assert not view.default_runs_dir.exists()
    make_writable(view.root)


def _tamper_run(run_dir: Path, case: str) -> str:
    if case == "test_report":
        (run_dir / "test_report.json").write_text("{}")
        return "no_test_report"
    if case == "best_checkpoint":
        (run_dir / "terminal_model_checkpoint.pt").rename(run_dir / "best_model_checkpoint.pt")
        return "terminal_checkpoint"
    if case == "generated":
        summary = json.loads((run_dir / "dataset_summary.json").read_text())
        summary["runtime_preparation"]["generated_esm_files"] = 1
        (run_dir / "dataset_summary.json").write_text(json.dumps(summary))
        return "no_generated_features"
    if case == "short_history":
        config = json.loads((run_dir / "run_config.json").read_text())
        config["history"] = []
        (run_dir / "run_config.json").write_text(json.dumps(config))
        return "history_length"
    if case == "steps":
        groups = json.loads((run_dir / "optimizer_groups.json").read_text())
        groups["realised_total_steps"] = [value + 1 for value in groups["realised_total_steps"]]
        (run_dir / "optimizer_groups.json").write_text(json.dumps(groups))
        return "optimizer_groups"
    if case == "prediction_csv":
        with (run_dir / "ec_val_pocket_predictions.csv").open("a", encoding="utf-8") as handle:
            handle.write("extra\n")
        return "prediction_files"
    if case == "prediction_structures":
        receipt = json.loads((run_dir / "ec_prediction_receipt.json").read_text())
        receipt["structure_ids_sha256"] = "0" * 64
        (run_dir / "ec_prediction_receipt.json").write_text(json.dumps(receipt))
        return "prediction_structures"
    if case == "stray_probe_file":
        (run_dir / "ec_probe_pocket_predictions.csv").write_text("x\n")
        return "prediction_files"
    raise AssertionError(case)


def _copy_unit(runs: Path, unit_id: str, target: Path) -> None:
    target.mkdir(exist_ok=True)
    shutil.copytree(runs / unit_id, target / unit_id)
    shutil.copyfile(runs / f"{unit_id}.runner.json", target / f"{unit_id}.runner.json")


@pytest.mark.parametrize("case", ["test_report", "best_checkpoint", "generated", "short_history", "steps",
                                  "prediction_csv", "prediction_structures", "stray_probe_file"])
def test_verify_unit_rejects_broken_receipts(smoke_runs, tmp_path, case):
    runs, receipts = smoke_runs
    unit_id = next(unit for unit in receipts if "__inner0__" in unit and "__F4_" in unit)
    copy = tmp_path / "runs_copy"
    _copy_unit(runs, unit_id, copy)
    failed = _tamper_run(copy / unit_id, case)
    result = verify_unit(copy, unit_id)
    assert not result["passed"] and not result["checks"][failed]["ok"]


def _refit_tamper(run_dir: Path, case: str) -> str:
    receipt = json.loads((run_dir / "ec_prediction_receipt.json").read_text())
    if case == "reconciliation":
        receipt["reconciliation_status"] = "match"
        check = "ec_predictions"
    elif case == "global_block":
        receipt["global_esm"]["manifest_sha256"] = "0" * 64
        check = "global_attach"
    else:
        raise AssertionError(case)
    (run_dir / "ec_prediction_receipt.json").write_text(json.dumps(receipt))
    return check


@pytest.mark.parametrize("case", ["reconciliation", "global_block"])
def test_verify_unit_rejects_broken_refit_receipts(smoke_runs, tmp_path, case):
    runs, receipts = smoke_runs
    unit_id = next(unit for unit in receipts if "__refit__" in unit)
    copy = tmp_path / "runs_copy"
    _copy_unit(runs, unit_id, copy)
    failed = _refit_tamper(copy / unit_id, case)
    result = verify_unit(copy, unit_id)
    assert not result["passed"] and not result["checks"][failed]["ok"]


def test_verify_unit_rejects_a_global_block_in_a_family_without_the_branch(smoke_runs, tmp_path):
    runs, receipts = smoke_runs
    unit_id = next(unit for unit in receipts if "__F2_" in unit)
    copy = tmp_path / "runs_copy"
    _copy_unit(runs, unit_id, copy)
    path = copy / unit_id / "ec_prediction_receipt.json"
    receipt = json.loads(path.read_text())
    receipt["global_esm"] = {"manifest_sha256": "0" * 64, "layer": "final", "standardization": "none"}
    path.write_text(json.dumps(receipt))
    result = verify_unit(copy, unit_id)
    assert not result["passed"] and not result["checks"]["no_global_branch"]["ok"]


def test_freeze2_manifest_is_accepted_by_the_test_pass_checks(smoke_runs, tmp_path):
    import training.ec_prediction_export as export

    runs, receipts = smoke_runs
    verify_units(runs, provenance=PROVENANCE)
    refit_unit = next(unit for unit in receipts if "__refit__" in unit)
    refit_sha = provenance.sha256_file(runs / refit_unit / "terminal_model_checkpoint.pt")
    pass_root = tmp_path / "pass_root"

    # A smoke manifest (engineering checkpoints) is written, but the test pass refuses it.
    out = tmp_path / "freeze2_smoke_manifest.json"
    assert _script().main(["manifest", "--kind", "freeze2", "--smoke", "--runs", str(runs),
                           "--test-pass-output-root", str(pass_root), "--out", str(out)]) == 0
    smoke_manifest = json.loads(out.read_text())
    assert smoke_manifest["schema_version"] == provenance.FREEZE2_SMOKE_SCHEMA_VERSION and smoke_manifest["smoke"]
    assert "test_pass_output_root" not in smoke_manifest and smoke_manifest["checkpoint_sha256s"] == [refit_sha]
    with pytest.raises(export.EcPredictionExportError, match="schema_version 1"):
        export._read_freeze2_manifest(out, provenance.sha256_file(out))
    with pytest.raises(FileExistsError):
        provenance.write_new_json(out, smoke_manifest)

    # A real manifest from (synthetic) non-smoke refit records is accepted by the test pass's checks.
    records = [{**item, "smoke": False} for item in smoke_manifest["checkpoints"]]
    manifest = provenance.build_freeze2_manifest(checkpoints=records, test_pass_output_root=pass_root, smoke=False,
                                                 provenance=PROVENANCE)
    real = tmp_path / "freeze2_manifest.json"
    digest = provenance.write_new_json(real, manifest)
    loaded = export._read_freeze2_manifest(real, digest)
    pinned = export._verify_freeze2_code(loaded["code_sha256"])
    assert set(export.freeze2_required_code_files()) <= set(pinned)
    assert "src/ec_clean30/commands.py" in pinned and "scripts/run_ec_clean30_campaign.py" in \
        loaded["additional_code_sha256"]
    assert loaded["schema_version"] == export.FREEZE2_SCHEMA_VERSION and loaded["checkpoint_sha256s"] == [refit_sha]
    assert loaded["test_pass_output_root"] == str(pass_root) and loaded["smoke"] is False
    with pytest.raises(provenance.FreezeManifestError, match="absolute"):
        provenance.build_freeze2_manifest(checkpoints=records, test_pass_output_root="relative/root", smoke=False,
                                          provenance=PROVENANCE)
    with pytest.raises(provenance.FreezeManifestError, match="needs an absolute test_pass_output_root"):
        provenance.build_freeze2_manifest(checkpoints=records, test_pass_output_root=None, smoke=False,
                                          provenance=PROVENANCE)
    with pytest.raises(provenance.FreezeManifestError, match="not recorded as a real"):
        provenance.build_freeze2_manifest(checkpoints=smoke_manifest["checkpoints"], test_pass_output_root=pass_root,
                                          smoke=False, provenance=PROVENANCE)
    # The real CLI route needs complete refit coverage; one smoke refit is refused.
    with pytest.raises(CommandError, match="real manifest"):
        collect_refit_checkpoints([runs], smoke=False)


def test_freeze2_checkpoints_are_verified_again(smoke_runs, tmp_path):
    runs, receipts = smoke_runs
    refit_unit = next(unit for unit in receipts if "__refit__" in unit)
    copy = tmp_path / "runs_copy"
    _copy_unit(runs, refit_unit, copy)
    stored = verify_units(copy, provenance=PROVENANCE)
    assert stored["status"] == "passed"
    assert [item["sha256"] for item in collect_refit_checkpoints([copy], smoke=True)] == \
        [stored["units"][0]["terminal_checkpoint_sha256"]]
    _tamper_run(copy / refit_unit, "generated")
    with pytest.raises(CommandError, match="does not pass verify-unit now"):
        collect_refit_checkpoints([copy], smoke=True)


def test_refit_coverage_needs_every_fold_and_seed():
    lr = {"learning_rate": 1e-4}

    def record(fold, seed, family="F2", step="S4"):
        return {"fold": fold, "seed": seed, "step": step, "recipe": {"family": family, "learning_rate": lr}}

    complete = [record(fold, seed) for fold in range(5) for seed in (42, 43, 44, 45, 46)]
    assert refit_coverage_problems(complete) == []
    assert any("missing [(4, 46)]" in problem for problem in refit_coverage_problems(complete[:-1]))
    assert any("more than once" in problem for problem in refit_coverage_problems(complete + [record(0, 42)]))
    mixed = complete[:-1] + [record(4, 46, family="F4")]
    assert any("fold 4 refits use 2 different recipes" in problem for problem in refit_coverage_problems(mixed))
    assert any("not S4" in problem for problem in refit_coverage_problems(complete[:-1] + [record(4, 46, step="smoke")]))


def test_verify_unit_can_delete_verified_smoke_checkpoints(smoke_runs, tmp_path):
    runs, receipts = smoke_runs
    copy = tmp_path / "runs_copy"
    for unit_id in receipts:
        _copy_unit(runs, unit_id, copy)
    script = _script()
    with pytest.raises(SystemExit, match="needs --smoke"):
        script.main(["verify-unit", "--runs", str(copy), "--delete-smoke-checkpoints"])
    assert script.main(["verify-unit", "--smoke", "--runs", str(copy), "--delete-smoke-checkpoints"]) == 0
    deletion = json.loads((copy / "smoke_checkpoint_deletion_receipt.json").read_text())
    assert deletion["verify_unit_receipt_sha256"] == provenance.sha256_file(copy / "verify_unit_receipt.json")
    assert sorted({item["unit_id"] for item in deletion["deleted_checkpoints"]}) == sorted(receipts)
    assert not any(path.name.endswith("_checkpoint.pt") for unit_id in receipts for path in (copy / unit_id).iterdir())
    assert any(item["path"].endswith("/last_model_checkpoint.pt") for item in deletion["deleted_checkpoints"])
    assert all((runs / unit_id / "terminal_model_checkpoint.pt").is_file() for unit_id in receipts)


def test_freeze1_manifest_records_plans_constants_and_views(smoke_runs, smoke_view, tmp_path):
    out = tmp_path / "freeze1_inputs.json"
    assert _script().main(["manifest", "--kind", "freeze1", "--smoke", "--view", str(smoke_view.root),
                           "--out", str(out)]) == 0
    manifest = json.loads(out.read_text())
    assert sorted(manifest["plans"]) == ["0", "1", "2", "3", "4"]
    assert all(entry["counts"]["total"] == 181 for entry in manifest["plans"].values())
    assert manifest["plans"]["0"]["plan_sha256"] == build_fold_plan(0)["plan_sha256"]
    [record] = manifest["views"]
    dry = smoke_view.receipts / "dry_run_receipt.json"
    assert record == {
        "fold": 0, "kind": "smoke_view", "bundle_id": "g1-smoke-view",
        "view_manifest_sha256": smoke_view.manifest_sha256, "allowlist_sha256": smoke_view.allowlist_sha256,
        "eligible_ids_sha256": smoke_view.manifest["eligible"]["ids_sha256"], "probe_sha256": smoke_view.probe_sha256,
        "membership": {"role": "smoke", "file": "smoke_membership.csv", "sha256": smoke_view.membership["sha256"],
                       "identity_sha256": smoke_view.membership["identity_sha256"], "n_folds": 2,
                       "certificate_sha256": None},
        "chain_means": {"final": smoke_view.chain_means_manifest_sha256("final")},
        "file_modes_enforced": True, "overlay": None,
        "dry_run_receipt": {"status": "passed", "sha256": provenance.sha256_file(dry),
                            "view_manifest_sha256": smoke_view.manifest_sha256},
    }
    assert "L3" in manifest["provisional_lever_settings"]
    assert manifest["constants"]["s2_execution_order"][-1] == "L3"
    assert manifest["selection_constants"]["MIN_GAIN_POINTS"] == 1.0
    assert "src/ec_clean30/campaign.py" in manifest["code_sha256"]


def test_freeze1_view_records_need_a_verified_view_and_a_real_dry_run(source, tmp_path):
    view = build_smoke_view(source, _smoke_dir(tmp_path), chain_means=False, provenance=PROVENANCE)
    with pytest.raises(ViewError, match="chain means for layer final are required"):
        freeze1_view_record(view.root, require_dry_run=False, provenance=PROVENANCE)
    build_chain_means(view.root)
    assert freeze1_view_record(view.root, require_dry_run=False, provenance=PROVENANCE)["dry_run_receipt"] is None
    with pytest.raises(ViewError, match="run dry-run first"):
        freeze1_view_record(view.root, require_dry_run=True, provenance=PROVENANCE)
    receipt = {"status": "passed", "view_manifest_sha256": view.manifest_sha256, "verified_view": True,
               "verification": {"chain_means": {"final": {"manifest_sha256": "0" * 64}}}}
    (view.receipts / "dry_run_receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ViewError, match="chain means differ from the ones the dry run verified"):
        freeze1_view_record(view.root, require_dry_run=True, provenance=PROVENANCE)
    make_writable(view.root)


# ---------------------------------------------------------------------------
# Command-line interface


def test_cli_smoke_view_plan_and_verify(source, tmp_path, capsys):
    script = _script()
    out = _smoke_dir(tmp_path, "cli_view")
    capsys.readouterr()
    assert script.main(["smoke-view", "--fold", "0", "--max-proteins", "6", "--seed", "3", "--out", str(out),
                        "--data-root", str(source["data_root"])]) == 0
    view = View.load(out)
    assert view.bundle_id == "g1-smoke-view" and len(view.structure_ids()) == 6
    assert script.main(["verify-view", "--view", str(out), "--smoke"]) == 0
    # Neither action prints eligible or membership row counts (per-fold class numbers stay off the screen).
    text = capsys.readouterr().out
    built, end = json.JSONDecoder().raw_decode(text)
    verified, _end = json.JSONDecoder().raw_decode(text, end + 1)
    assert "n_eligible" not in text and "n_rows" not in text
    for printed in (built, verified):
        assert printed["n_structures"] == 6 and printed["eligible_matches_membership"] is True
        assert "n_eligible" not in printed and "n_rows" not in printed["membership"]
        assert printed["membership"]["sha256"] == view.membership["sha256"]
    assert built["eligible_ids_sha256"] == view.manifest["eligible"]["ids_sha256"]
    assert verified["status"] == "verified" and verified["file_modes_enforced"] is True
    stored = json.loads((view.receipts / views.VERIFY_RECEIPT_NAME).read_text())
    assert stored["n_eligible"] == 6 and stored["membership"]["n_rows"] == 6  # kept in the private receipt
    with pytest.raises(SystemExit, match="pass --smoke exactly"):
        script.main(["verify-view", "--view", str(out)])
    assert script.main(["build-view", "--fold", "0", "--out", str(tmp_path / "x_test"),
                        "--data-root", str(source["data_root"])]) == 2
    capsys.readouterr()
    assert script.main(["plan", "--fold", "1"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["counts"]["total"] == 181 and printed["plan_sha256"] == build_fold_plan(1)["plan_sha256"]
    plan_file = tmp_path / "plan_fold1.json"
    assert script.main(["plan", "--fold", "1", "--out", str(plan_file), "--smoke"]) == 0
    assert json.loads(plan_file.read_text())["plan_sha256"] == build_fold_plan(1)["plan_sha256"]
    assert json.loads((tmp_path / "plan_fold1.json.receipt.json").read_text())["provenance"]["smoke"] is True
    assert script.main(["plan", "--fold", "0", "--families", "F9"]) == 2
    make_writable(out)


def test_cli_stubs_and_real_run_unit_are_not_implemented(smoke_view, capsys):
    script = _script()
    for action in ("select", "gap-gate", "refit", "execute"):
        assert script.main([action, "--fold", "0"]) == script.EXIT_NOT_IMPLEMENTED
        assert "not implemented yet" in capsys.readouterr().out
    assert script.main(["run-unit", "--view", str(smoke_view.root), "--family", "F4", "--runs",
                        str(smoke_view.root.parent / "runs_real")]) == script.EXIT_NOT_IMPLEMENTED


def test_cli_help_and_torch_free_imports():
    env = {**os.environ, "MKL_THREADING_LAYER": "GNU"}
    completed = subprocess.run([PYTHON, str(SCRIPT), "--help"], capture_output=True, text=True, env=env, timeout=120)
    assert completed.returncode == 0 and "dry-run" in completed.stdout
    code = ("import sys; sys.path.insert(0, 'src'); import ec_clean30.campaign, ec_clean30.provenance, "
            "ec_clean30.views, ec_clean30.commands; print('torch' in sys.modules)")
    completed = subprocess.run([PYTHON, "-c", code], cwd=ROOT, capture_output=True, text=True, env=env, timeout=120)
    assert completed.returncode == 0 and completed.stdout.strip() == "False"


def _clean_fit_dir(root: Path, fold: int, regime: str = "r1", seed: int = 42, epochs: int = 2500,
                   embedding: str = "esm1b_t33_650M_UR50S_mean_layer33") -> Path:
    model = f"split30_fold{fold}_{regime}_{embedding}_seed{seed}_d1280"
    work = root / model
    (work / "data" / "model").mkdir(parents=True)
    weights = work / "data" / "model" / f"{model}.pth"
    weights.write_bytes(f"weights {model}".encode())
    prepare = {"fold": fold, "regime": regime, "seed": seed, "embedding": embedding, "input_dim": 1280}
    (work / "prepare_manifest.json").write_text(json.dumps(prepare))
    fit = {"prepare_manifest_sha256": provenance.sha256_file(work / "prepare_manifest.json"),
           "recipe": {"epochs": epochs, "adaptive_rate": 100},
           "training_receipt": {"model_name": model, "seed": seed, "input_dim": 1280,
                                "final_weights_sha256": provenance.sha256_file(weights)}}
    (work / "fit_manifest.json").write_text(json.dumps(fit))
    return work


def test_clean_fits_are_re_hashed_for_freeze2(tmp_path):
    from ec_clean30.commands import collect_clean_checkpoints

    fits = [_clean_fit_dir(tmp_path, fold) for fold in range(campaign.N_OUTER_FOLDS)]
    records = collect_clean_checkpoints(fits, smoke=False)
    assert [record["fold"] for record in records] == [0, 1, 2, 3, 4]
    assert all(record["system"] == "clean" and record["smoke"] is False for record in records)
    manifest = provenance.build_freeze2_manifest(checkpoints=records, test_pass_output_root=tmp_path / "pass",
                                                 smoke=False, provenance=PROVENANCE)
    assert set(manifest["checkpoint_sha256s"]) == {record["sha256"] for record in records}
    with pytest.raises(CommandError, match="outer folds \\[4\\]"):
        collect_clean_checkpoints(fits[:4], smoke=False)
    with pytest.raises(CommandError, match="given twice"):
        collect_clean_checkpoints([fits[0], fits[0]], smoke=True)
    short = _clean_fit_dir(tmp_path / "short", 0, epochs=200)
    with pytest.raises(CommandError, match="published epochs"):
        collect_clean_checkpoints([short, *fits[1:]], smoke=False)
    assert collect_clean_checkpoints([short], smoke=True)[0]["recipe"]["epochs"] == 200
    inner = _clean_fit_dir(tmp_path / "inner", 0, regime="r1_inner")
    with pytest.raises(CommandError, match="inner-fold"):
        collect_clean_checkpoints([inner], smoke=True)
    weights = next((fits[1] / "data" / "model").glob("*.pth"))
    weights.write_bytes(b"changed")
    with pytest.raises(CommandError, match="do not match the training receipt"):
        collect_clean_checkpoints(fits, smoke=False)


def test_freeze2_pins_every_scorer_and_baseline_file(tmp_path):
    from ec_clean30 import scoring

    record = {"unit_id": "u", "sha256": "a" * 64, "smoke": False}
    manifest = provenance.build_freeze2_manifest(checkpoints=[record], test_pass_output_root=tmp_path / "pass",
                                                 smoke=False, provenance=PROVENANCE)
    pinned = set(manifest["code_sha256"]) | set(manifest["additional_code_sha256"])
    assert set(scoring.SCORER_CODE_PATHS) <= pinned, sorted(set(scoring.SCORER_CODE_PATHS) - pinned)
    assert {"src/ec_clean30/baseline.py", "scripts/run_ec_clean30_baseline.py"} <= pinned


# ---------------------------------------------------------------------------
# Phase 3 hooks, overlay identity and file modes from the command line


def test_unit_identity_carries_the_freeze1b_and_overlay_keys(smoke_view):
    command = _command(smoke_view, "F2")
    assert {"freeze1_sha256", "freeze1b_sha256", "overlay_id", "overlay_sha256"} <= set(command.identity)
    assert command.identity["freeze1b_sha256"] is None and command.identity["overlay_id"] is None
    recipe = default_smoke_recipe("F2", lr_index=1)
    pinned = build_unit_command(smoke_view, recipe, unit_id=smoke_unit_id(0, recipe, refit=False, seed=42),
                                kind=KIND_INNER, seed=42, runs_dir=smoke_view.root.parent / "runs", step="smoke",
                                inner_fold=0, freeze1_sha256="a" * 64, freeze1b_sha256="b" * 64)
    assert (pinned.identity["freeze1_sha256"], pinned.identity["freeze1b_sha256"]) == ("a" * 64, "b" * 64)
    assert json.loads(pinned.argv[list(pinned.argv).index("--campaign-run-identity") + 1])["freeze1b_sha256"] == \
        "b" * 64
    assert check_command(pinned, smoke_view) == []
    # A view without an overlay record takes no overlay values.
    with pytest.raises(CommandError, match="overlay record"):
        build_unit_command(smoke_view, recipe, unit_id="u", kind=KIND_INNER, seed=42, runs_dir=smoke_view.root.parent,
                           step="smoke", inner_fold=0, overlay_id="ecc30-overlay-v1")


def test_verify_unit_honours_a_raised_memory_limit_and_the_overlay_rule(smoke_runs, source, tmp_path):
    runs, receipts = smoke_runs
    unit_id = next(unit for unit in receipts if "__inner0__" in unit and "__F4_" in unit)
    copy = tmp_path / "runs_copy"
    _copy_unit(runs, unit_id, copy)
    receipt_path = copy / f"{unit_id}.runner.json"
    runner = json.loads(receipt_path.read_text())
    runner["children_peak_rss_bytes"] = 4 * 1024 ** 3
    receipt_path.write_text(json.dumps(runner))
    default = verify_unit(copy, unit_id)
    assert not default["passed"] and not default["checks"]["peak_memory"]["ok"]
    assert default["checks"]["peak_memory"]["detail"]["limit_bytes"] == 3 * 1024 ** 3
    raised = verify_unit(copy, unit_id, memory_limit_bytes=8 * 1024 ** 3)
    assert raised["passed"], {name: item for name, item in raised["checks"].items() if not item["ok"]}
    assert verify_units(copy, memory_limit_bytes=8 * 1024 ** 3, write=False,
                        provenance=PROVENANCE)["memory_limit_bytes"] == 8 * 1024 ** 3
    assert raised["checks"]["overlay_identity"]["ok"] and raised["checks"]["capped"]["ok"]
    # The view, not the editable receipt, decides whether a unit is real: checked against a fold view (no
    # overlay), the unit must name the view's overlay and must have run capped, whatever its identity says.
    fold_root = _fold_dir(tmp_path / "fold_view")
    build_view(fold=0, layout=SourceLayout.from_data_root(source["data_root"]), out_dir=fold_root,
               provenance=PROVENANCE)
    real = verify_unit(copy, unit_id, view=View.load(fold_root), memory_limit_bytes=8 * 1024 ** 3)
    assert not real["checks"]["smoke_matches_view"]["ok"]
    assert not real["checks"]["overlay_identity"]["ok"] and not real["checks"]["capped"]["ok"]
    make_writable(fold_root)
    runner["unit"]["identity"]["smoke"] = False
    receipt_path.write_text(json.dumps(runner))
    flipped = verify_unit(copy, unit_id, memory_limit_bytes=8 * 1024 ** 3)
    assert not flipped["passed"] and not flipped["checks"]["smoke_matches_view"]["ok"]
    assert flipped["checks"]["overlay_identity"]["ok"] and flipped["checks"]["capped"]["ok"]


def test_fold_views_need_a_cap_prefix_for_probes_and_units(source, tmp_path):
    from ec_clean30.commands import CAP_PREFIX

    root = _fold_dir(tmp_path)
    build_view(fold=0, layout=SourceLayout.from_data_root(source["data_root"]), out_dir=root, provenance=PROVENANCE)
    build_chain_means(root)
    view = View.load(root)
    with pytest.raises(CommandError, match="only under a cap prefix"):
        dry_run(view, probe_families=(), cap_prefix=(), provenance=PROVENANCE, write=False)
    recipe = default_smoke_recipe("F2")
    command = build_unit_command(view, recipe, unit_id="ecc30__fold0__cap_case__refit__seed42", kind=KIND_REFIT,
                                 seed=42, runs_dir=view.default_runs_dir, step="S4")
    with pytest.raises(CommandError, match="only under a cap prefix"):
        run_unit(command, view=view, cap_prefix=(), provenance=PROVENANCE)
    assert not view.default_runs_dir.exists()
    # A fold view's dry-run receipt without a cap prefix does not clear the view.
    final = view.chain_means_manifest_sha256("final")
    receipt = {"status": "passed", "view_manifest_sha256": view.manifest_sha256, "verified_view": True,
               "verification": {"chain_means": {"final": {"manifest_sha256": final}}}, "cap_prefix": []}
    (view.receipts / "dry_run_receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ViewError, match="ran its probes without a cap prefix"):
        freeze1_view_record(view.root, require_dry_run=True, provenance=PROVENANCE)
    receipt["cap_prefix"] = list(CAP_PREFIX)
    (view.receipts / "dry_run_receipt.json").write_text(json.dumps(receipt))
    assert freeze1_view_record(view.root, require_dry_run=True, provenance=PROVENANCE)["dry_run_receipt"][
        "status"] == "passed"
    make_writable(root)


def test_cli_dry_run_passes_the_cap_prefix(smoke_view, monkeypatch, capsys):
    from ec_clean30 import commands

    seen = {}

    def fake_dry_run(view, **kwargs):
        seen.clear()
        seen.update(kwargs, view=view.root)
        return {"status": "passed", "n_commands_checked": 0, "static_failures": [], "probes": []}

    monkeypatch.setattr(commands, "dry_run", fake_dry_run)
    script = _script()
    base = ["dry-run", "--smoke", "--view", str(smoke_view.root)]
    assert script.main([*base, "--cap-prefix", "systemd-run --scope -q -p 'MemoryMax=20G' -p CPUQuota=800%"]) == 0
    assert seen["cap_prefix"] == ("systemd-run", "--scope", "-q", "-p", "MemoryMax=20G", "-p", "CPUQuota=800%")
    assert script.main(base) == 0 and seen["cap_prefix"] == commands.CAP_PREFIX
    with pytest.raises(SystemExit, match="needs a command prefix"):
        script.main([*base, "--cap-prefix", "  "])
    capsys.readouterr()


def test_cli_apply_view_modes(source, tmp_path, capsys):
    script = _script()
    out = _smoke_dir(tmp_path)
    build_smoke_view(source, out, chain_means=False, provenance=PROVENANCE)
    manifest_sha = provenance.sha256_file(out / views.VIEW_MANIFEST_NAME)
    capsys.readouterr()
    assert script.main(["apply-view-modes", "--smoke", "--view", str(out)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["outcome"] == "enforced" and printed["view_manifest_sha256"] == manifest_sha
    stored = out / "receipts" / views.FILE_MODES_APPLIED_RECEIPT_NAME
    assert provenance.sha256_file(stored) == printed["receipt_sha256"]
    assert json.loads(stored.read_text())["provenance"]["smoke"] is True
    with pytest.raises(SystemExit, match="pass --smoke exactly"):
        script.main(["apply-view-modes", "--view", str(out)])
    assert script.main(["verify-view", "--smoke", "--view", str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["file_modes_enforced"] is True
    make_writable(out)


# ---------------------------------------------------------------------------
# Training-side input actions from the command line (cohort-audit, r2-lists, overlay, simulate)

CLEAN_GIT = {"head": "c" * 40, "branch": "main", "clean": True, "status_entries": 0, "status_sample": []}
DIRTY_ENTRY = "untracked_entry_marker.txt"
SWEEP = {
    "format": simulation.SWEEP_FORMAT, "sweep_id": "cli_sweep", "n_replicates": 2,
    "rules": {"current": {"n_draws": 50, "stratified": True}}, "noise_base": {"fit_shift_sd": 0.15},
    "noise_points": {"central": {}},
    "groups": [{"group": "null", "folds": [0, 1], "rules": ["current"], "looks": [1], "noise_points": ["central"]}],
}


def _git_states(monkeypatch, modules, *, repo=None):
    """Stand in for each module's clean-tree check: a clean checkout, or the real check on ``repo``."""
    calls = []

    def state(*, smoke):
        calls.append(smoke)
        return dict(CLEAN_GIT) if repo is None else provenance.real_action_git_state(smoke=smoke, repo=repo)

    for module in modules:
        monkeypatch.setattr(module, "real_action_git_state", state)
    return calls


def _run_cli(capsys, argv):
    """Run the runner's main on ``argv``; returns (exit code, stdout, stderr)."""
    capsys.readouterr()
    code = _script().main([str(item) for item in argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _named(text: str, world) -> list[str]:
    return sorted(accession for accession in world.stems if accession in text)


@pytest.fixture(scope="module")
def dirty_checkout(tmp_path_factory):
    """A git checkout with one untracked file, whose name a refusal must not show."""
    repo = tmp_path_factory.mktemp("dirty_checkout")
    _git(repo, "init", "-q")
    (repo / "tracked.txt").write_text("one\n")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-q", "-m", "fixture")
    (repo / DIRTY_ENTRY).write_text("new\n")
    return repo


@pytest.fixture(scope="module")
def phase2_world(tmp_path_factory):
    """Five-fold synthetic data root, a passed cohort audit, the training-side identity chain (fake pinned search)
    and R2 lists, in a campaign root named after the campaign (the overlay tests' builder)."""
    return build_world(tmp_path_factory.mktemp("phase2_world"))


@pytest.fixture
def pinned_search(phase2_world, monkeypatch):
    """The fake search binary counts as the pinned MMseqs2 build (the identity loaders require it)."""
    monkeypatch.setattr(identity, "MMSEQS_BINARY_SHA256", phase2_world.binary_sha)


@pytest.fixture(scope="module")
def cli_overlay(phase2_world, tmp_path_factory):
    """A real overlay built through the runner (the clean-tree check stood in), with what the build printed."""
    world = phase2_world
    campaign_dir = tmp_path_factory.mktemp("phase2_overlay") / campaign.CAMPAIGN_ID
    root = campaign_dir / overlay.OVERLAYS_DIR / overlay.DEFAULT_OVERLAY_ID
    printed = io.StringIO()
    with pytest.MonkeyPatch.context() as patch, contextlib.redirect_stdout(printed):
        patch.setattr(identity, "MMSEQS_BINARY_SHA256", world.binary_sha)
        calls = _git_states(patch, [overlay])
        code = _script().main(["overlay", "build", "--out", str(root), "--cohort", str(world.cohort),
                               "--identity-root", str(world.identity_root), "--r2-lists", str(world.r2),
                               "--d2-option", "b"])
    return {"root": root, "code": code, "printed": printed.getvalue(), "calls": calls}


def test_cli_wires_the_input_actions_to_their_modules():
    script = _script()
    parser = script.build_parser()
    cases = [
        (["cohort-audit", "--out", "o", "--label-table", "l", "--label-table-sha256", "a", "--sequence-map", "m",
          "--sequence-map-sha256", "b"], cohort.action_cohort_audit),
        (["r2-lists", "--identity-root", "i", "--cohort", "c", "--out", "o"], r2_lists.action_r2_lists),
        (["overlay", "build", "--out", "o", "--cohort", "c", "--identity-root", "i", "--r2-lists", "r",
          "--d2-option", "b"], overlay.action_overlay_build),
        (["overlay", "verify", "--overlay", "o"], overlay.action_overlay_verify),
        (["simulate", "--sweep", "s", "--out", "o", "--calibrate-from", "0=p", "--calibrate-from", "2=q"],
         script.action_simulate),
    ]
    for argv, handler in cases:
        assert parser.parse_args(argv).handler is handler, argv
    simulate = parser.parse_args(cases[-1][0])
    assert simulate.calibrate_from == ["0=p", "2=q"] and simulate.points is None and simulate.smoke is False
    assert simulate.max_forecast_cpu_hours == simulation.DEFAULT_MAX_FORECAST_CPU_HOURS
    audit = parser.parse_args(cases[0][0])
    assert audit.coordinate_tolerance == cohort.A3_COORDINATE_TOLERANCE and audit.expected_sequence_difference == []
    for argv in (["overlay"], ["overlay", "build", "--out", "o", "--cohort", "c", "--identity-root", "i",
                               "--r2-lists", "r", "--d2-option", "c"],
                 [*cases[0][0], "--expected-label-eligible", "1,2,3,4,5"]):
        with pytest.raises(SystemExit):
            parser.parse_args(argv)
    # Every action has an entry in the docstring's action list, which is the --help description.
    sub = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
    assert {"cohort-audit", "r2-lists", "overlay", "simulate"} <= set(sub.choices)
    missing = [name for name in sub.choices
               if not re.search(rf"^  (?:[a-z-]+, )*{re.escape(name)}(?:,| |$)", script.__doc__, re.M)]
    assert missing == [] and script.__doc__.strip() in parser.format_help()
    assert {"overlay build", "overlay verify"} <= {line.strip()[:14].strip() for line in script.__doc__.splitlines()}


def test_cli_parser_is_torch_free_and_protected_modules_load_no_input_module():
    """Building the parser imports the input modules (torch-free); the modules rule R-CMD protects load none."""
    code = "\n".join([
        "import importlib.util, sys",
        "sys.path.insert(0, 'src')",
        "import ec_clean30.campaign, ec_clean30.commands, ec_clean30.views, ec_clean30.provenance, ec_clean30.identity",
        "print(sorted(name for name in ('cohort', 'overlay', 'r2_lists', 'simulation')",
        "             if 'ec_clean30.' + name in sys.modules))",
        "spec = importlib.util.spec_from_file_location('runner', 'scripts/run_ec_clean30_campaign.py')",
        "module = importlib.util.module_from_spec(spec)",
        "spec.loader.exec_module(module)",
        "module.build_parser()",
        "print('torch' in sys.modules, 'ec_clean30.overlay' in sys.modules, 'ec_clean30.simulation' in sys.modules)",
    ])
    env = {**os.environ, "MKL_THREADING_LAYER": "GNU"}
    completed = subprocess.run([PYTHON, "-c", code], cwd=ROOT, capture_output=True, text=True, env=env, timeout=120)
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert completed.stdout.splitlines() == ["[]", "False True False"]


def test_cli_cohort_audit_exit_codes_and_refusals(phase2_world, dirty_checkout, tmp_path, monkeypatch, capsys):
    world = phase2_world
    labels = world.base / "source" / "labels"
    table, sequence_map = labels / "native_label_table.tsv", labels / "sequence_map.tsv"

    def argv(out, *extra, label_table=table):
        return ["cohort-audit", "--out", out, "--label-table", label_table,
                "--label-table-sha256", provenance.sha256_file(label_table), "--sequence-map", sequence_map,
                "--sequence-map-sha256", provenance.sha256_file(sequence_map), "--data-root", world.data_root, *extra]

    calls = _git_states(monkeypatch, [cohort])
    out = tmp_path / campaign.CAMPAIGN_ID / cohort.COHORT_FOLDER_NAME
    code, text, err = _run_cli(capsys, argv(out))
    assert code == 0, err
    printed = json.loads(text)
    assert calls == [False] and printed["status"] == "passed" and printed["failed_checks"] == []
    assert printed["n_proteins"] == len(world.stems) and printed["out"] == str(out)
    assert not _named(text + err, world) and "__EC_" not in text + err
    receipt = json.loads((out / cohort.RECEIPT_NAME).read_text())
    assert receipt["smoke"] is False and receipt["provenance"]["git"]["head"] == CLEAN_GIT["head"]
    # Same inputs and code as the world's own audit: the same tables.
    for name in (cohort.PROTEINS_NAME, cohort.SITES_NAME, cohort.FOLD_COUNTS_NAME.format(fold=0)):
        assert (out / name).read_bytes() == (world.cohort / name).read_bytes(), name
    # A written cohort folder is not written again (an input error, exit 2, not a traceback).
    code, _text, err = _run_cli(capsys, argv(out))
    assert code == 2 and err.startswith("error: ") and "not empty" in err
    # A check that fails stops the audit (exit 1); its outputs and receipt are still written.
    stopped = tmp_path / "stopped" / campaign.CAMPAIGN_ID / cohort.COHORT_FOLDER_NAME
    code, text, _err = _run_cli(capsys, argv(stopped, "--smoke", "--expected-sequence-difference", "P1000:A1G"))
    printed = json.loads(text)
    assert code == 1 and printed["status"] == "stopped" and calls[-1] is True
    assert printed["failed_checks"] == ["sequence_map_differences_as_expected"]
    assert json.loads((stopped / cohort.RECEIPT_NAME).read_text())["smoke"] is True
    # A test-named input is refused before the clean-tree check, and nothing is written.
    named = tmp_path / "copies" / "native_label_table_test.tsv"
    named.parent.mkdir()
    shutil.copyfile(table, named)
    before = list(calls)
    for case in (argv(tmp_path / "named" / campaign.CAMPAIGN_ID / cohort.COHORT_FOLDER_NAME, label_table=named),
                 argv(tmp_path / "named" / campaign.CAMPAIGN_ID / "cohort_test")):
        code, _text, err = _run_cli(capsys, case)
        assert code == 2 and err.startswith("refused: ") and "test" in err
    assert calls == before and not (tmp_path / "named").exists()
    # A real audit from a dirty checkout is refused without naming its status entries.
    _git_states(monkeypatch, [cohort], repo=dirty_checkout)
    code, _text, err = _run_cli(capsys, argv(tmp_path / "dirty" / campaign.CAMPAIGN_ID / cohort.COHORT_FOLDER_NAME))
    assert code == 2 and err.startswith("refused: ") and "has 1 uncommitted or untracked entries" in err
    assert DIRTY_ENTRY not in err and "git status lists the entries" in err and not (tmp_path / "dirty").exists()


def test_cli_r2_lists_exit_codes_and_refusals(phase2_world, dirty_checkout, pinned_search, tmp_path_factory,
                                              monkeypatch, capsys):
    world = phase2_world
    # The comparator refuses an allowlist path with any test-named part, and pytest's own tmp_path names the test.
    tmp_path = tmp_path_factory.mktemp("r2cli")
    base = ["r2-lists", "--identity-root", world.identity_root, "--cohort", world.cohort]
    calls = _git_states(monkeypatch, [r2_lists])
    out = tmp_path / "r2_lists"
    code, text, err = _run_cli(capsys, [*base, "--out", out, "--fold-work", f"1={world.work(1)}"])
    assert code == 0, err
    printed = json.loads(text)
    assert calls == [False] and printed["smoke"] is False and printed["out"] == str(out)
    assert printed["r2_allowlists"]["0"]["n_accessions"] == len(world.training(0))
    for fold in range(1, campaign.N_OUTER_FOLDS):
        assert printed["r2_allowlists"][str(fold)] == {"file": f"r2_allowlist_fold{fold}.txt", "size_in_receipt": True}
    assert not _named(text + err, world)
    assert r2_lists.verify_r2_lists(out)["status"] == "verified"
    for fold in range(campaign.N_OUTER_FOLDS):
        name = r2_lists.R2_ALLOWLIST_NAME.format(fold=fold)
        assert (out / name).read_bytes() == (world.r2 / name).read_bytes(), name
    # Input errors are one line on the console (exit 2), from the action itself or from main.
    code, _text, err = _run_cli(capsys, [*base, "--out", out])
    assert code == 2 and err.startswith("error: The output folder exists")
    assert calls == [False, False]
    code, _text, err = _run_cli(capsys, [*base, "--out", tmp_path / "other", "--fold-work", "9=/a"])
    assert code == 2 and err.startswith("error: ") and "outside 0..4" in err
    for argv in (["--out", tmp_path / "lists_test"],
                 ["--out", tmp_path / "other", "--fold-work", f"2={tmp_path / 'fold2_test'}"]):
        code, _text, err = _run_cli(capsys, [*base, *argv])
        assert code == 2 and err.startswith("refused: ")
    assert calls == [False, False]  # each refused before the clean-tree check
    assert not (tmp_path / "lists_test").exists() and not (tmp_path / "other").exists()
    _git_states(monkeypatch, [r2_lists], repo=dirty_checkout)
    code, _text, err = _run_cli(capsys, [*base, "--out", tmp_path / "dirty"])
    assert code == 2 and err.startswith("refused: ") and DIRTY_ENTRY not in err and not (tmp_path / "dirty").exists()
    code, text, _err = _run_cli(capsys, [*base, "--out", tmp_path / "smoked", "--smoke"])
    assert code == 0 and json.loads(text)["smoke"] is True


def test_cli_overlay_build_and_verify(phase2_world, cli_overlay, dirty_checkout, pinned_search, tmp_path,
                                      monkeypatch, capsys):
    world, root = phase2_world, cli_overlay["root"]
    assert cli_overlay["code"] == 0 and cli_overlay["calls"] == [False]
    printed = json.loads(cli_overlay["printed"])
    assert printed["status"] == "built" and printed["smoke"] is False and printed["overlay"] == str(root)
    assert printed["n_proteins"] == len(world.stems) and not _named(cli_overlay["printed"], world)
    manifest = json.loads((root / overlay.MANIFEST_NAME).read_text())
    assert manifest["provenance"]["git"]["head"] == CLEAN_GIT["head"] and manifest["smoke"] is False

    calls = _git_states(monkeypatch, [overlay])
    receipt = tmp_path / "overlay_verify.json"
    code, text, err = _run_cli(capsys, ["overlay", "verify", "--overlay", root, "--receipt", receipt])
    assert code == 0, err
    result = json.loads(text)
    assert calls == [False] and result["status"] == "verified"
    assert result["manifest_sha256"] == printed["manifest_sha256"] and result["tar_sha256"] == printed["tar_sha256"]
    assert provenance.sha256_file(receipt) == result["receipt_sha256"]
    # A changed copy fails verification: exit 1 with the problems printed.
    copy = tmp_path / "copy" / campaign.CAMPAIGN_ID / overlay.OVERLAYS_DIR
    shutil.copytree(root.parent, copy)
    make_writable(copy)
    (copy / overlay.DEFAULT_OVERLAY_ID / overlay.LABEL_FREE_INDEX_NAME).write_text("accession\n")
    code, text, _err = _run_cli(capsys, ["overlay", "verify", "--overlay", copy / overlay.DEFAULT_OVERLAY_ID])
    failed = json.loads(text)
    assert code == 1 and failed["status"] == "failed" and failed["problems"] and not _named(text, world)

    # Build refusals: one line on the console (exit 2) and nothing written.
    other = tmp_path / "refusals" / campaign.CAMPAIGN_ID / overlay.OVERLAYS_DIR
    build = ["overlay", "build", "--out", other / overlay.DEFAULT_OVERLAY_ID, "--cohort", world.cohort,
             "--identity-root", world.identity_root, "--r2-lists", world.r2]
    code, _text, err = _run_cli(capsys, [*build, "--d2-option", "d"])
    assert code == 2 and err.startswith("error: ") and "not built" in err and calls == [False] * 3
    for argv in ([*build, "--d2-option", "b", "--fold-work", f"2={tmp_path / 'fold2_test'}"],
                 ["overlay", "build", "--out", other / overlay.DEFAULT_OVERLAY_ID, "--cohort", tmp_path / "cohort_test",
                  "--identity-root", world.identity_root, "--r2-lists", world.r2, "--d2-option", "b"]):
        code, _text, err = _run_cli(capsys, argv)
        assert code == 2 and err.startswith("refused: ")
    assert calls == [False] * 3  # refused before the clean-tree check
    _git_states(monkeypatch, [overlay], repo=dirty_checkout)
    code, _text, err = _run_cli(capsys, [*build, "--d2-option", "b"])
    assert code == 2 and err.startswith("refused: ") and DIRTY_ENTRY not in err
    code, _text, err = _run_cli(capsys, ["overlay", "verify", "--overlay", root])
    assert code == 2 and err.startswith("refused: ") and DIRTY_ENTRY not in err  # a real overlay: clean tree only
    assert not other.exists() or not any(other.iterdir())


def test_cli_simulate_on_the_overlay_copies(phase2_world, cli_overlay, dirty_checkout, pinned_search, tmp_path,
                                            monkeypatch, capsys):
    monkeypatch.setattr(simulation, "SEED_SPREAD_POPULATIONS", 40)
    world, root = phase2_world, cli_overlay["root"]
    sweep = tmp_path / "sweep.json"
    sweep.write_text(json.dumps(SWEEP, indent=1))
    calibrate = [part for fold in (0, 1)
                 for part in ("--calibrate-from", f"{fold}={root / overlay.IDENTITY_DIR / f'fold{fold}'}")]
    calls = _git_states(monkeypatch, [simulation])
    out = tmp_path / "sweep_v1"
    code, text, err = _run_cli(capsys, ["simulate", "--sweep", sweep, "--out", out, *calibrate])
    assert code == 0, err
    payload = json.loads(text)
    assert calls == [False] and payload["status"] == simulation.STATUS_COMPLETE
    assert payload["smoke"] is False and payload["points_recorded"] == 2
    assert "class_proportions" not in text and not _named(text + err, world)
    # Eligible counts are printed for fold 0 only; the private sweep receipt keeps every fold's.
    stored = json.loads((out / simulation.RECEIPT_FILE).read_text())["calibrations"]
    assert payload["calibrations"]["0"] == stored["0"] and stored["1"]["n_proteins"] > 0
    assert payload["calibrations"]["1"] == {key: value for key, value in stored["1"].items() if key != "n_proteins"}
    assert len(list((out / simulation.RECORDS_FOLDER).glob("*.json"))) == 2
    # The forecast limit stops before the next point (exit 4).
    code, text, _err = _run_cli(capsys, ["simulate", "--sweep", sweep, "--out", tmp_path / "stopped", *calibrate,
                                         "--max-forecast-cpu-hours", "1e-9"])
    assert code == simulation.EXIT_STOPPED == 4 and json.loads(text)["status"] == simulation.STATUS_STOPPED
    # Input errors: one line on the console (exit 2), nothing written.
    for argv, message in (
            (["--out", tmp_path / "identity_read", "--calibrate-from", f"0={world.work(0)}",
              "--calibrate-from", f"1={world.work(1)}"], "campaign's identity folder"),
            (["--out", tmp_path / "one_fold", calibrate[0], calibrate[1]], "without a --calibrate-from"),
            (["--out", tmp_path / "unknown", *calibrate, "--points", "no_such_point"], "does not define")):
        code, _text, err = _run_cli(capsys, ["simulate", "--sweep", sweep, *argv])
        assert code == 2 and err.startswith("error: ") and message in err, err
        assert not Path(argv[1]).exists()
    # Test-named paths are refused before the clean-tree check (a K=PATH value by its path part too).
    named_sweep = tmp_path / "sweep_test.json"
    shutil.copyfile(sweep, named_sweep)
    before = list(calls)
    for argv in (["--sweep", named_sweep, "--out", tmp_path / "o", *calibrate],
                 ["--sweep", sweep, "--out", tmp_path / "out_test", *calibrate],
                 ["--sweep", sweep, "--out", tmp_path / "o", "--calibrate-from", "0=test", *calibrate[2:]]):
        code, _text, err = _run_cli(capsys, ["simulate", *argv])
        assert code == 2 and err.startswith("refused: ")
    assert calls == before and not (tmp_path / "o").exists()
    _git_states(monkeypatch, [simulation], repo=dirty_checkout)
    code, _text, err = _run_cli(capsys, ["simulate", "--sweep", sweep, "--out", tmp_path / "dirty", *calibrate])
    assert code == 2 and err.startswith("refused: ") and DIRTY_ENTRY not in err and not (tmp_path / "dirty").exists()
