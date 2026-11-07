"""Amendment 2: F6 and raw-distance (P14) recipes, their trainer arguments and the replay of the historical plans.

Synthetic smoke views only; the historical plan hashes are read from the tracked Freeze-1 record.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT / "tests")]

from ec_clean30 import campaign, pretrain_corpus as pc  # noqa: E402
from ec_clean30.campaign import (  # noqa: E402
    AMENDMENT2_RBF_DISTANCES, F6_PRETRAINING, FAMILY_SPECS, FAMILY_SPECS_V2, KIND_INNER, KIND_REFIT, CampaignPlanError,
    PlannedUnit, Recipe, amendment2_recipe, build_fold_plan, default_smoke_recipe, plan_units, recipe_for_unit,
    recipe_with_lever, smoke_unit_id,
)
from ec_clean30.commands import CommandError, build_unit_command, check_command  # noqa: E402
from ec_clean30.views import SMOKE_VIEWS_DIR  # noqa: E402
from ec_clean30_fixtures import build_smoke_view, unlock, write_clean30_data_root  # noqa: E402
from test_ec_clean30_pretrain_corpus import SMOKE_PROVENANCE, build_smoke_view_with_corpus  # noqa: E402

FREEZE1 = ROOT / "docs" / "campaigns" / "ec_class_vs_clean30_2026-10" / "freeze1.json"


@pytest.fixture(scope="module")
def f6_view(tmp_path_factory):
    view, info = build_smoke_view_with_corpus(tmp_path_factory.mktemp("f6cmd"))
    yield view
    unlock(info["source"])


@pytest.fixture(scope="module")
def plain_view(tmp_path_factory):
    base = tmp_path_factory.mktemp("f6plain")
    source = write_clean30_data_root(base / "source")
    yield build_smoke_view(source, base / campaign.CAMPAIGN_ID / SMOKE_VIEWS_DIR / "plain", provenance=SMOKE_PROVENANCE)
    unlock(source)


def _command(view, recipe: Recipe, *, refit: bool = False):
    unit_id = smoke_unit_id(view.fold, recipe, refit=refit, seed=42)
    return build_unit_command(view, recipe, unit_id=unit_id, kind=KIND_REFIT if refit else KIND_INNER, seed=42,
                              runs_dir=view.default_runs_dir, step="smoke", inner_fold=None if refit else 0)


def _head(root: str, view, architecture: str, extra_head: list[str]) -> list[str]:
    return [
        "--task", "ec", "--ec-label-depth", "1", "--ec-class-vocabulary", "fixed_seven", "--ec-conflicting-labels",
        "fail", "--site-extraction", "selected_sites", "--selected-site-pocket-grouping", "structure",
        "--model-architecture", architecture, "--learning-rate", "3e-4", "--use-global-esm-branch",
        "--global-esm-cache-dir", f"{root}/chain_means_final", "--global-esm-manifest-sha256",
        view.chain_means_manifest_sha256("final"), *extra_head,
        "--ec-group-weighting", "structure_id", "--ec-class-weight-unit", "group", "--batch-size", "4", "--epochs", "1",
        "--lr-schedule", "cosine", "--checkpoint-rule", "terminal", "--edge-radius", "6.0", "--ec-contrastive-weight",
        "0", "--no-prepare-missing-esm-embeddings", "--no-prepare-missing-ring-edges",
        "--no-prepare-missing-external-features",
    ]


def _inner_tail(root: str, view) -> list[str]:
    return [
        "--fold-split-source", "membership", "--fold-membership-csv", f"{root}/smoke_membership.csv",
        "--fold-membership-sha256", view.membership["sha256"], "--n-folds", "2", "--fold-index", "0",
        "--export-ec-predictions", "--structure-dir", f"{root}/structures", "--summary-csv",
        f"{root}/site_summary_train.csv", "--esm-embeddings-dir", f"{root}/esm", "--external-features-root-dir",
        f"{root}/external", "--dataset-bundle-id", "g1-smoke-view", "--dataset-bundle-sha256", view.manifest_sha256,
        "--device", "cpu", "--load-workers", "1", "--num-workers", "0", "--seed", "42",
    ]


def test_the_f5_argv_and_identity_keys_are_unchanged(plain_view):
    view = plain_view
    command = _command(view, default_smoke_recipe("F5"))
    root = str(view.root)
    reference = _head(root, view, "only_esm", []) + _inner_tail(root, view)
    assert list(command.argv[: len(reference)]) == reference
    assert list(command.argv[len(reference):len(reference) + 6]) == [
        "--ring-features-dir", f"{root}/ring", "--runs-dir", str(view.default_runs_dir), "--run-name", command.unit_id]
    assert not any(token.startswith(("--global-esm-pretrain", "--global-esm-branch-init", "--rbf-use-raw"))
                   for token in command.argv)
    assert set(command.identity) == {
        "campaign_id", "unit_id", "step", "kind", "fold", "inner_fold", "seed", "smoke", "recipe",
        "view_manifest_sha256", "allowlist_sha256", "membership_sha256", "probe_sha256",
        "chain_means_manifest_sha256", "freeze1_sha256", "freeze1b_sha256", "freeze1_amendment_chain", "overlay_id",
        "overlay_sha256", "git_head", "source_tree_sha256", "resolved_config_sha256"}
    assert command.identity["recipe"] == {"family": "F5", "learning_rate": {"learning_rate": 3e-4}, "epochs": 1,
                                          "lever_configs": []}
    assert check_command(command, view) == []


@pytest.mark.parametrize("refit", [False, True], ids=["inner", "refit"])
def test_f6_commands_render_the_pretraining_inputs(f6_view, refit):
    from training.config import parse_args

    view = f6_view
    inputs = pc.view_pretrain_inputs(view)
    command = _command(view, default_smoke_recipe("F6"), refit=refit)
    root = str(view.root)
    pretrain = [
        "--global-esm-branch-init", "pretrained_frozen",
        "--global-esm-pretrain-cache-dir", f"{root}/whole_protein_pretrain/seq_means_final",
        "--global-esm-pretrain-manifest-sha256", inputs.manifest_sha256,
        "--global-esm-pretrain-corpus", f"{root}/whole_protein_pretrain/corpus.tsv",
        "--global-esm-pretrain-corpus-sha256", inputs.corpus_sha256,
        "--global-esm-pretrain-split", "refit" if refit else "inner0",
        "--global-esm-pretrain-epochs", "30", "--global-esm-pretrain-learning-rate", "1e-3",
        "--global-esm-pretrain-batch-size", "64",
    ]
    reference = _head(root, view, "only_esm", pretrain)
    assert list(command.argv[: len(reference)]) == reference
    if not refit:
        assert list(command.argv[len(reference):len(reference) + len(_inner_tail(root, view))]) == \
            _inner_tail(root, view)
    assert check_command(command, view) == []
    assert command.identity["wp_pretrain_corpus_sha256"] == inputs.corpus_sha256
    assert command.identity["wp_pretrain_manifest_sha256"] == inputs.manifest_sha256
    assert command.identity["recipe"]["family"] == "F6" and "rbf_distances" not in command.identity["recipe"]
    config = parse_args(list(command.argv))
    assert (config.model_architecture, config.use_global_esm_branch, config.global_esm_branch_init) == \
        ("only_esm", True, "pretrained_frozen")
    assert (config.global_esm_pretrain_epochs, config.global_esm_pretrain_learning_rate,
            config.global_esm_pretrain_batch_size) == (F6_PRETRAINING["epochs"], F6_PRETRAINING["learning_rate"],
                                                       F6_PRETRAINING["batch_size"])
    assert config.global_esm_pretrain_split == ("refit" if refit else "inner0")
    if refit:
        assert config.val_fraction == 0.0 and config.n_folds is None


def _set(argv, flag, value):
    argv = list(argv)
    argv[argv.index(flag) + 1] = value
    return argv


def test_the_static_check_binds_the_pretraining_inputs(f6_view, plain_view, tmp_path):
    view = f6_view
    command = _command(view, default_smoke_recipe("F6"))
    outside = tmp_path / "corpus.tsv"
    outside.write_bytes(pc.view_pretrain_inputs(view).corpus.read_bytes())
    for value, message in ((str(outside), "lies outside the view"),
                           (str(view.caches / "corpus.tsv"), "points into the view's writable caches or receipts")):
        problems = check_command(replace(command, argv=tuple(_set(command.argv, "--global-esm-pretrain-corpus", value))),
                                 view)
        assert any(message in problem for problem in problems), problems
    split = check_command(replace(command, argv=tuple(_set(command.argv, "--global-esm-pretrain-split", "inner1"))),
                          view)
    assert any("the trainer refuses the arguments" in problem for problem in split), split
    tampered = {**command.identity, "wp_pretrain_corpus_sha256": "0" * 64}
    argv = _set(command.argv, "--campaign-run-identity", campaign.canonical_json(tampered))
    problems = check_command(replace(command, argv=tuple(argv), identity=tampered), view)
    assert any("wp_pretrain_corpus_sha256 does not match" in problem for problem in problems), problems
    f5 = _command(view, default_smoke_recipe("F5"))
    extra = list(f5.argv) + ["--global-esm-pretrain-split", "refit"]
    assert any("the trainer refuses the arguments" in problem for problem in check_command(replace(f5, argv=tuple(extra)),
                                                                                           view))
    with pytest.raises(CommandError, match="whole-protein pretraining corpus"):
        _command(plain_view, default_smoke_recipe("F6"))


@pytest.mark.parametrize("family", ["F1", "F2", "F3", "F4", "F5", "F6", "C-G"])
def test_amendment2_gvp_recipes_render_raw_distance_rbfs(f6_view, family):
    from training.config import parse_args

    recipe = default_smoke_recipe(family, rbf_distances=AMENDMENT2_RBF_DISTANCES[family])
    command = _command(f6_view, recipe)
    raw = FAMILY_SPECS_V2[family].has_gvp_trunk
    assert command.argv.count("--rbf-use-raw-distances") == (1 if raw else 0)
    assert check_command(command, f6_view) == []
    config = parse_args(list(command.argv))
    assert config.node_rbf_use_raw_distances is raw and config.edge_rbf_use_raw_distances is raw
    assert config.edge_radius == 6.0 and config.metal_node_mode == "none"
    assert ("rbf_distances" in command.identity["recipe"]) is raw
    if raw:
        assert command.unit_id.endswith("_rbfraw__inner0__seed42")
        historical = _command(f6_view, default_smoke_recipe(family))
        assert "--rbf-use-raw-distances" not in historical.argv and historical.unit_id != command.unit_id


def test_recipe_versioning_keeps_the_historical_forms():
    grid = campaign.recipe_learning_rate_grid
    for family in ("F2", "F5", "F6", "C-G"):
        with pytest.raises(CampaignPlanError, match="no GVP trunk"):
            Recipe(family, grid(family)[0], rbf_distances="raw")
    with pytest.raises(CampaignPlanError, match="rbf_distances"):
        Recipe("F3", grid("F3")[0], rbf_distances="angstrom")
    with pytest.raises(CampaignPlanError, match="L2 already sets"):
        Recipe("F3", grid("F3")[0], lever_configs=("L2",), rbf_distances="raw")
    with pytest.raises(CampaignPlanError, match="no S2 lever"):
        Recipe("F6", grid("F6")[0], lever_configs=("L6",))
    with pytest.raises(CampaignPlanError, match="not in the S1 grid"):
        Recipe("F6", (("learning_rate", 1e-3),))
    assert grid("F6") == grid("F5")
    historical = Recipe("F3", grid("F3")[1])
    assert historical.to_dict() == {"family": "F3", "learning_rate": dict(grid("F3")[1]), "epochs": 30,
                                    "lever_configs": []}
    raw = amendment2_recipe("F3", grid("F3")[1])
    assert raw.to_dict() == {**historical.to_dict(), "rbf_distances": "raw"}
    assert Recipe.from_dict(raw.to_dict()) == raw and Recipe.from_dict(historical.to_dict()) == historical
    assert recipe_with_lever(raw, "L7e12").rbf_distances == "raw"
    assert amendment2_recipe("F2", grid("F2")[0]).rbf_distances == "normalized"
    assert "F6" not in FAMILY_SPECS and FAMILY_SPECS_V2["F6"] == campaign.F6_FAMILY_SPEC
    assert FAMILY_SPECS_V2["F6"].model_architecture == FAMILY_SPECS["F5"].model_architecture == "only_esm"
    assert campaign.wp_pretrain_split(KIND_INNER, 2) == "inner2" and campaign.wp_pretrain_split(KIND_REFIT, None) == \
        "refit"
    assert campaign.WHOLE_PROTEIN_PRETRAIN_CONTROL["selectable"] is False
    with pytest.raises(CampaignPlanError):
        campaign.check_family("F6")  # the historical plan builder still knows only the Freeze-1 families


def test_planned_units_carry_the_raw_encoding_only_when_set():
    plan = build_fold_plan(0)
    assert all("rbf_distances" not in unit for unit in plan["units"])
    unit = plan_units(plan)[0]
    assert unit.rbf_distances is None and "rbf_distances" not in unit.to_dict()
    gvp = next(item for item in plan_units(plan) if item.family == "F3")
    raw = replace(gvp, rbf_distances="raw")
    assert raw.to_dict()["rbf_distances"] == "raw"
    assert recipe_for_unit(raw).rbf_distances == "raw" and recipe_for_unit(gvp).rbf_distances == "normalized"
    with pytest.raises(CampaignPlanError):
        replace(gvp, rbf_distances="metres")
    body = {**{key: value for key, value in plan.items() if key != "plan_sha256"},
            "units": [raw.to_dict() if item["unit_id"] == gvp.unit_id else item for item in plan["units"]]}
    rebuilt = plan_units({**body, "plan_sha256": campaign.canonical_sha256(body)})
    assert next(item for item in rebuilt if item.unit_id == gvp.unit_id).rbf_distances == "raw"


def test_the_historical_plans_and_constants_replay_byte_for_byte():
    record = json.loads(FREEZE1.read_text(encoding="utf-8"))
    for fold in range(campaign.N_OUTER_FOLDS):
        assert build_fold_plan(fold)["plan_sha256"] == record["plans"][str(fold)]["plan_sha256"]
    assert json.loads(json.dumps(campaign.campaign_constants())) == record["constants"]["campaign"]["campaign_constants"]
