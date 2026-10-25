"""Optimizer groups, GVP weight decay (TECH-032) and target shrink (synthetic)."""
import json
import math
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "analysis"))

from training.config import TrainConfig, parse_args
from training.run import (
    build_optimizer_param_groups, finalize_optimizer_groups, scheduler_epoch_lr_factors,
    validate_training_configuration,
)


class TinyGVP(torch.nn.Module):
    """Parameter names follow the GVP classifier: a trunk under ``layers.`` and encoders, plus heads."""

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(3, 3)])
        self.node_scalar_encoder = torch.nn.Linear(2, 3)
        self.gvp_fusion_proj = torch.nn.Linear(3, 3)
        self.head_ec = torch.nn.Linear(3, 7)


class TinyESM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(3, 3)
        self.head_ec = torch.nn.Linear(3, 7)


def _groups(model, steps_per_epoch=10, **overrides):
    config = TrainConfig(model_architecture="only_gvp", learning_rate=1e-3, weight_decay=1e-4, epochs=4, **overrides)
    return build_optimizer_param_groups(model, config, steps_per_epoch=steps_per_epoch)


def _names(record, group):
    return next(g for g in record["groups"] if g["name"] == group)["parameter_names"]


def test_gvp_weight_decay_alone_now_splits_groups():
    groups, record = _groups(TinyGVP(), gvp_weight_decay=0.5)
    assert record["split_reason"] == ["gvp_weight_decay"]
    assert [(g["lr"], g["weight_decay"]) for g in groups] == [(1e-3, 0.5), (1e-3, 1e-4)]
    assert all(name.startswith(("layers.", "node_scalar_encoder.")) for name in _names(record, "gvp_trunk"))
    assert "head_ec.weight" in _names(record, "base")


def test_equal_gvp_learning_rate_with_decay_splits():
    groups, record = _groups(TinyGVP(), gvp_learning_rate=1e-3, gvp_weight_decay=0.2)
    assert len(groups) == 2 and groups[0]["weight_decay"] == 0.2


def test_no_gvp_options_keeps_one_group_with_every_parameter():
    model = TinyGVP()
    model.head_ec.bias.requires_grad_(False)
    groups, record = _groups(model)
    assert len(groups) == 1 and record["split_reason"] == []
    assert len(groups[0]["params"]) == len(list(model.parameters()))
    assert record["gvp_rate_parameters"] == [] and "head_ec.bias" in record["base_rate_parameters"]


@pytest.mark.parametrize("scope", ["trunk", "structural"])
def test_distinct_learning_rate_matches_the_legacy_rule(scope):
    from analyze_effective_weight_decay import trainer_groups

    model = TinyGVP()
    config = TrainConfig(model_architecture="gvp", learning_rate=1e-3, weight_decay=1e-4, gvp_learning_rate=3e-4,
                         gvp_lr_scope=scope)
    groups, record = build_optimizer_param_groups(model, config, steps_per_epoch=5)
    legacy = trainer_groups(model, config)
    assert [(g["lr"], g["weight_decay"]) for g in groups] == [(g["lr"], g["weight_decay"]) for g in legacy]
    assert [[id(p) for p in g["params"]] for g in groups] == [[id(p) for _, p in g["named"]] for g in legacy]
    assert record["split_reason"] == ["distinct_gvp_learning_rate"]


def test_gvp_learning_rate_on_a_model_without_trunk_is_recorded_as_ignored():
    config = TrainConfig(model_architecture="only_esm", gvp_learning_rate=3e-4)
    groups, record = build_optimizer_param_groups(TinyESM(), config, steps_per_epoch=3)
    assert len(groups) == 1 and record["ignored_flags"] == ["gvp_learning_rate"]
    with pytest.raises(ValueError, match="GVP trunk"):
        build_optimizer_param_groups(TinyESM(), TrainConfig(model_architecture="gvp", gvp_weight_decay=0.1),
                                     steps_per_epoch=3)


@pytest.mark.parametrize("schedule", ["fixed", "cosine"])
def test_target_shrink_reaches_the_target(schedule):
    # Realistic EC fit size (about 600 proteins at batch 4); the exponential definition and the exact
    # per-step AdamW product then agree closely.
    steps, epochs, target = 150, 30, 0.059
    config = TrainConfig(model_architecture="only_gvp", learning_rate=3e-4, epochs=epochs, lr_schedule=schedule,
                         gvp_target_shrink=target)
    groups, record = build_optimizer_param_groups(TinyGVP(), config, steps_per_epoch=steps)
    factors = scheduler_epoch_lr_factors(config)
    wd = groups[0]["weight_decay"]
    assert math.exp(-wd * steps * 3e-4 * sum(factors)) == pytest.approx(target)
    detail = record["target_shrink"]["groups"]["gvp_trunk"]
    assert detail["computed_weight_decay"] == pytest.approx(wd)
    assert detail["planned_product_shrink"] == pytest.approx(target, rel=0.01)
    assert groups[1]["weight_decay"] == config.weight_decay
    if schedule == "cosine":
        assert factors[0] == pytest.approx(1.0) and factors[-1] < factors[0]


def test_target_shrink_reports_the_exact_product_for_small_fits():
    # Few large steps: the pre-declared exponential target holds, and the exact AdamW product is reported
    # separately because it shrinks further.
    config = TrainConfig(model_architecture="only_gvp", learning_rate=3e-4, epochs=5, gvp_target_shrink=0.059)
    groups, record = build_optimizer_param_groups(TinyGVP(), config, steps_per_epoch=7)
    detail = record["target_shrink"]["groups"]["gvp_trunk"]
    assert math.exp(-groups[0]["weight_decay"] * 7 * 3e-4 * 5) == pytest.approx(0.059)
    assert detail["planned_product_shrink"] < 0.059


def test_target_shrink_all_scope_sets_every_group():
    config = TrainConfig(model_architecture="only_gvp", learning_rate=1e-3, epochs=3, gvp_target_shrink=0.5,
                         gvp_target_shrink_scope="all")
    groups, record = build_optimizer_param_groups(TinyGVP(), config, steps_per_epoch=4)
    assert len(groups) == 1 and set(record["target_shrink"]["groups"]) == {"all"}
    assert math.exp(-groups[0]["weight_decay"] * 4 * 1e-3 * 3) == pytest.approx(0.5)


def test_v3_shrink_is_reproduced():
    # v3 Only-GVP: weight decay 1.0, 371 steps per epoch, 3e-4 peak LR, 50-epoch cosine -> about 0.0585.
    config = TrainConfig(model_architecture="only_gvp", learning_rate=3e-4, epochs=50, lr_schedule="cosine")
    factors = scheduler_epoch_lr_factors(config)
    assert math.exp(-1.0 * 371 * 3e-4 * sum(factors)) == pytest.approx(0.0585, abs=5e-4)


@pytest.mark.parametrize(
    "argv",
    [
        ["--model-architecture", "only_esm", "--gvp-weight-decay", "0.1"],
        ["--model-architecture", "only_esm", "--gvp-target-shrink", "0.1"],
        ["--model-architecture", "only_gvp", "--gvp-target-shrink", "1.5"],
        ["--model-architecture", "only_gvp", "--gvp-target-shrink", "0.1", "--gvp-weight-decay", "0.1"],
        ["--model-architecture", "only_gvp", "--gvp-weight-decay", "-1"],
    ],
)
def test_refusals_at_parse_and_validation(argv):
    with pytest.raises(SystemExit):
        parse_args(["--task", "metal", *argv])


def test_validation_mirrors_refusals():
    with pytest.raises(ValueError, match="require --model-architecture"):
        validate_training_configuration(TrainConfig(task="metal", model_architecture="only_esm", gvp_weight_decay=0.1))
    with pytest.raises(ValueError, match="in \\(0, 1\\)"):
        validate_training_configuration(TrainConfig(task="metal", model_architecture="only_gvp", gvp_target_shrink=2.0))


def test_finalize_records_realised_steps_and_fails_on_mismatch(tmp_path):
    model = TinyGVP()
    config = TrainConfig(model_architecture="only_gvp", learning_rate=1e-2, epochs=2, gvp_weight_decay=0.1)
    groups, record = build_optimizer_param_groups(model, config, steps_per_epoch=3)
    (tmp_path / "optimizer_groups.json").write_text(json.dumps(record))
    optimizer = torch.optim.AdamW(groups)
    for _ in range(6):
        loss = sum(param.sum() for param in model.parameters())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    final = finalize_optimizer_groups(tmp_path, optimizer, 2)
    assert final["realised_total_steps"] == [6, 6] and final["realised_steps_match"]
    with pytest.raises(ValueError, match="differ from the planned"):
        finalize_optimizer_groups(tmp_path, optimizer, 3)
    assert finalize_optimizer_groups(tmp_path, optimizer, 3, amp_active=True)["realised_steps_match"] is False


def test_tiny_only_gvp_fit_records_groups(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "tests" / "benchmarking"))
    from test_pinmymetal_training_options import tiny_argv
    from training.run import run_training

    argv = tiny_argv(tmp_path, monkeypatch, "gvp_decay", ["--gvp-weight-decay", "0.5", "--checkpoint-rule", "terminal"])
    argv[argv.index("only_esm")] = "only_gvp"
    run_dir = run_training(parse_args(argv))
    record = json.loads((run_dir / "optimizer_groups.json").read_text())
    assert record["split_reason"] == ["gvp_weight_decay"] and record["realised_steps_match"]
    assert [g["weight_decay"] for g in record["groups"]] == [0.5, parse_args(argv).weight_decay]
    checkpoint = torch.load(run_dir / "terminal_model_checkpoint.pt", map_location="cpu", weights_only=False)
    saved = checkpoint["optimizer_state_dict"]["param_groups"]
    assert [g["weight_decay"] for g in saved] == [g["weight_decay"] for g in record["groups"]]
    history = json.loads((run_dir / "run_config.json").read_text())["history"]
    assert all(len(epoch["lr_groups"]) == 2 for epoch in history)
