"""Tests for the effective weight-decay audit (audit_v3_effective_decay.py): groups and the zero-gradient run."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import audit_v3_effective_decay as audit  # noqa: E402


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4)])  # "layers." is a GVP-rate prefix in the trainer
        self.init_vec_proj = nn.Linear(4, 4)           # joins the GVP-rate group only under the structural scope
        self.head_metal = nn.Linear(4, 2)


def config(**overrides):
    values = dict(learning_rate=3e-5, gvp_learning_rate=3e-4, gvp_lr_scope="trunk", weight_decay=0.1,
                  gvp_weight_decay=None, lr_schedule="cosine", epochs=4, batch_size=16)
    return SimpleNamespace(**{**values, **overrides})


def test_groups_mirror_the_trainer_and_are_checked_against_the_saved_optimizer():
    model = Tiny()
    groups = audit.trainer_groups(model, config())
    assert [g["group"] for g in groups] == ["gvp_rate (trunk)", "base_rate"]
    assert [name for name, _ in groups[0]["named"]] == ["layers.0.weight", "layers.0.bias"]
    assert (groups[0]["lr"], groups[1]["lr"]) == (3e-4, 3e-5) and groups[0]["weight_decay"] == 0.1  # inherited
    structural = audit.trainer_groups(model, config(gvp_lr_scope="structural"))
    assert [name for name, _ in structural[0]["named"]] == ["layers.0.weight", "layers.0.bias",
                                                           "init_vec_proj.weight", "init_vec_proj.bias"]
    single = audit.trainer_groups(model, config(gvp_learning_rate=None, learning_rate=3e-4))
    assert [g["group"] for g in single] == ["all"] and len(single[0]["named"]) == 6
    separate = audit.trainer_groups(model, config(gvp_weight_decay=0.5))
    assert (separate[0]["weight_decay"], separate[1]["weight_decay"]) == (0.5, 0.1)
    saved = {"optimizer_state_dict": {"param_groups": [
        {"params": [0, 1], "initial_lr": 3e-4, "weight_decay": 0.1},
        {"params": [2, 3, 4, 5], "initial_lr": 3e-5, "weight_decay": 0.1}]}}
    assert audit.verify_against_saved_groups(groups, saved)["passed"]
    assert not audit.verify_against_saved_groups(structural, saved)["passed"]


def test_zero_gradient_run_measures_the_decay_alone_and_skips_parameters_without_gradients():
    torch.manual_seed(0)
    model = Tiny()
    cfg = config(weight_decay=1.0)
    groups = audit.trainer_groups(model, cfg)
    untouched = model.head_metal.bias.detach().clone()
    steps = 50
    out = audit.zero_gradient_decay(cfg, groups, steps_per_epoch=steps, updated={0, 1, 2, 3, 4})  # head bias: no gradient
    rates = [0.5 * (1 + math.cos(math.pi * epoch / cfg.epochs)) for epoch in range(cfg.epochs)]
    for group, lr in zip(out, (3e-4, 3e-5)):
        ideal = math.prod((1 - lr * rate * 1.0) ** steps for rate in rates)
        assert group["ideal_cumulative_factor"] == pytest.approx(ideal, rel=1e-12)
        assert group["measured_fp32_factor"]["median"] == pytest.approx(ideal, abs=1e-5)
        assert abs(group["measured_minus_ideal"]) < 1e-5 and group["first_epoch_lr"] == lr
    assert out[1]["not_updated_no_gradient_in_the_real_run"] == ["head_metal.bias"]
    assert torch.equal(model.head_metal.bias.detach(), untouched)  # the optimizer never touches it
    assert out[0]["ideal_fraction_removed"] > 9 * out[1]["ideal_fraction_removed"]  # one coefficient, two strengths


def test_a_tiny_coefficient_is_inert_in_float32_at_the_low_rate():
    model = Tiny()
    cfg = config(weight_decay=1e-4)
    out = audit.zero_gradient_decay(cfg, audit.trainer_groups(model, cfg), steps_per_epoch=20, updated=set(range(6)))
    assert out[1]["first_step_factor_is_one_in_float32"] and out[1]["measured_fp32_factor"]["median"] == 1.0
    assert out[1]["ideal_cumulative_factor"] < 1.0  # the ideal product still shrinks; float32 does not (TECH-029)
    assert not out[0]["first_step_factor_is_one_in_float32"]
