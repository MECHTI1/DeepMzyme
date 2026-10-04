"""Tests for the v3 step-D model options (plan step A2): all off by default."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model import SimpleGVPLayer, vector_channel_dropout, vector_layer_norm  # noqa: E402
from model_variants import build_pocket_classifier  # noqa: E402
from training.config import parse_args  # noqa: E402
from training.run import GVP_STRUCTURAL_PREFIXES, GVP_TRUNK_PREFIXES, gvp_learning_rate_prefixes  # noqa: E402

DIMS = dict(esm_dim=8, hidden_s=16, hidden_v=4, edge_hidden=8, n_layers=2, n_metal=6, n_ec=1,
            esm_fusion_dim=8, predict_ec=False)


def build(architecture="gvp", seed=0, **options):
    torch.manual_seed(seed)
    return build_pocket_classifier(model_architecture=architecture, **DIMS, **options)


def layer_inputs(seed=1, n=6, s_dim=16, v_dim=4, e_dim=8):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(n, s_dim, generator=g)
    v = torch.randn(n, v_dim, 3, generator=g)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 1], [1, 2, 3, 4, 5, 0, 3]])
    edge_s = torch.randn(edge_index.size(1), e_dim, generator=g)
    edge_v = torch.randn(edge_index.size(1), 1, 3, generator=g)
    return s, v, edge_index, edge_s, edge_v


@pytest.mark.parametrize("architecture", ["gvp", "only_gvp"])
def test_defaults_keep_identical_parameters(architecture):
    plain = build(architecture)
    explicit = build(architecture, gvp_residual_dropout=0.0, gvp_vector_norm=False)
    assert plain.state_dict().keys() == explicit.state_dict().keys()
    for name, tensor in plain.state_dict().items():
        assert torch.equal(tensor, explicit.state_dict()[name]), name


def test_residual_dropout_only_acts_in_training_and_drops_whole_vector_channels():
    torch.manual_seed(0)
    base = SimpleGVPLayer(16, 4, 8)
    torch.manual_seed(0)
    dropped = SimpleGVPLayer(16, 4, 8, residual_dropout=0.5)
    inputs = layer_inputs()
    base.eval(), dropped.eval()
    for a, b in zip(base(*inputs), dropped(*inputs)):
        assert torch.equal(a, b)
    v = torch.ones(5, 4, 3)
    torch.manual_seed(3)
    out = vector_channel_dropout(v, 0.5, training=True)
    per_channel = out.reshape(-1, 3)
    assert all(row.unique().numel() == 1 for row in per_channel)  # all three components share one fate
    assert set(per_channel[:, 0].tolist()) <= {0.0, 2.0}
    assert torch.equal(vector_channel_dropout(v, 0.5, training=False), v)


def test_vector_norm_gives_unit_mean_squared_channel_norm():
    v = torch.randn(7, 4, 3) * 5.0
    normed = vector_layer_norm(v)
    mean_sq = (normed * normed).sum(-1).mean(-1)
    assert torch.allclose(mean_sq, torch.ones(7), atol=1e-5)
    layer = SimpleGVPLayer(16, 4, 8, vector_norm=True).eval()
    _, v_out = layer(*layer_inputs())
    assert torch.allclose((v_out * v_out).sum(-1).mean(-1), torch.ones(v_out.size(0)), atol=1e-4)


def test_auxiliary_head_is_built_last_and_never_changes_shared_initialization():
    control = build("gvp")
    aux = build("gvp", gvp_auxiliary_loss_weight=0.3)
    extra = set(aux.state_dict()) - set(control.state_dict())
    assert extra == {"gvp_aux_head.weight", "gvp_aux_head.bias"}
    for name, tensor in control.state_dict().items():
        assert torch.equal(tensor, aux.state_dict()[name]), name


def test_modality_options_require_late_fusion_with_esm():
    with pytest.raises(ValueError):
        build("only_gvp", gvp_auxiliary_loss_weight=0.3)
    with pytest.raises(ValueError):
        build("only_gvp", esm_modality_dropout=0.2)
    with pytest.raises(ValueError):
        build("only_esm", gvp_residual_dropout=0.1)
    build("only_esm")  # defaults are accepted by non-GVP families


def test_structural_learning_rate_scope_moves_only_structural_modules():
    model = build("gvp")
    names = [name for name, _ in model.named_parameters()]

    def owned(prefixes):
        return {name for name in names if any(name.startswith(p) for p in prefixes)}

    trunk, structural = owned(gvp_learning_rate_prefixes("trunk")), owned(gvp_learning_rate_prefixes("structural"))
    moved = structural - trunk
    assert trunk < structural and moved
    assert all(name.split(".")[0] in {"init_vec_proj", "gvp_attn_pool", "gvp_fusion_proj"} for name in moved)
    assert not any(name.startswith(("esm_", "fusion_gate", "head_metal")) for name in structural)
    assert GVP_STRUCTURAL_PREFIXES[: len(GVP_TRUNK_PREFIXES)] == GVP_TRUNK_PREFIXES


def test_old_configs_without_v3_fields_still_rebuild_their_model():
    import ast

    from export_validation_predictions import V3_OPTIONAL_CONFIG_DEFAULTS, factory_kwargs

    root = Path(__file__).resolve().parents[1]
    config = {key: value for key, value in vars(parse_args(["--task", "metal"])).items()}
    for key in V3_OPTIONAL_CONFIG_DEFAULTS:
        config.pop(key, None)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in config.items()}
    checkpoint = {"model_state_dict": {}}
    kwargs = factory_kwargs(root / "src" / "training" / "run.py", config, checkpoint, ["Mn"] * 6, {})
    for key, default in V3_OPTIONAL_CONFIG_DEFAULTS.items():
        assert kwargs[key] == default
    assert ast  # module import sanity


@pytest.mark.parametrize("argv", [
    ["--model-architecture", "only_esm", "--gvp-residual-dropout", "0.1"],
    ["--model-architecture", "only_gvp", "--gvp-auxiliary-loss-weight", "0.3"],
    ["--model-architecture", "gvp", "--gvp-lr-scope", "structural"],
    ["--model-architecture", "gvp", "--esm-modality-dropout", "1.0"],
])
def test_cli_rejects_invalid_combinations(argv):
    with pytest.raises(SystemExit):
        parse_args(["--task", "metal", *argv])


def test_cli_accepts_the_round_a_settings():
    args = parse_args(["--task", "metal", "--model-architecture", "gvp", "--gvp-learning-rate", "3e-4",
                       "--gvp-lr-scope", "structural", "--gvp-residual-dropout", "0.1",
                       "--gvp-auxiliary-loss-weight", "0.3", "--esm-modality-dropout", "0.2", "--gvp-vector-norm"])
    assert (args.gvp_lr_scope, args.gvp_residual_dropout, args.gvp_auxiliary_loss_weight,
            args.esm_modality_dropout, args.gvp_vector_norm) == ("structural", 0.1, 0.3, 0.2, True)


def test_tiny_late_fusion_fit_with_every_step_d_option(tmp_path, monkeypatch):
    import json

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_v3_training_options import tiny_argv
    from training.run import run_training

    argv = tiny_argv(tmp_path, monkeypatch, "all_options", [
        "--checkpoint-rule", "terminal", "--lr-schedule", "cosine", "--fold-split-source", "membership",
        "--gvp-learning-rate", "1e-3", "--gvp-lr-scope", "structural", "--gvp-residual-dropout", "0.1",
        "--gvp-vector-norm", "--gvp-auxiliary-loss-weight", "0.3", "--esm-modality-dropout", "0.2",
        "--fusion-mode", "late_fusion",
    ])
    argv[argv.index("only_esm")] = "gvp"
    run_dir = run_training(parse_args(argv))
    groups = json.loads((run_dir / "optimizer_groups.json").read_text())
    assert any(name.startswith("gvp_fusion_proj.") for name in groups["gvp_rate_parameters"])
    assert all(not name.startswith("gvp_fusion_proj.") for name in groups["base_rate_parameters"])
    assert any(name.startswith("gvp_aux_head.") for name in groups["base_rate_parameters"])
    receipt = json.loads((run_dir / "selected_checkpoint.json").read_text())
    assert receipt["selected_epoch"] == 3 and receipt["reconciliation_status"] == "match"
