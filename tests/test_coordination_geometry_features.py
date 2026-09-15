"""Controlled site geometry inputs and an actual GVP angular-information path."""

from pathlib import Path
import math
import sys

import pytest
import torch
from torch_geometric.data import Batch, Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_structures import EDGE_SOURCE_TYPES, INTERACTION_SUMMARIES_OPTIONAL_WITH_RING
from model import DEFAULT_NODE_RESCHEM_DIM, GVPPocketClassifier, SimpleGVPLayer
from model_variants.factory import build_pocket_classifier
from training.config import TrainConfig, config_to_payload, parse_args
from training.graph_dataset import apply_feature_normalization, compute_feature_normalization_stats


def small_model(**kwargs):
    options = dict(
        esm_dim=4, hidden_s=8, hidden_v=2, edge_hidden=4, n_layers=4,
        n_metal=4, n_ec=1, esm_fusion_dim=4, site_feature_dim=4,
        predict_ec=False, use_esm_branch=False, head_mlp_dropout=0.0,
        structural_readout_scope="residue_only",
    )
    options.update(kwargs)
    return GVPPocketClassifier(**options).eval()


def site_data():
    return Data(
        site_metal_stats=torch.tensor([[2., 1., 3., 4.]]),
        site_ligand_angle_stats=torch.tensor([[4., 6., 60., 90., 120., 30., 20., 10.]]),
        site_ligand_angle_stats_raw=torch.tensor([[4., 6., 60., 90., 120., 30., 20., 10.]]),
    )


def site_inputs(model, data):
    return model._site_feature_tensor(data, 1, torch.float32, torch.device("cpu"))


@pytest.mark.parametrize("angle_enabled", [False, True])
def test_legacy_preserves_raw_inputs_checkpoint_shapes_and_default_initialization(angle_enabled):
    options = dict(use_site_angle_features=angle_enabled, use_node_type_embedding=angle_enabled)
    torch.manual_seed(31)
    default = small_model(**options)
    torch.manual_seed(31)
    legacy = small_model(site_geometry_features="legacy", **options)
    assert default.site_feature_encoder[0].in_features == (12 if angle_enabled else 4)
    assert (default.node_type_embedding is not None) == angle_enabled
    assert default.state_dict().keys() == legacy.state_dict().keys()
    for key, value in default.state_dict().items():
        torch.testing.assert_close(value, legacy.state_dict()[key], rtol=0, atol=0)
    legacy.load_state_dict(default.state_dict(), strict=True)
    data = site_data()
    expected = torch.cat([data.site_metal_stats, data.site_ligand_angle_stats], -1) if angle_enabled else data.site_metal_stats
    torch.testing.assert_close(site_inputs(legacy, data), expected, rtol=0, atol=0)
    del data.site_ligand_angle_stats
    fallback = site_inputs(legacy, data)
    torch.testing.assert_close(fallback[:, :4], data.site_metal_stats)
    assert torch.count_nonzero(fallback[:, 4:]) == 0


@pytest.mark.parametrize("mode", ["none", "counts", "counts_angles"])
def test_explicit_geometry_masks_and_units_are_independent_of_legacy_angle_switch(mode):
    data = site_data()
    model = small_model(site_geometry_features=mode, use_site_angle_features=False)
    result = site_inputs(model, data)
    assert model.site_feature_encoder[0].in_features == 12
    assert model.node_type_embedding is not None
    assert model.node_type_embedding.num_embeddings == 2  # residue / generic metal, no element label
    torch.testing.assert_close(result[:, :4], data.site_metal_stats)
    expected = torch.zeros(1, 8)
    if mode != "none":
        expected[:, :2] = torch.tensor([[math.log(5), math.log(7)]])
    if mode == "counts_angles":
        expected[:, 2:] = torch.tensor([[1/3, 1/2, 2/3, 1/6, 1/9, 1/18]])
    torch.testing.assert_close(result[:, 4:], expected)
    torch.testing.assert_close(
        result, site_inputs(small_model(site_geometry_features=mode, use_site_angle_features=True), data),
    )


def test_explicit_arms_share_all_parameter_shapes_and_initial_values():
    states = []
    for mode in ("none", "counts", "counts_angles"):
        torch.manual_seed(17)
        states.append(small_model(site_geometry_features=mode).state_dict())
    for state in states[1:]:
        assert state.keys() == states[0].keys()
        for key, value in state.items():
            torch.testing.assert_close(value, states[0][key], rtol=0, atol=0)


def test_counts_control_cannot_depend_on_masked_angles():
    model = small_model(site_geometry_features="counts")
    data = site_data()
    reference = site_inputs(model, data)
    data.site_ligand_angle_stats_raw[:, 2:] = torch.tensor([[180., 140., 150., 5., 1., 3.]])
    torch.testing.assert_close(reference, site_inputs(model, data), rtol=0, atol=0)
    data.site_ligand_angle_stats_raw[:, 2:] = float("nan")
    torch.testing.assert_close(reference, site_inputs(model, data), rtol=0, atol=0)
    data.site_ligand_angle_stats_raw.requires_grad_()
    site_inputs(model, data).sum().backward()
    assert torch.count_nonzero(data.site_ligand_angle_stats_raw.grad[:, :2]) == 2
    assert torch.count_nonzero(data.site_ligand_angle_stats_raw.grad[:, 2:]) == 0


@pytest.mark.parametrize("mode", ["counts", "counts_angles"])
def test_explicit_measured_inputs_reject_missing_or_invalid_geometry(mode):
    model = small_model(site_geometry_features=mode)
    with pytest.raises(ValueError, match="require site_ligand_angle_stats_raw"):
        site_inputs(model, Data(site_metal_stats=torch.zeros(1, 4), site_ligand_angle_stats=torch.zeros(1, 8)))
    data = site_data()
    data.site_ligand_angle_stats_raw[0, 0] = -1
    with pytest.raises(ValueError, match="non-negative counts"):
        site_inputs(model, data)
    data.site_ligand_angle_stats_raw = torch.zeros(1, 7)
    with pytest.raises(ValueError, match="must have shape"):
        site_inputs(model, data)


def test_normalization_preserves_physical_geometry_for_explicit_modes_and_legacy_values():
    original = site_data()
    del original.site_ligand_angle_stats_raw
    second = original.clone()
    second.site_ligand_angle_stats *= 2
    stats = compute_feature_normalization_stats([original, second])
    normalized = apply_feature_normalization(original.clone(), stats)
    torch.testing.assert_close(normalized.site_ligand_angle_stats_raw, original.site_ligand_angle_stats)
    assert (normalized.site_ligand_angle_stats[:, :2] < 0).all()
    normalized.num_nodes = 1
    batch = Batch.from_data_list([normalized])
    explicit = site_inputs(small_model(site_geometry_features="counts_angles"), batch)
    torch.testing.assert_close(explicit[:, 4:6], torch.log1p(original.site_ligand_angle_stats[:, :2]))
    torch.testing.assert_close(explicit[:, 6:], original.site_ligand_angle_stats[:, 2:] / 180.)
    legacy = site_inputs(small_model(use_site_angle_features=True), batch)
    torch.testing.assert_close(legacy[:, 4:], normalized.site_ligand_angle_stats)
    reapplied = apply_feature_normalization(normalized, stats)
    torch.testing.assert_close(reapplied.site_ligand_angle_stats_raw, original.site_ligand_angle_stats)


@pytest.mark.parametrize("architecture", ["only_esm", "simple_gnn_esm"])
def test_unsupported_families_reject_explicit_geometry_in_cli_and_factory(architecture):
    with pytest.raises(SystemExit):
        parse_args(["--model-architecture", architecture, "--site-geometry-features", "counts"])
    with pytest.raises(ValueError, match="requires gvp or only_gvp"):
        build_pocket_classifier(model_architecture=architecture, esm_dim=4, site_geometry_features="counts")
    # Passing the new default through existing callers must remain compatible.
    model = build_pocket_classifier(model_architecture=architecture, esm_dim=4, site_geometry_features="legacy")
    assert model is not None


@pytest.mark.parametrize("mode", ["legacy", "none", "counts", "counts_angles"])
def test_cli_records_geometry_mode_with_independent_metal_nodes(mode):
    for metal_mode in ("none", "per_metal"):
        config = parse_args([
            "--task", "metal", "--model-architecture", "only_gvp",
            "--site-geometry-features", mode, "--metal-node-mode", metal_mode,
            "--structural-readout-scope", "residue_only",
        ])
        assert config_to_payload(config)["site_geometry_features"] == mode
        assert config.metal_node_mode == metal_mode
        assert config.structural_readout_scope == "residue_only"
    assert TrainConfig().site_geometry_features == "legacy"


def two_ligand_star(angle):
    """Keep every scalar input fixed; only the two bond directions vary."""
    angle = torch.as_tensor(angle)
    u = torch.stack([angle.new_tensor(1.), angle.new_tensor(0.), angle.new_tensor(0.)])
    v = torch.stack([angle.cos(), angle.sin(), angle.new_tensor(0.)])
    return Data(
        batch=torch.zeros(3, dtype=torch.long), num_nodes=3,
        x_reschem=torch.zeros(3, DEFAULT_NODE_RESCHEM_DIM),
        hydrophobicity_kd=torch.zeros(3, 1), x_role=torch.zeros(3, 2),
        x_dist_raw=torch.ones(3, 3), x_misc=torch.zeros(3, 1),
        x_env_burial=torch.zeros(3, 1), x_env_electrostatics=torch.zeros(3, 2),
        x_vec=torch.zeros(3, 2, 3), x_esm=torch.zeros(3, 4),
        node_type_id=torch.tensor([0, 0, 1]),
        residue_node_mask=torch.tensor([True, True, False]),
        metal_node_mask=torch.tensor([False, False, True]),
        edge_index=torch.tensor([[0, 1, 2, 2], [2, 2, 0, 1]]),
        edge_dist_raw=torch.ones(4, 2), edge_seqsep=torch.zeros(4, 1),
        edge_same_chain=torch.zeros(4, 1),
        edge_interaction_type=torch.zeros(4, len(INTERACTION_SUMMARIES_OPTIONAL_WITH_RING)),
        edge_source_type=torch.zeros(4, len(EDGE_SOURCE_TYPES)),
        edge_vector_raw=torch.stack([u, v, -u, -v]),
        site_metal_stats=torch.zeros(1, 4), site_ligand_angle_stats=torch.zeros(1, 8),
    )


def test_two_bond_vectors_reach_classifier_without_explicit_angle_features_and_are_rotation_invariant():
    torch.manual_seed(42)
    model = small_model(site_geometry_features="none")
    assert not model.use_site_angle_features
    angle = torch.tensor(1.1, requires_grad=True)
    data = two_ligand_star(angle)
    logits = model(data)["logits_metal"]
    other = model(two_ligand_star(torch.tensor(2.3)))["logits_metal"]
    assert not torch.allclose(logits, other, atol=1e-7, rtol=0)
    logits[0, 0].backward()
    assert angle.grad is not None and angle.grad.abs().item() > 1e-8
    rz = torch.tensor([[.8, -.6, 0.], [.6, .8, 0.], [0., 0., 1.]])
    rx = torch.tensor([[1., 0., 0.], [0., .6, -.8], [0., .8, .6]])
    rotated = data.clone()
    rotated.edge_vector_raw = data.edge_vector_raw.detach() @ (rz @ rx).T
    torch.testing.assert_close(logits.detach(), model(rotated)["logits_metal"], atol=2e-6, rtol=2e-6)


def test_message_aggregation_exposes_the_norm_of_two_bond_vectors_to_metal_scalar_update():
    layer = SimpleGVPLayer(s_dim=3, v_dim=1, e_dim=1)
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.zero_()
        # The constant gate is 1/2; selecting edge_v with weight2 passes it unchanged.
        layer.message_gvp.vector_linear.weight[0, -1] = 2.
    received = []
    hook = layer.update_gvp.register_forward_pre_hook(lambda _module, inputs: received.append(inputs[1].clone()))
    for angle in (math.pi / 2, math.pi):
        data = two_ligand_star(torch.tensor(angle))
        layer(torch.zeros(3, 3), torch.zeros(3, 1, 3), data.edge_index,
              torch.zeros(4, 1), data.edge_vector_raw.unsqueeze(1))
    hook.remove()
    assert received[0][2, -1].norm().item() == pytest.approx(math.sqrt(2), abs=1e-6)
    assert received[1][2, -1].norm().item() < 1e-6
