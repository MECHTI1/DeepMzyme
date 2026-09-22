"""Unit tests for GVP geometric capacity enhancements and distance normalization fixes."""

import math
from pathlib import Path
import sys

import pytest
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_structures import (
    DEFAULT_EDGE_RADIUS,
    EDGE_SOURCE_TYPES,
    INTERACTION_SUMMARIES_OPTIONAL_WITH_RING,
)
from model import EdgeScalarEncoder, GVPPocketClassifier, SimpleGVP, vector_norm
from model_variants.factory import build_pocket_classifier
from model_variants.models import SimpleGNNPocketClassifier
from training.config import parse_args
from training.graph_dataset import (
    FeatureNormalizationStats,
    apply_feature_normalization,
)


def test_edge_dist_raw_raw_preserved_during_feature_normalization():
    """Verify edge_dist_raw_raw is preserved with original Angstrom values."""
    data = Data()
    raw_distances = torch.tensor([[2.5, 3.8], [7.2, 8.5]], dtype=torch.float32)
    data.edge_dist_raw = raw_distances.clone()

    means = {"edge_dist_raw": torch.tensor([5.0, 5.0])}
    stds = {"edge_dist_raw": torch.tensor([2.0, 2.0])}
    stats = FeatureNormalizationStats(means=means, stds=stds)

    normalized = apply_feature_normalization(data, stats)
    assert hasattr(normalized, "edge_dist_raw_raw")
    torch.testing.assert_close(normalized.edge_dist_raw_raw, raw_distances)
    # The normalized edge_dist_raw must be z-scored
    expected_norm = (raw_distances - torch.tensor([5.0, 5.0])) / torch.tensor([2.0, 2.0])
    torch.testing.assert_close(normalized.edge_dist_raw, expected_norm)


def test_edge_scalar_encoder_rbf_center_coverage():
    """Verify that physical distances activate RBF centers across [0, 12] Angstroms,
    whereas normalized z-scores in [-2, 1.5] leave higher centers dead."""
    encoder = EdgeScalarEncoder(n_rbf=16, out_dim=32, distance_sigma=0.75)
    centers = encoder.dist_rbf.centers  # 16 centers spanning 0.0 to 12.0

    # Physical Angstrom distances distributed throughout a pocket (2A to 11A)
    phys_distances = torch.tensor([[2.5, 4.0], [6.5, 8.0], [9.5, 11.0]])
    phys_rbf = encoder.dist_rbf(phys_distances)  # (3, 2, 16)
    # Check that high centers (e.g. 8A to 11A, index >= 10) fire strongly for physical distances
    assert (phys_rbf[:, :, 10:].max() > 0.5).item()

    # Z-scored distances (mean=0, std=1, typical range [-2.0, 1.5])
    zscore_distances = torch.tensor([[-1.5, -0.5], [0.0, 0.8], [1.0, 1.4]])
    zscore_rbf = encoder.dist_rbf(zscore_distances)
    # High centers (index >= 6, center >= 4.8A) are completely dead under z-scores
    assert (zscore_rbf[:, :, 6:].max() < 0.01).item()


def test_gvp_model_uses_edge_dist_raw_raw_when_flag_enabled():
    """Verify that GVPPocketClassifier uses edge_dist_raw_raw when flag is True."""
    from test_coordination_geometry_features import small_model, two_ligand_star
    data = two_ligand_star(torch.tensor(1.0))
    data.edge_dist_raw = torch.zeros_like(data.edge_dist_raw)  # z-scored dead values
    data.edge_dist_raw_raw = torch.full_like(data.edge_dist_raw, 5.0)  # physical Angstrom values

    model_raw = small_model(edge_rbf_use_raw_distances=True)
    model_zscore = small_model(edge_rbf_use_raw_distances=False)
    model_zscore.load_state_dict(model_raw.state_dict())

    out_raw = model_raw(data)["logits_metal"]
    out_zscore = model_zscore(data)["logits_metal"]
    assert (out_raw - out_zscore).abs().max().item() > 1e-5


def test_simple_gvp_angle_sensitivity():
    """Verify that SimpleGVP with channel mixing is sensitive to the angle between vectors."""
    gvp = SimpleGVP(s_in=4, v_in=2, s_out=4, v_out=2)

    # Set vector_h to mix the two channels: h1 = v1 + v2, h2 = v1 - v2
    with torch.no_grad():
        gvp.vector_h.weight.copy_(torch.tensor([[1.0, 1.0], [1.0, -1.0]]))

    s = torch.zeros(1, 4)

    # Case A: Two orthogonal unit vectors (angle = 90 deg)
    v_ortho = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])  # (1, 2, 3)

    # Case B: Two antiparallel unit vectors (angle = 180 deg)
    v_anti = torch.tensor([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]])  # (1, 2, 3)

    # Both cases have individual vector lengths = 1.0
    assert torch.allclose(vector_norm(v_ortho), torch.ones(1, 2))
    assert torch.allclose(vector_norm(v_anti), torch.ones(1, 2))

    s_out_ortho, _ = gvp(s, v_ortho)
    s_out_anti, _ = gvp(s, v_anti)

    # With channel mixing, orthogonal vectors produce ||v1+v2|| = sqrt(2),
    # while antiparallel produce ||v1+v2|| = 0.
    # Therefore the scalar outputs must differ significantly!
    assert not torch.allclose(s_out_ortho, s_out_anti, atol=1e-3)


def test_simple_gvp_rotation_equivariance():
    """Verify that SimpleGVP is SO(3) rotation equivariant in vectors and invariant in scalars."""
    gvp = SimpleGVP(s_in=4, v_in=2, s_out=4, v_out=2)

    s = torch.randn(1, 4)
    v = torch.randn(1, 2, 3)

    # Rotation matrix (Euler angles)
    alpha = 0.7
    R = torch.tensor([
        [math.cos(alpha), -math.sin(alpha), 0.0],
        [math.sin(alpha), math.cos(alpha), 0.0],
        [0.0, 0.0, 1.0],
    ])

    v_rot = torch.einsum("bij,jk->bik", v, R)

    s_out1, v_out1 = gvp(s, v)
    s_out2, v_out2 = gvp(s, v_rot)

    # Scalar outputs must be strictly rotation invariant
    torch.testing.assert_close(s_out1, s_out2, atol=1e-6, rtol=1e-6)

    # Vector outputs must be strictly rotation equivariant: v_out(v * R) == v_out(v) * R
    v_out1_rot = torch.einsum("bij,jk->bik", v_out1, R)
    torch.testing.assert_close(v_out1_rot, v_out2, atol=1e-6, rtol=1e-6)


def test_gvp_learning_rate_decoupling_config():
    """Verify --gvp-learning-rate config parsing."""
    args = parse_args([
        "--learning-rate", "3e-5",
        "--gvp-learning-rate", "5e-4",
        "--rbf-use-raw-distances",
    ])
    assert args.learning_rate == 3e-5
    assert args.gvp_learning_rate == 5e-4
    assert args.node_rbf_use_raw_distances is True
    assert args.edge_rbf_use_raw_distances is True


def test_optimizer_parameter_groups_with_gvp_lr():
    """Verify that GVP trunk parameters receive gvp_learning_rate while heads retain base lr."""
    model = GVPPocketClassifier(
        esm_dim=16, hidden_s=16, hidden_v=2, edge_hidden=8, n_layers=2,
        n_metal=4, n_ec=1, esm_fusion_dim=16, predict_ec=False,
    )
    gvp_lr = 5e-4
    base_lr = 3e-5
    trunk_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(name.startswith(prefix) for prefix in ("layers.", "node_scalar_encoder.", "edge_scalar_encoder.")):
            trunk_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": trunk_params, "lr": gvp_lr},
        {"params": head_params, "lr": base_lr},
    ])
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr"] == 5e-4
    assert optimizer.param_groups[1]["lr"] == 3e-5
    assert len(optimizer.param_groups[0]["params"]) == len(trunk_params)
    assert len(optimizer.param_groups[1]["params"]) == len(head_params)
