"""Amendment 2, P14: raw-distance RBFs for the GVP families and the exact-zero mask of the 999 A sentinel (synthetic)."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from data_structures import NORMALIZABLE_FEATURE_NAMES  # noqa: E402
from featurization import MultinuclearSiteHandler  # noqa: E402
from model import EdgeScalarEncoder, NodeScalarEncoder, RBFExpansion  # noqa: E402
from test_coordination_geometry_features import small_model, two_ligand_star  # noqa: E402
from training.graph_dataset import apply_feature_normalization, compute_feature_normalization_stats  # noqa: E402

SENTINEL = 999.0
FAR = torch.cat([torch.arange(20.0, 200.0, 0.25), torch.tensor([250.0, 500.0, SENTINEL])])


def _campaign_rbfs() -> list[RBFExpansion]:
    """The node and edge RBF modules exactly as the GVP families build them (16 centres 0-12 A, sigma 0.75)."""
    return [NodeScalarEncoder(n_rbf=16, out_dim=8, distance_sigma=0.75).dist_rbf,
            EdgeScalarEncoder(n_rbf=16, out_dim=8, distance_sigma=0.75).dist_rbf]


def test_the_featurizer_keeps_the_999_sentinel():
    assert MultinuclearSiteHandler.min_distance_to_metals(torch.zeros(0, 3), torch.zeros(1, 3)) == SENTINEL
    assert {"x_dist_raw", "edge_dist_raw"} <= set(NORMALIZABLE_FEATURE_NAMES)


@pytest.mark.parametrize("rbf", _campaign_rbfs(), ids=["node", "edge"])
def test_every_distance_of_20_a_or_more_gives_an_exactly_zero_rbf_row(rbf):
    assert rbf.centers.dtype == torch.float32 and float(rbf.centers.max()) == 12.0
    values = rbf(FAR)
    assert values.dtype == torch.float32 and values.shape == (FAR.numel(), 16)
    assert torch.count_nonzero(values) == 0
    with torch.autocast("cpu", dtype=torch.bfloat16):
        autocast_values = rbf(FAR)
    assert torch.count_nonzero(autocast_values) == 0
    assert torch.count_nonzero(rbf(FAR.to(torch.bfloat16).float())) == 0
    # Real pocket distances keep a non-zero expansion.
    near = rbf(torch.tensor([0.0, 3.5, 7.0, 11.9, 15.0]))
    assert bool((near.sum(dim=-1) > 0).all())


def _sentinel_graph() -> Data:
    data = two_ligand_star(torch.tensor(1.1))
    raw = torch.tensor([[3.8, 2.9, SENTINEL], [5.1, 4.4, 2.1], [0.0, 0.0, 0.0]])
    data.x_dist_raw = raw.clone()
    data.edge_dist_raw = torch.tensor([[2.2, 3.0], [2.4, 3.1], [2.2, 3.0], [2.4, 3.1]])
    stats = compute_feature_normalization_stats([data.clone()])
    return apply_feature_normalization(data, stats)


def test_normalisation_keeps_the_raw_copies_the_rbfs_read():
    data = _sentinel_graph()
    assert hasattr(data, "x_dist_raw_raw") and hasattr(data, "edge_dist_raw_raw")
    assert float(data.x_dist_raw_raw[0, 2]) == SENTINEL
    assert float(data.x_dist_raw.abs().max()) <= 5.0  # the z-scored copy is clamped; the raw copy is not


def test_the_raw_flag_feeds_raw_tensors_to_both_encoders():
    data = _sentinel_graph()
    seen: dict[str, torch.Tensor] = {}
    for flag in (True, False):
        torch.manual_seed(5)
        model = small_model(node_rbf_use_raw_distances=flag, edge_rbf_use_raw_distances=flag)
        model.node_scalar_encoder.register_forward_pre_hook(
            lambda _module, args, flag=flag: seen.__setitem__(f"node{flag}", args[3].detach().clone()))
        model.edge_scalar_encoder.register_forward_pre_hook(
            lambda _module, args, flag=flag: seen.__setitem__(f"edge{flag}", args[0].detach().clone()))
        model(data)
    assert torch.equal(seen["nodeTrue"], data.x_dist_raw_raw) and torch.equal(seen["edgeTrue"], data.edge_dist_raw_raw)
    assert torch.equal(seen["nodeFalse"], data.x_dist_raw) and torch.equal(seen["edgeFalse"], data.edge_dist_raw)
    # The sentinel row reaches the node encoder as an all-zero RBF block.
    model = small_model(node_rbf_use_raw_distances=True, edge_rbf_use_raw_distances=True)
    assert torch.count_nonzero(model.node_scalar_encoder.dist_rbf(seen["nodeTrue"])[0, 2]) == 0


def test_a_gvp_fit_step_with_sentinel_rows_has_finite_loss_and_gradients():
    torch.manual_seed(11)
    model = small_model(node_rbf_use_raw_distances=True, edge_rbf_use_raw_distances=True).train()
    data = _sentinel_graph()
    logits = model(data)["logits_metal"]
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1]))
    loss.backward()
    assert torch.isfinite(loss)
    grads = [param.grad for param in model.parameters() if param.grad is not None]
    assert grads and all(bool(torch.isfinite(grad).all()) for grad in grads)
