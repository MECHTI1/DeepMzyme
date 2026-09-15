"""Keep node-role features independent of the RING edge ablation when requested."""

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_structures import EDGE_SOURCE_TO_INDEX, GRAPH_NODE_TENSOR_FIELDS, PocketRecord, ResidueRecord
from graph.construction import pocket_to_pyg_data
from graph.shell_roles import compute_shell_roles
from training.config import TrainConfig, config_to_payload, parse_args
from training import graph_dataset, preflight
from training.graph_dataset import (
    PocketGraphDataset,
    apply_feature_normalization,
    augment_pocket_for_training,
    build_graph_data_list,
    compute_feature_normalization_stats,
    summarize_graph_dataset,
)


@pytest.fixture
def pocket(tmp_path):
    # The artificial RING contact intentionally disagrees with centroid shell
    # membership, so accidentally following the edge mode cannot pass the test.
    ring = tmp_path / "fixture_ringEdges"
    ring.write_text(
        "NodeId1\tNodeId2\tInteraction\tAtom1\tAtom2\n"
        "A:1:_:ASP\tA:3:_:ASP\tHBOND:SC_SC\tOD1\tOD1\n"
    )
    residues = []
    for index, x in enumerate([1.5, 4.5, 8.5], start=1):
        residues.append(ResidueRecord(
            chain_id="A", resseq=index, icode="", resname="ASP",
            atoms={name: torch.tensor([x, y, z]) for name, y, z in [
                ("CA", 0., 0.), ("CB", 0., .8), ("CG", 0., .3),
                ("OD1", 0., 0.), ("OD2", .4, 0.),
            ]},
            esm_embedding=torch.full((4,), float(index)),
        ))
    return PocketRecord(
        structure_id="fixture", pocket_id="fixture_metal_0", metal_element="ZN",
        metal_coords=[torch.zeros(3)], residues=residues, y_metal=2,
        metadata={"ring_edges_path": str(ring)},
    )


GEOMETRIC_ROLES = [[1., 0.], [0., 1.], [0., 0.]]
RING_ROLES = [[1., 0.], [0., 0.], [0., 1.]]


def assert_node_inputs_equal(left, right):
    for name in GRAPH_NODE_TENSOR_FIELDS:
        torch.testing.assert_close(getattr(left, name), getattr(right, name), rtol=0, atol=0)


@pytest.mark.parametrize("use_ring", [False, True])
def test_default_preserves_legacy_ring_dependent_roles_and_all_graph_tensors(pocket, use_ring):
    default = pocket_to_pyg_data(pocket, esm_dim=4, edge_radius=6, use_ring_edges=use_ring)
    explicit = pocket_to_pyg_data(
        pocket, esm_dim=4, edge_radius=6, use_ring_edges=use_ring, shell_role_source="edge_mode",
    )
    torch.testing.assert_close(default.x_role, torch.tensor(RING_ROLES if use_ring else GEOMETRIC_ROLES))
    assert default.keys() == explicit.keys()
    for key, value in default:
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, explicit[key], rtol=0, atol=0)
        else:
            assert value == explicit[key]


def test_geometry_source_holds_every_node_input_fixed_while_ring_adds_edges(pocket):
    options = dict(esm_dim=4, edge_radius=6, shell_role_source="geometry")
    off = pocket_to_pyg_data(pocket, **options)
    # require_ring_edges also activates RING; it must not change the role source.
    on = pocket_to_pyg_data(pocket, require_ring_edges=True, **options)
    assert_node_inputs_equal(off, on)
    torch.testing.assert_close(on.x_role, torch.tensor(GEOMETRIC_ROLES))
    assert off.num_nodes == on.num_nodes == 3
    assert off.edge_index.size(1) == 4 and on.edge_index.size(1) == 6
    ring_idx = EDGE_SOURCE_TO_INDEX["ring"]
    assert on.edge_source_type[:, ring_idx].sum().item() == 2
    assert off.edge_source_type[:, ring_idx].sum().item() == 0
    torch.testing.assert_close(off.site_metal_stats, on.site_metal_stats, rtol=0, atol=0)
    torch.testing.assert_close(off.site_ligand_angle_stats, on.site_ligand_angle_stats, rtol=0, atol=0)


def test_train_fitted_normalization_keeps_nodes_equal_but_records_changed_edge_distribution(pocket):
    graphs = [build_graph_data_list(
        [pocket], esm_dim=4, edge_radius=6, use_ring_edges=ring, shell_role_source="geometry",
    )[0] for ring in [False, True]]
    stats = [compute_feature_normalization_stats([graph]) for graph in graphs]
    assert not torch.equal(stats[0].means["edge_dist_raw"], stats[1].means["edge_dist_raw"])
    normalized = [apply_feature_normalization(graph.clone(), fitted) for graph, fitted in zip(graphs, stats)]
    assert_node_inputs_equal(*normalized)
    for graph in normalized:
        assert torch.isfinite(graph.edge_dist_raw).all()


@pytest.mark.parametrize("precomputed", [False, True])
def test_dataset_loading_preserves_geometry_roles(pocket, precomputed):
    options = dict(esm_dim=4, edge_radius=6, use_ring_edges=True, shell_role_source="geometry")
    data = build_graph_data_list([pocket], **options) if precomputed else None
    actual = PocketGraphDataset([pocket], precomputed_data=data, **options)[0]
    torch.testing.assert_close(actual.x_role, torch.tensor(GEOMETRIC_ROLES))
    assert actual.edge_index.size(1) == 6
    summary = summarize_graph_dataset([pocket], **options)[0]
    assert summary["n_nodes"] == 3 and summary["n_ring_edges"] == 2


def test_shell_dropout_and_augmented_graphs_use_the_same_explicit_role_source(pocket):
    for ring in [False, True]:
        augmented = augment_pocket_for_training(
            pocket, second_shell_dropout=1, use_ring_edges=ring, shell_role_source="geometry",
        )
        assert [residue.resseq for residue in augmented.residues] == [1, 3]
        data = PocketGraphDataset(
            [pocket], esm_dim=4, edge_radius=10, use_ring_edges=ring,
            second_shell_dropout=1, shell_role_source="geometry",
        )[0]
        assert data.num_nodes == 2
        torch.testing.assert_close(data.pos[:, 0], torch.tensor([1.5, 8.5]))
        torch.testing.assert_close(data.x_role, torch.tensor([[1., 0.], [0., 0.]]))
    legacy = augment_pocket_for_training(pocket, second_shell_dropout=1, use_ring_edges=True)
    assert [residue.resseq for residue in legacy.residues] == [1, 2]
    assert [residue.resseq for residue in pocket.residues] == [1, 2, 3]


def test_normalization_preparation_and_preflight_receive_geometry_node_roles(pocket, monkeypatch):
    original_compute = graph_dataset.compute_feature_normalization_stats
    seen = []

    def inspect_normalization(data, **kwargs):
        seen.append(data[0].x_role.clone())
        return original_compute(data, **kwargs)

    monkeypatch.setattr(graph_dataset, "compute_feature_normalization_stats", inspect_normalization)
    PocketGraphDataset.fit_normalization_stats(
        [pocket], esm_dim=4, edge_radius=6, use_ring_edges=True, shell_role_source="geometry",
    )
    original_build = preflight.pocket_to_pyg_data

    def inspect_preflight(*args, **kwargs):
        graph = original_build(*args, **kwargs)
        seen.append(graph.x_role.clone())
        return graph

    monkeypatch.setattr(preflight, "pocket_to_pyg_data", inspect_preflight)
    config = TrainConfig(esm_dim=4, edge_radius=6, use_ring_edges=True, shell_role_source="geometry")
    preflight.validate_graphs([pocket], config)
    assert len(seen) == 2
    for roles in seen:
        torch.testing.assert_close(roles, torch.tensor(GEOMETRIC_ROLES))


def test_cli_records_source_and_rejects_unknown_modes(pocket):
    assert TrainConfig().shell_role_source == "edge_mode"
    assert parse_args([]).shell_role_source == "edge_mode"
    config = parse_args(["--shell-role-source", "geometry"])
    assert config_to_payload(config)["shell_role_source"] == "geometry"
    with pytest.raises(SystemExit):
        parse_args(["--shell-role-source", "ring_only"])
    with pytest.raises(ValueError, match="shell_role_source"):
        compute_shell_roles(pocket, shell_role_source="ring_only")
    with pytest.raises(ValueError, match="shell_role_source"):
        PocketGraphDataset([pocket], esm_dim=4, shell_role_source="ring_only")
