"""Synthetic scientific invariants for the bounded metal architecture pilot."""

from pathlib import Path
import sys

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_structures import PocketRecord, ResidueRecord
from graph.construction import pocket_to_pyg_data
from label_schemes import (
    map_site_metal_symbols_for_scheme,
    metal_labels_for_scheme,
)
from metal_objectives import collapse_metal_logits_to_4, collapse_metal_targets_to_4
from model import AttentionPool, metal_distance_pool_mask, pool_graph_states
from model_variants import build_pocket_classifier
from training.splits import split_pockets, split_pockets_k_fold
from training.structure_loading import pocket_has_required_supervision


SCHEMES = ("four_class", "five_class", "six_class")
SYMBOLS = ("MN", "CU", "ZN", "FE", "CO", "NI")


def labelled_pocket(index, symbols, scheme):
    return PocketRecord(
        structure_id=f"p{index:03d}__chain_A__EC_1.1.1.1",
        pocket_id=f"p{index:03d}_site0",
        metal_element=symbols[0],
        metal_coords=[torch.zeros(3)],
        residues=[],
        y_metal=map_site_metal_symbols_for_scheme(symbols, scheme_name=scheme),
        metadata={"matched_summary_site_metal_types": list(symbols)},
    )


@pytest.mark.parametrize("fold", [None, 0, 1, 2, 3, 4])
def test_three_target_schemes_retain_identical_examples_and_grouped_splits(fold):
    splits = []
    for scheme in SCHEMES:
        pockets = [labelled_pocket(i, (SYMBOLS[i % 6],), scheme) for i in range(90)]
        pockets += [labelled_pocket(90, ("CO", "NI"), scheme),
                    labelled_pocket(91, ("FE", "CO"), scheme)]
        eligible = [p for p in pockets if pocket_has_required_supervision(
            p, required_targets=("metal",), metal_eligibility_scheme="six_class"
        )]
        assert len(eligible) == 90
        kwargs = dict(split_by="pdbid", seed=42, task="metal", stratify_by="metal_site")
        split = (split_pockets(eligible, val_fraction=0.15, **kwargs) if fold is None else
                 split_pockets_k_fold(eligible, n_folds=5, fold_index=fold, **kwargs))
        identity = tuple(tuple(p.pocket_id for p in getattr(split, part))
                         for part in ("train_pockets", "val_pockets"))
        assert not (set(identity[0]) & set(identity[1]))
        assert {p.y_metal for p in split.train_pockets + split.val_pockets} == set(
            range(len(metal_labels_for_scheme(scheme)))
        )
        splits.append(identity)
    assert splits[0] == splits[1] == splits[2]


def test_five_class_keeps_fe_distinct_but_mixed_co_ni_is_not_shared_eligible():
    assert map_site_metal_symbols_for_scheme("FE", scheme_name="five_class") == 3
    assert map_site_metal_symbols_for_scheme("CO", scheme_name="five_class") == 4
    assert map_site_metal_symbols_for_scheme("NI", scheme_name="five_class") == 4
    pocket = labelled_pocket(0, ("CO", "NI"), "five_class")
    assert pocket_has_required_supervision(pocket, ("metal",), "active")
    assert not pocket_has_required_supervision(pocket, ("metal",), "six_class")


@pytest.mark.parametrize("scheme,probabilities", [
    ("four_class", [0.35, 0.05, 0.10, 0.50]),
    ("five_class", [0.35, 0.05, 0.10, 0.25, 0.25]),
    ("six_class", [0.35, 0.05, 0.10, 0.20, 0.15, 0.15]),
])
def test_collapsed_predictions_sum_probability_mass_before_argmax(scheme, probabilities):
    label_map = metal_labels_for_scheme(scheme)
    logits = torch.tensor([probabilities], dtype=torch.float64).log()
    collapsed = collapse_metal_logits_to_4(logits, label_map=label_map).softmax(-1)
    torch.testing.assert_close(collapsed, torch.tensor([[0.35, 0.05, 0.10, 0.50]], dtype=torch.float64))
    assert collapsed.argmax(-1).item() == 3
    if scheme != "four_class":
        # Native argmax selects Mn; remapping that discrete prediction is wrong.
        native_prediction = logits.argmax(-1)
        assert collapse_metal_targets_to_4(native_prediction, label_map=label_map).item() == 0
    targets = torch.arange(len(label_map))
    assert collapse_metal_targets_to_4(targets, label_map=label_map).tolist() == [
        0, 1, 2, *([3] * (len(label_map) - 3))
    ]


def test_zero_pooling_cutoff_reads_every_residue_even_beyond_edge_radius():
    data = Data(batch=torch.tensor([0, 0, 1, 1]),
                x_dist_raw_raw=torch.tensor([[2., 1.], [9., 2.], [3., 1.], [10., 2.]]))
    states = torch.tensor([[1., 2.], [9., 10.], [3., 4.], [7., 8.]])
    mask = metal_distance_pool_mask(data, 0.0, base_mask=torch.ones(4, dtype=torch.bool))
    assert mask.tolist() == [True, True, True, True]
    attention = AttentionPool(2)
    for parameter in attention.parameters():
        torch.nn.init.zeros_(parameter)
    result = pool_graph_states(states, data.batch, attention, mask)
    torch.testing.assert_close(result, torch.tensor([[5., 6., 5., 6.], [5., 6., 5., 6.]]))


def test_positive_pooling_cutoff_uses_raw_ca_distance_and_preserves_graph_nodes():
    data = Data(
        batch=torch.tensor([0, 0, 0, 0]), num_nodes=4,
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        # Scaled values must not replace Angstrom distances for the cutoff.
        x_dist_raw=torch.zeros(4, 2),
        # The second residue has a nearby functional group but a distant CA.
        x_dist_raw_raw=torch.tensor([[2., 1.], [8., 1.], [6., 3.], [0., 0.]]),
    )
    original_edges = data.edge_index.clone()
    mask = metal_distance_pool_mask(data, 6.0, base_mask=torch.tensor([True, True, True, False]))
    assert mask.tolist() == [True, False, True, False]
    assert data.num_nodes == 4
    torch.testing.assert_close(data.edge_index, original_edges)


@pytest.mark.parametrize("architecture,fusion,expected_paths", [
    ("only_gvp", "late_fusion", set()),
    ("gvp", "early_fusion", {"early"}),
    ("gvp", "late_fusion", {"late"}),
    ("gvp", "hybrid", {"early", "late"}),
])
def test_fusion_variants_execute_the_expected_esm_paths(architecture, fusion, expected_paths):
    torch.manual_seed(42)
    pocket = labelled_pocket(0, ("MN",), "four_class")
    pocket.residues = [ResidueRecord(
        chain_id="A", resseq=i + 1, icode="", resname="CYS",
        atoms={"CA": torch.tensor([float(i + 1), 0., 0.]),
               "CB": torch.tensor([float(i + 1), 1., 0.]),
               "SG": torch.tensor([float(i + 1), 1.5, 0.])},
    ) for i in range(3)]
    graph = pocket_to_pyg_data(pocket, esm_dim=4, metal_node_mode="none")
    graph.x_esm = torch.randn(3, 4)
    batch = next(iter(DataLoader([graph], batch_size=1)))
    model = build_pocket_classifier(
        model_architecture=architecture, fusion_mode=fusion, esm_dim=4,
        hidden_s=8, hidden_v=2, edge_hidden=8, n_layers=1,
        n_metal=4, n_ec=1, predict_metal=True, predict_ec=False,
        early_esm_dim=2, esm_fusion_dim=4, head_mlp_dropout=0.0,
    )
    seen = set()
    handles = [model.esm_graph_encoder.register_forward_hook(lambda *args: seen.add("late"))]
    if model.early_esm_proj is not None:
        handles.append(model.early_esm_proj.register_forward_hook(lambda *args: seen.add("early")))
    try:
        outputs = model(batch)
        outputs["loss"].backward()
    finally:
        for handle in handles:
            handle.remove()
    assert seen == expected_paths
    assert outputs["logits_metal"].shape == (1, 4)
    assert torch.isfinite(outputs["loss"])
    for path, module in (("early", model.early_esm_proj), ("late", model.esm_graph_encoder)):
        if path in expected_paths:
            assert any(p.grad is not None and bool((p.grad != 0).any()) for p in module.parameters())
