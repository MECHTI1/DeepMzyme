from __future__ import annotations

import json
from typing import Dict, List

import torch
from torch import Tensor
from torch_geometric.data import Data

from data_structures import (
    DEFAULT_EDGE_RADIUS,
    GRAPH_NODE_TENSOR_FIELDS,
    GRAPH_SITE_TENSOR_FIELDS,
    GRAPH_TARGET_FIELDS,
    MISSING_CLASS_LABEL,
    PocketRecord,
)
from featurization import (
    MultinuclearSiteHandler,
    compute_net_ligand_vector,
    residue_to_stage1_node_features,
)
from graph.edge_postprocess import merge_edge_records, stack_edge_features, stack_metal_edge_features
from graph.edge_sources import build_radius_edge_records_from_residues, build_ring_edge_records
from graph.shell_roles import compute_shell_roles
from graph.ring_edges import canonical_ring_edges_output_path
from graph.structure_parsing import extract_metal_pockets_from_structure, parse_structure_file

(
    _GRAPH_METAL_POS_FIELD,
    _GRAPH_METAL_CENTER_POS_FIELD,
    _GRAPH_METAL_COUNT_FIELD,
    _GRAPH_IS_MULTINUCLEAR_FIELD,
    _GRAPH_SITE_METAL_STATS_FIELD,
) = GRAPH_SITE_TENSOR_FIELDS
_GRAPH_Y_METAL_FIELD, _GRAPH_Y_EC_FIELD = GRAPH_TARGET_FIELDS
_GRAPH_EC_SAMPLE_WEIGHT_FIELD = "ec_sample_weight"
_GRAPH_EC_GROUP_ID_FIELD = "ec_group_id"


def stack_node_features(node_dicts: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    return {
        field_name: torch.stack([node[field_name] for node in node_dicts], dim=0)
        for field_name in GRAPH_NODE_TENSOR_FIELDS
    }


def pocket_to_pyg_data(
    pocket: PocketRecord,
    esm_dim: int,
    edge_radius: float = DEFAULT_EDGE_RADIUS,
    use_ring_edges: bool = False,
    require_ring_edges: bool = False,
    node_feature_set: str = "conservative",
    omit_node_features: tuple[str, ...] | list[str] = (),
) -> Data:
    effective_use_ring_edges = bool(use_ring_edges or require_ring_edges)
    shell_roles = compute_shell_roles(pocket, use_ring_edges=effective_use_ring_edges)
    v_net = compute_net_ligand_vector(pocket)
    node_features = stack_node_features(
        [
            residue_to_stage1_node_features(
                residue,
                pocket,
                esm_dim,
                v_net,
                node_feature_set=node_feature_set,
                omit_node_features=omit_node_features,
                is_first_shell=is_first_shell,
                is_second_shell=is_second_shell,
            )
            for residue, (is_first_shell, is_second_shell) in zip(pocket.residues, shell_roles)
        ]
    )

    residue_edge_records = build_radius_edge_records_from_residues(pocket, edge_radius)
    if effective_use_ring_edges:
        ring_residue_edge_records, metal_edge_records = build_ring_edge_records(
            pocket,
            require_ring_edges=require_ring_edges,
        )
    else:
        ring_residue_edge_records, metal_edge_records = [], []
    residue_edge_records.extend(ring_residue_edge_records)
    residue_edge_records = merge_edge_records(residue_edge_records)
    if not residue_edge_records:
        raise ValueError(
            f"Pocket {pocket.pocket_id} produced a graph with no edges at edge_radius={edge_radius}. "
            "Increase the radius, inspect the pocket residues, or provide ring interaction edges."
        )
    edge_features = stack_edge_features(residue_edge_records)
    metal_edge_features = stack_metal_edge_features(metal_edge_records)
    site_tensors = dict(
        zip(
            GRAPH_SITE_TENSOR_FIELDS,
            (
                MultinuclearSiteHandler.metal_coords_for_pocket(pocket),
                pocket.metal_coord.unsqueeze(0),
                torch.tensor([pocket.metal_count()], dtype=torch.long),
                torch.tensor([int(pocket.is_multinuclear())], dtype=torch.long),
                MultinuclearSiteHandler.site_metal_stats(pocket).unsqueeze(0),
            ),
        )
    )

    data = Data(
        **node_features,
        **edge_features,
        **metal_edge_features,
        **site_tensors,
    )
    target_values = (
        pocket.y_metal if pocket.y_metal is not None else MISSING_CLASS_LABEL,
        pocket.y_ec if pocket.y_ec is not None else MISSING_CLASS_LABEL,
    )
    for field_name, value in zip(GRAPH_TARGET_FIELDS, target_values):
        setattr(data, field_name, torch.tensor([value], dtype=torch.long))
    setattr(
        data,
        _GRAPH_EC_SAMPLE_WEIGHT_FIELD,
        torch.tensor([float(pocket.metadata.get(_GRAPH_EC_SAMPLE_WEIGHT_FIELD, 1.0))], dtype=torch.float32),
    )
    setattr(
        data,
        _GRAPH_EC_GROUP_ID_FIELD,
        torch.tensor([int(pocket.metadata.get(_GRAPH_EC_GROUP_ID_FIELD, -1))], dtype=torch.long),
    )
    return data


def save_pocket_metadata_json(pocket: PocketRecord, outpath: str) -> None:
    shell_roles = compute_shell_roles(pocket)
    payload = {
        "structure_id": pocket.structure_id,
        "pocket_id": pocket.pocket_id,
        "metal_element": pocket.metal_element,
        "metal_coord": pocket.metal_coord.tolist(),
        "metal_coords": [coord.tolist() for coord in pocket.metal_coords],
        _GRAPH_METAL_COUNT_FIELD: pocket.metal_count(),
        _GRAPH_IS_MULTINUCLEAR_FIELD: pocket.is_multinuclear(),
        _GRAPH_Y_METAL_FIELD: pocket.y_metal,
        _GRAPH_Y_EC_FIELD: pocket.y_ec,
        "residues": [
            {
                "chain_id": residue.chain_id,
                "resseq": residue.resseq,
                "icode": residue.icode,
                "resname": residue.resname,
                "is_first_shell": is_first_shell,
                "is_second_shell": is_second_shell,
                "has_esm_embedding": residue.has_esm_embedding,
                "has_external_features": residue.has_external_features,
                "external_features": residue.external_features,
                "atom_names": sorted(list(residue.atoms.keys())),
            }
            for residue, (is_first_shell, is_second_shell) in zip(pocket.residues, shell_roles)
        ],
    }
    with open(outpath, "w") as handle:
        json.dump(payload, handle, indent=2)
