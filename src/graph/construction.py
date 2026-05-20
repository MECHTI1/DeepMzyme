from __future__ import annotations

import json
from typing import Dict, List

import torch
from torch import Tensor
from torch_geometric.data import Data

from data_structures import (
    AA_ORDER,
    DEFAULT_EDGE_RADIUS,
    DEFAULT_SITE_LIGAND_ANGLE_FEATURE_DIM,
    EDGE_SOURCE_TO_INDEX,
    EDGE_SOURCE_TYPES,
    GRAPH_NODE_TENSOR_FIELDS,
    GRAPH_NODE_MASK_FIELDS,
    GRAPH_SITE_TENSOR_FIELDS,
    GRAPH_SITE_GEOMETRY_TENSOR_FIELDS,
    GRAPH_TARGET_FIELDS,
    INTERACTION_SUMMARIES_OPTIONAL_WITH_RING,
    MISSING_CLASS_LABEL,
    METAL_NODE_MODE_CHOICES,
    NODE_TYPE_GENERIC_METAL,
    NODE_TYPE_RESIDUE,
    PocketRecord,
)
from featurization import (
    MultinuclearSiteHandler,
    compute_site_ligand_angle_stats,
    compute_net_ligand_vector,
    residue_to_stage1_node_features,
    safe_norm,
)
from graph.edge_postprocess import merge_edge_records, stack_edge_features, stack_metal_edge_features
from graph.edge_records import ResidueEdgeRecord, ResidueMetalEdgeRecord
from graph.edge_sources import (
    build_geometric_metal_edge_records,
    build_radius_edge_records_from_residues,
    build_ring_edge_records,
)
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
_GRAPH_RESIDUE_NODE_MASK_FIELD, _GRAPH_METAL_NODE_MASK_FIELD, _GRAPH_NODE_TYPE_ID_FIELD = GRAPH_NODE_MASK_FIELDS
(_GRAPH_SITE_LIGAND_ANGLE_STATS_FIELD,) = GRAPH_SITE_GEOMETRY_TENSOR_FIELDS


class PocketData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "metal_edge_index":
            return torch.tensor(
                [[self.num_nodes], [self.metal_pos.size(0)]],
                dtype=torch.long,
            )
        return super().__inc__(key, value, *args, **kwargs)


def stack_node_features(node_dicts: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    return {
        field_name: torch.stack([node[field_name] for node in node_dicts], dim=0)
        for field_name in GRAPH_NODE_TENSOR_FIELDS
    }


def validate_metal_node_mode(metal_node_mode: str) -> str:
    if metal_node_mode not in METAL_NODE_MODE_CHOICES:
        valid = ", ".join(METAL_NODE_MODE_CHOICES)
        raise ValueError(f"Unsupported metal_node_mode {metal_node_mode!r}. Expected one of: {valid}.")
    return metal_node_mode


def generic_metal_node_features(metal_coord: Tensor, esm_dim: int) -> Dict[str, Tensor]:
    metal_coord = metal_coord.float()
    return {
        "x_esm": torch.zeros(esm_dim, dtype=torch.float32),
        "hydrophobicity_kd": torch.zeros(1, dtype=torch.float32),
        "x_reschem": torch.zeros(len(AA_ORDER) + 5, dtype=torch.float32),
        "x_role": torch.zeros(2, dtype=torch.float32),
        "x_dist_raw": torch.zeros(3, dtype=torch.float32),
        "x_misc": torch.zeros(1, dtype=torch.float32),
        "x_env_burial": torch.zeros(1, dtype=torch.float32),
        "x_env_electrostatics": torch.zeros(2, dtype=torch.float32),
        "x_vec": torch.zeros(2, 3, dtype=torch.float32),
        "donor_coords": torch.zeros(2, 3, dtype=torch.float32),
        "donor_mask": torch.zeros(2, dtype=torch.bool),
        "fg_centroid": metal_coord,
        "pos": metal_coord,
    }


def graph_node_masks(n_residue_nodes: int, n_metal_nodes: int) -> Dict[str, Tensor]:
    residue_mask = torch.cat(
        [
            torch.ones(n_residue_nodes, dtype=torch.bool),
            torch.zeros(n_metal_nodes, dtype=torch.bool),
        ],
        dim=0,
    )
    metal_mask = torch.cat(
        [
            torch.zeros(n_residue_nodes, dtype=torch.bool),
            torch.ones(n_metal_nodes, dtype=torch.bool),
        ],
        dim=0,
    )
    node_type_id = torch.where(
        metal_mask,
        torch.full_like(metal_mask, NODE_TYPE_GENERIC_METAL, dtype=torch.long),
        torch.full_like(metal_mask, NODE_TYPE_RESIDUE, dtype=torch.long),
    )
    return {
        _GRAPH_RESIDUE_NODE_MASK_FIELD: residue_mask,
        _GRAPH_METAL_NODE_MASK_FIELD: metal_mask,
        _GRAPH_NODE_TYPE_ID_FIELD: node_type_id,
    }


def _zero_interaction_type() -> Tensor:
    return torch.zeros(len(INTERACTION_SUMMARIES_OPTIONAL_WITH_RING), dtype=torch.float32)


def _radius_source_type() -> Tensor:
    source_type = torch.zeros(len(EDGE_SOURCE_TYPES), dtype=torch.float32)
    source_type[EDGE_SOURCE_TO_INDEX["radius"]] = 1.0
    return source_type


def _merge_metal_edge_records(
    ring_metal_edge_records: list[ResidueMetalEdgeRecord],
    fallback_metal_edge_records: list[ResidueMetalEdgeRecord],
) -> list[ResidueMetalEdgeRecord]:
    merged: dict[tuple[int, int], ResidueMetalEdgeRecord] = {}
    for record in fallback_metal_edge_records:
        merged[(int(record.residue_idx), int(record.metal_idx))] = record.clone()
    for record in ring_metal_edge_records:
        merged[(int(record.residue_idx), int(record.metal_idx))] = record.clone()
    return [merged[key] for key in sorted(merged)]


def _promote_metal_edges_to_residue_edge_records(
    pocket: PocketRecord,
    metal_edge_records: list[ResidueMetalEdgeRecord],
    *,
    metal_node_offset: int,
) -> list[ResidueEdgeRecord]:
    promoted: list[ResidueEdgeRecord] = []
    metal_coords = MultinuclearSiteHandler.metal_coords_for_pocket(pocket)
    for record in metal_edge_records:
        residue_idx = int(record.residue_idx)
        metal_idx = int(record.metal_idx)
        if not (0 <= residue_idx < len(pocket.residues)) or not (0 <= metal_idx < metal_coords.size(0)):
            continue
        residue = pocket.residues[residue_idx]
        ca = residue.ca()
        if ca is None:
            continue
        ca_to_metal = float(safe_norm(metal_coords[metal_idx].float() - ca.float(), dim=-1).item())
        contact_distance = float(record.dist_raw.view(-1)[0].item())
        promoted.append(
            ResidueEdgeRecord(
                src=residue_idx,
                dst=metal_node_offset + metal_idx,
                dist_raw=torch.tensor([contact_distance, ca_to_metal], dtype=torch.float32),
                seqsep=0.0,
                same_chain=0.0,
                vector_raw=record.vector_raw.float().clone(),
                interaction_type=record.interaction_type.float().clone(),
                source_type=record.source_type.float().clone(),
                geometry_label=record.geometry_label,
            )
        )
    return promoted


def _build_metal_metal_edge_records(pocket: PocketRecord, *, metal_node_offset: int) -> list[ResidueEdgeRecord]:
    metal_coords = MultinuclearSiteHandler.metal_coords_for_pocket(pocket)
    if metal_coords.size(0) < 2:
        return []
    records: list[ResidueEdgeRecord] = []
    for src_idx in range(metal_coords.size(0)):
        for dst_idx in range(src_idx + 1, metal_coords.size(0)):
            vector_raw = (metal_coords[dst_idx].float() - metal_coords[src_idx].float()).float()
            distance = float(safe_norm(vector_raw, dim=-1).item())
            records.append(
                ResidueEdgeRecord(
                    src=metal_node_offset + src_idx,
                    dst=metal_node_offset + dst_idx,
                    dist_raw=torch.tensor([distance, distance], dtype=torch.float32),
                    seqsep=0.0,
                    same_chain=0.0,
                    vector_raw=vector_raw,
                    interaction_type=_zero_interaction_type(),
                    source_type=_radius_source_type(),
                    geometry_label="metal_metal",
                )
            )
    return records


def pocket_to_pyg_data(
    pocket: PocketRecord,
    esm_dim: int,
    edge_radius: float = DEFAULT_EDGE_RADIUS,
    use_ring_edges: bool = False,
    require_ring_edges: bool = False,
    node_feature_set: str = "conservative",
    omit_node_features: tuple[str, ...] | list[str] = (),
    metal_node_mode: str = "none",
) -> Data:
    metal_node_mode = validate_metal_node_mode(metal_node_mode)
    effective_use_ring_edges = bool(use_ring_edges or require_ring_edges)
    shell_roles = compute_shell_roles(pocket, use_ring_edges=effective_use_ring_edges)
    v_net = compute_net_ligand_vector(pocket)
    residue_node_dicts = [
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
    metal_node_dicts = (
        [
            generic_metal_node_features(metal_coord, esm_dim)
            for metal_coord in MultinuclearSiteHandler.metal_coords_for_pocket(pocket)
        ]
        if metal_node_mode == "per_metal"
        else []
    )
    node_features = stack_node_features(residue_node_dicts + metal_node_dicts)

    residue_edge_records = build_radius_edge_records_from_residues(pocket, edge_radius)
    if effective_use_ring_edges:
        ring_residue_edge_records, metal_edge_records = build_ring_edge_records(
            pocket,
            require_ring_edges=require_ring_edges,
        )
    else:
        ring_residue_edge_records, metal_edge_records = [], []
    fallback_metal_edge_records = (
        build_geometric_metal_edge_records(pocket, shell_roles=shell_roles)
        if metal_node_mode == "per_metal"
        else []
    )
    metal_edge_records = _merge_metal_edge_records(metal_edge_records, fallback_metal_edge_records)
    residue_edge_records.extend(ring_residue_edge_records)
    if metal_node_mode == "per_metal":
        metal_node_offset = len(pocket.residues)
        residue_edge_records.extend(
            _promote_metal_edges_to_residue_edge_records(
                pocket,
                metal_edge_records,
                metal_node_offset=metal_node_offset,
            )
        )
        residue_edge_records.extend(_build_metal_metal_edge_records(pocket, metal_node_offset=metal_node_offset))
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
    site_geometry_tensors = {
        _GRAPH_SITE_LIGAND_ANGLE_STATS_FIELD: compute_site_ligand_angle_stats(
            pocket,
            shell_roles=shell_roles,
        ).view(1, DEFAULT_SITE_LIGAND_ANGLE_FEATURE_DIM),
    }

    data = PocketData(
        **node_features,
        **edge_features,
        **metal_edge_features,
        **site_tensors,
        **site_geometry_tensors,
        **graph_node_masks(len(pocket.residues), len(metal_node_dicts)),
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
