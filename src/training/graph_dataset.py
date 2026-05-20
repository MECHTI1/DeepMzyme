from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Data

from data_structures import (
    DEFAULT_EDGE_RADIUS,
    EDGE_SOURCE_TO_INDEX,
    GRAPH_EDGE_TENSOR_FIELDS,
    GRAPH_METAL_EDGE_TENSOR_FIELDS,
    GRAPH_NODE_TENSOR_FIELDS,
    GRAPH_SITE_TENSOR_FIELDS,
    NORMALIZABLE_FEATURE_NAMES,
    PocketRecord,
    ResidueRecord,
)
from graph.construction import pocket_to_pyg_data
from graph.shell_roles import compute_shell_roles

_GRAPH_EDGE_INDEX_FIELD = GRAPH_EDGE_TENSOR_FIELDS[0]
_GRAPH_EDGE_SOURCE_TYPE_FIELD = GRAPH_EDGE_TENSOR_FIELDS[-1]
_GRAPH_METAL_EDGE_INDEX_FIELD = GRAPH_METAL_EDGE_TENSOR_FIELDS[0]
_GRAPH_POS_FIELD = GRAPH_NODE_TENSOR_FIELDS[-1]
_GRAPH_METAL_COUNT_FIELD = GRAPH_SITE_TENSOR_FIELDS[2]
_GRAPH_IS_MULTINUCLEAR_FIELD = GRAPH_SITE_TENSOR_FIELDS[3]


@dataclass
class FeatureNormalizationStats:
    means: dict[str, Tensor]
    stds: dict[str, Tensor]
    clamp_value: float = 5.0


def graph_augmentation_enabled(
    *,
    position_noise_std: float = 0.0,
    second_shell_dropout: float = 0.0,
) -> bool:
    return float(position_noise_std) > 0.0 or float(second_shell_dropout) > 0.0


def _clone_residue_with_atoms(residue: ResidueRecord, atoms: dict[str, Tensor]) -> ResidueRecord:
    return ResidueRecord(
        chain_id=residue.chain_id,
        resseq=residue.resseq,
        icode=residue.icode,
        resname=residue.resname,
        atoms=atoms,
        esm_embedding=residue.esm_embedding.clone() if residue.esm_embedding is not None else None,
        has_esm_embedding=residue.has_esm_embedding,
        is_first_shell=residue.is_first_shell,
        is_second_shell=residue.is_second_shell,
        external_features=dict(residue.external_features),
        has_external_features=residue.has_external_features,
    )


def _clone_metadata_with_metal_coords(
    metadata: dict[str, Any],
    metal_coords: list[Tensor],
) -> dict[str, Any]:
    cloned = dict(metadata)
    coord_map = metadata.get("metal_site_coord_map")
    if isinstance(coord_map, dict):
        site_ids = list(metadata.get("metal_site_ids", []))
        updated_coord_map = {}
        for site_key, coord in coord_map.items():
            if site_key in site_ids:
                site_index = int(site_ids.index(site_key))
                if site_index < len(metal_coords):
                    updated_coord_map[site_key] = metal_coords[site_index].clone()
                    continue
            updated_coord_map[site_key] = torch.as_tensor(coord).float().clone()
        cloned["metal_site_coord_map"] = updated_coord_map
    return cloned


def _filter_second_shell_residues(
    pocket: PocketRecord,
    *,
    second_shell_dropout: float,
    use_ring_edges: bool,
) -> list[ResidueRecord]:
    if second_shell_dropout <= 0.0:
        return list(pocket.residues)

    shell_roles = compute_shell_roles(pocket, use_ring_edges=use_ring_edges)
    keep_residues: list[ResidueRecord] = []
    for residue, (is_first_shell, is_second_shell) in zip(pocket.residues, shell_roles):
        if is_second_shell and not is_first_shell and bool(torch.rand(()) < float(second_shell_dropout)):
            continue
        keep_residues.append(residue)
    return keep_residues or list(pocket.residues)


def augment_pocket_for_training(
    pocket: PocketRecord,
    *,
    position_noise_std: float = 0.0,
    second_shell_dropout: float = 0.0,
    use_ring_edges: bool = False,
) -> PocketRecord:
    """Return an in-memory augmented pocket without mutating the loaded records."""
    if not graph_augmentation_enabled(
        position_noise_std=position_noise_std,
        second_shell_dropout=second_shell_dropout,
    ):
        return pocket

    residues = _filter_second_shell_residues(
        pocket,
        second_shell_dropout=float(second_shell_dropout),
        use_ring_edges=use_ring_edges,
    )
    if position_noise_std > 0.0:
        noise_std = float(position_noise_std)
        residues = [
            _clone_residue_with_atoms(
                residue,
                {
                    atom_name: coord.float().clone() + torch.randn_like(coord.float()) * noise_std
                    for atom_name, coord in residue.atoms.items()
                },
            )
            for residue in residues
        ]
        metal_coords = [
            coord.float().clone() + torch.randn_like(coord.float()) * noise_std
            for coord in pocket.metal_coords
        ]
        metadata = _clone_metadata_with_metal_coords(pocket.metadata, metal_coords)
    else:
        metal_coords = [coord.clone() for coord in pocket.metal_coords]
        metadata = _clone_metadata_with_metal_coords(pocket.metadata, metal_coords)

    return PocketRecord(
        structure_id=pocket.structure_id,
        pocket_id=pocket.pocket_id,
        metal_element=pocket.metal_element,
        metal_coords=metal_coords,
        residues=residues,
        y_metal=pocket.y_metal,
        y_ec=pocket.y_ec,
        metadata=metadata,
    )


def build_graph_data_list(
    pockets: list[PocketRecord],
    esm_dim: int,
    edge_radius: float = DEFAULT_EDGE_RADIUS,
    use_ring_edges: bool = False,
    require_ring_edges: bool = False,
    node_feature_set: str = "conservative",
    omit_node_features: tuple[str, ...] | list[str] = (),
    metal_node_mode: str = "none",
) -> list[Data]:
    return [
        pocket_to_pyg_data(
            pocket,
            esm_dim=esm_dim,
            edge_radius=edge_radius,
            use_ring_edges=use_ring_edges,
            require_ring_edges=require_ring_edges,
            node_feature_set=node_feature_set,
            omit_node_features=omit_node_features,
            metal_node_mode=metal_node_mode,
        )
        for pocket in pockets
    ]


def _normalization_tensor_for_feature(data: Data, feature_name: str) -> Tensor | None:
    if not hasattr(data, feature_name):
        return None
    value = getattr(data, feature_name).float()
    if (
        hasattr(data, "residue_node_mask")
        and value.ndim > 0
        and value.size(0) == int(data.residue_node_mask.numel())
        and feature_name
        in {
            "hydrophobicity_kd",
            "x_dist_raw",
            "x_misc",
            "x_env_burial",
            "x_env_electrostatics",
        }
    ):
        return value[data.residue_node_mask.to(dtype=torch.bool, device=value.device)]
    return value


def compute_feature_normalization_stats(
    data_list: list[Data],
    clamp_value: float = 5.0,
) -> FeatureNormalizationStats:
    means: dict[str, Tensor] = {}
    stds: dict[str, Tensor] = {}

    for feature_name in NORMALIZABLE_FEATURE_NAMES:
        tensors = [
            tensor
            for data in data_list
            for tensor in [_normalization_tensor_for_feature(data, feature_name)]
            if tensor is not None and tensor.numel() > 0
        ]
        if not tensors:
            continue
        merged = torch.cat([tensor.float() for tensor in tensors], dim=0)
        mean = merged.mean(dim=0, keepdim=True)
        std = merged.std(dim=0, unbiased=False, keepdim=True)
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        means[feature_name] = mean
        stds[feature_name] = std

    return FeatureNormalizationStats(means=means, stds=stds, clamp_value=clamp_value)


def apply_feature_normalization(data: Data, stats: FeatureNormalizationStats | None) -> Data:
    if stats is None:
        return data

    for feature_name, mean in stats.means.items():
        if not hasattr(data, feature_name):
            continue
        value = getattr(data, feature_name).float()
        if feature_name == "x_dist_raw" and not hasattr(data, "x_dist_raw_raw"):
            setattr(data, "x_dist_raw_raw", value.clone())
        std = stats.stds[feature_name].to(value.device)
        normalized = (value - mean.to(value.device)) / std
        if (
            hasattr(data, "metal_node_mask")
            and normalized.ndim > 0
            and normalized.size(0) == int(data.metal_node_mask.numel())
            and feature_name
            in {
                "hydrophobicity_kd",
                "x_dist_raw",
                "x_misc",
                "x_env_burial",
                "x_env_electrostatics",
            }
        ):
            normalized = normalized.clone()
            normalized[data.metal_node_mask.to(dtype=torch.bool, device=normalized.device)] = 0.0
        setattr(data, feature_name, normalized.clamp(-stats.clamp_value, stats.clamp_value))
    return data


def summarize_graph_dataset(
    pockets: list[PocketRecord],
    esm_dim: int,
    edge_radius: float = DEFAULT_EDGE_RADIUS,
    use_ring_edges: bool = False,
    require_ring_edges: bool = False,
    node_feature_set: str = "conservative",
    omit_node_features: tuple[str, ...] | list[str] = (),
    metal_node_mode: str = "none",
) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    ring_idx = EDGE_SOURCE_TO_INDEX["ring"]

    for pocket in pockets:
        data = pocket_to_pyg_data(
            pocket,
            esm_dim=esm_dim,
            edge_radius=edge_radius,
            use_ring_edges=use_ring_edges,
            require_ring_edges=require_ring_edges,
            node_feature_set=node_feature_set,
            omit_node_features=omit_node_features,
            metal_node_mode=metal_node_mode,
        )
        edge_index = getattr(data, _GRAPH_EDGE_INDEX_FIELD)
        edge_source_type = getattr(data, _GRAPH_EDGE_SOURCE_TYPE_FIELD)
        edge_pairs = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        radius_idx = EDGE_SOURCE_TO_INDEX["radius"]
        ring_mask = edge_source_type[:, ring_idx] > 0.5
        radius_mask = edge_source_type[:, radius_idx] > 0.5
        residue_node_mask = (
            data.residue_node_mask.to(dtype=torch.bool)
            if hasattr(data, "residue_node_mask")
            else torch.ones(getattr(data, _GRAPH_POS_FIELD).size(0), dtype=torch.bool)
        )
        metal_node_mask = (
            data.metal_node_mask.to(dtype=torch.bool)
            if hasattr(data, "metal_node_mask")
            else torch.zeros(getattr(data, _GRAPH_POS_FIELD).size(0), dtype=torch.bool)
        )
        report.append(
            {
                "pocket_id": pocket.pocket_id,
                "metal_count": int(getattr(data, _GRAPH_METAL_COUNT_FIELD).view(-1)[0].item()),
                "is_multinuclear": bool(getattr(data, _GRAPH_IS_MULTINUCLEAR_FIELD).view(-1)[0].item()),
                "n_nodes": int(getattr(data, _GRAPH_POS_FIELD).size(0)),
                "n_residues": int(residue_node_mask.sum().item()),
                "n_metal_nodes": int(metal_node_mask.sum().item()),
                "n_edges": int(edge_index.size(1)),
                "n_metal_edges": int(getattr(data, _GRAPH_METAL_EDGE_INDEX_FIELD).size(1)) if hasattr(data, _GRAPH_METAL_EDGE_INDEX_FIELD) else 0,
                "n_radius_edges": int(radius_mask.sum().item()),
                "n_ring_edges": int(ring_mask.sum().item()),
                "n_duplicate_pairs": len(edge_pairs) - len(set(edge_pairs)),
            }
        )
    return report


class PocketGraphDataset(Dataset):
    def __init__(
        self,
        pockets: list[PocketRecord],
        esm_dim: int,
        edge_radius: float = DEFAULT_EDGE_RADIUS,
        normalization_stats: FeatureNormalizationStats | None = None,
        use_ring_edges: bool = False,
        require_ring_edges: bool = False,
        precomputed_data: list[Data] | None = None,
        node_feature_set: str = "conservative",
        omit_node_features: tuple[str, ...] | list[str] = (),
        position_noise_std: float = 0.0,
        second_shell_dropout: float = 0.0,
        metal_node_mode: str = "none",
    ):
        self.pockets = pockets
        self.esm_dim = esm_dim
        self.edge_radius = edge_radius
        self.normalization_stats = normalization_stats
        self.use_ring_edges = use_ring_edges
        self.require_ring_edges = require_ring_edges
        self.node_feature_set = node_feature_set
        self.omit_node_features = tuple(omit_node_features)
        self.metal_node_mode = str(metal_node_mode)
        self.position_noise_std = float(position_noise_std)
        self.second_shell_dropout = float(second_shell_dropout)
        if precomputed_data is not None and len(precomputed_data) != len(pockets):
            raise ValueError("precomputed_data length must match pockets length.")
        self.precomputed_data = precomputed_data

    @classmethod
    def fit_normalization_stats(
        cls,
        pockets: list[PocketRecord],
        esm_dim: int,
        edge_radius: float = DEFAULT_EDGE_RADIUS,
        clamp_value: float = 5.0,
        use_ring_edges: bool = False,
        require_ring_edges: bool = False,
        precomputed_data: list[Data] | None = None,
        node_feature_set: str = "conservative",
        omit_node_features: tuple[str, ...] | list[str] = (),
        metal_node_mode: str = "none",
    ) -> FeatureNormalizationStats:
        data_list = precomputed_data
        if data_list is None:
            data_list = build_graph_data_list(
                pockets,
                esm_dim=esm_dim,
                edge_radius=edge_radius,
                use_ring_edges=use_ring_edges,
                require_ring_edges=require_ring_edges,
                node_feature_set=node_feature_set,
                omit_node_features=omit_node_features,
                metal_node_mode=metal_node_mode,
            )
        return compute_feature_normalization_stats(data_list, clamp_value=clamp_value)

    def __len__(self) -> int:
        return len(self.pockets)

    def __getitem__(self, idx: int) -> Data:
        if graph_augmentation_enabled(
            position_noise_std=self.position_noise_std,
            second_shell_dropout=self.second_shell_dropout,
        ):
            data = pocket_to_pyg_data(
                augment_pocket_for_training(
                    self.pockets[idx],
                    position_noise_std=self.position_noise_std,
                    second_shell_dropout=self.second_shell_dropout,
                    use_ring_edges=self.use_ring_edges or self.require_ring_edges,
                ),
                esm_dim=self.esm_dim,
                edge_radius=self.edge_radius,
                use_ring_edges=self.use_ring_edges,
                require_ring_edges=self.require_ring_edges,
                node_feature_set=self.node_feature_set,
                omit_node_features=self.omit_node_features,
                metal_node_mode=self.metal_node_mode,
            )
        elif self.precomputed_data is not None:
            data = self.precomputed_data[idx].clone()
        else:
            data = pocket_to_pyg_data(
                self.pockets[idx],
                esm_dim=self.esm_dim,
                edge_radius=self.edge_radius,
                use_ring_edges=self.use_ring_edges,
                require_ring_edges=self.require_ring_edges,
                node_feature_set=self.node_feature_set,
                omit_node_features=self.omit_node_features,
                metal_node_mode=self.metal_node_mode,
            )
        return apply_feature_normalization(data, self.normalization_stats)
