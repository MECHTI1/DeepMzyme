from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from data_structures import (
    AA_ORDER,
    AA_TO_INDEX,
    ACCEPTOR_CAPABLE,
    AROMATIC,
    BACKBONE_ATOMS,
    DEFAULT_SITE_LIGAND_ANGLE_FEATURE_DIM,
    DEFAULT_FIRST_SHELL_CUTOFF,
    DONOR_ATOMS_BY_RESIDUE,
    DONOR_CAPABLE,
    EXTERNAL_FEATURE_CUSTOM_CHARGE_DISTANCE_PROXY,
    EXTERNAL_FEATURE_DPKA_TITR,
    EXTERNAL_FEATURE_RESIDUE_SASA,
    HYDROPHOBICITY_KD,
    NODE_FEATURES_BY_SET,
    NEGATIVE,
    POSITIVE,
    PocketRecord,
    ResidueRecord,
    validate_node_feature_omissions,
)

BURIAL_FEATURE_NAMES = (EXTERNAL_FEATURE_RESIDUE_SASA,)
ELECTROSTATIC_FEATURE_NAMES = (
    EXTERNAL_FEATURE_CUSTOM_CHARGE_DISTANCE_PROXY,
    EXTERNAL_FEATURE_DPKA_TITR,
)


@dataclass(frozen=True)
class ResidueMetalLigandGeometry:
    residue_idx: int
    metal_idx: int
    residue_coord: Tensor
    metal_coord: Tensor
    distance: float

    @property
    def metal_to_ligand_vector(self) -> Tensor:
        return (self.residue_coord.float() - self.metal_coord.float()).float()


def safe_norm(x: Tensor, dim: int = -1, keepdim: bool = False, eps: float = 1e-8) -> Tensor:
    return torch.sqrt(torch.clamp((x * x).sum(dim=dim, keepdim=keepdim), min=eps))


def normalize_vec(x: Tensor, dim: int = -1, eps: float = 1e-8) -> Tensor:
    return x / safe_norm(x, dim=dim, keepdim=True, eps=eps)


def pairwise_distances(x: Tensor) -> Tensor:
    diff = x[:, None, :] - x[None, :, :]
    return safe_norm(diff, dim=-1)


def one_hot_index(index: int, size: int) -> Tensor:
    one_hot = torch.zeros(size, dtype=torch.float32)
    if 0 <= index < size:
        one_hot[index] = 1.0
    return one_hot


def residue_one_hot(resname: str) -> Tensor:
    idx = AA_TO_INDEX.get(resname, -1)
    return one_hot_index(idx, len(AA_ORDER))


def residue_hydrophobicity_kd(resname: str) -> Tensor:
    return torch.tensor([HYDROPHOBICITY_KD.get(resname.upper(), 0.0)], dtype=torch.float32)


def residue_chemistry_flags(resname: str) -> Tensor:
    flags = [
        float(resname in DONOR_CAPABLE),
        float(resname in ACCEPTOR_CAPABLE),
        float(resname in AROMATIC),
        float(resname in NEGATIVE),
        float(resname in POSITIVE),
    ]
    return torch.tensor(flags, dtype=torch.float32)


def build_x_reschem(residue: ResidueRecord) -> Tensor:
    return torch.cat(
        [
            residue_one_hot(residue.resname),
            residue_chemistry_flags(residue.resname),
        ],
        dim=-1,
    )


def donor_atom_names(resname: str) -> List[str]:
    return DONOR_ATOMS_BY_RESIDUE.get(resname, [])[:2]


def donor_coords_and_mask(residue: ResidueRecord, max_donors: int = 2) -> Tuple[Tensor, Tensor]:
    coords = torch.zeros(max_donors, 3, dtype=torch.float32)
    mask = torch.zeros(max_donors, dtype=torch.bool)

    names = donor_atom_names(residue.resname)
    for i, atom_name in enumerate(names[:max_donors]):
        atom = residue.get_atom(atom_name)
        if atom is not None:
            coords[i] = atom.float()
            mask[i] = True

    return coords, mask


def ligand_candidate_coords(residue: ResidueRecord) -> Tensor:
    donor_coords, donor_mask = donor_coords_and_mask(residue, max_donors=2)
    if donor_mask.any():
        return donor_coords[donor_mask].float()
    return functional_group_centroid(residue).float().unsqueeze(0)


def residue_metal_ligand_geometries(
    pocket: PocketRecord,
    shell_roles: list[tuple[bool, bool]] | None = None,
    *,
    ligand_cutoff: float = DEFAULT_FIRST_SHELL_CUTOFF,
    ensure_each_metal: bool = True,
) -> list[ResidueMetalLigandGeometry]:
    """Build generic metal-ligand geometry without exposing metal identity."""
    metal_coords = MultinuclearSiteHandler.metal_coords_for_pocket(pocket)
    geometries: list[ResidueMetalLigandGeometry] = []
    seen_pairs: set[tuple[int, int]] = set()

    for residue_idx, residue in enumerate(pocket.residues):
        candidate_coords = ligand_candidate_coords(residue)
        if candidate_coords.numel() == 0:
            continue
        is_first_shell = (
            bool(shell_roles[residue_idx][0])
            if shell_roles is not None and residue_idx < len(shell_roles)
            else bool(residue.is_first_shell)
        )
        distances = safe_norm(candidate_coords[:, None, :] - metal_coords[None, :, :], dim=-1)
        for metal_idx in range(metal_coords.size(0)):
            metal_distances = distances[:, metal_idx]
            closest_idx = int(torch.argmin(metal_distances).item())
            closest_distance = float(metal_distances[closest_idx].item())
            keep = closest_distance <= float(ligand_cutoff)
            if not keep and is_first_shell:
                nearest_metal_idx = int(torch.argmin(distances.min(dim=0).values).item())
                keep = metal_idx == nearest_metal_idx
            if not keep:
                continue
            pair_key = (residue_idx, metal_idx)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            geometries.append(
                ResidueMetalLigandGeometry(
                    residue_idx=residue_idx,
                    metal_idx=metal_idx,
                    residue_coord=candidate_coords[closest_idx].float(),
                    metal_coord=metal_coords[metal_idx].float(),
                    distance=closest_distance,
                )
            )

    if ensure_each_metal and pocket.residues:
        connected_metals = {geometry.metal_idx for geometry in geometries}
        for metal_idx, metal_coord in enumerate(metal_coords):
            if metal_idx in connected_metals:
                continue
            best: tuple[float, int, Tensor] | None = None
            for residue_idx, residue in enumerate(pocket.residues):
                candidate_coords = ligand_candidate_coords(residue)
                distances = safe_norm(candidate_coords - metal_coord.unsqueeze(0), dim=-1)
                closest_idx = int(torch.argmin(distances).item())
                distance = float(distances[closest_idx].item())
                if best is None or distance < best[0]:
                    best = (distance, residue_idx, candidate_coords[closest_idx].float())
            if best is None:
                continue
            distance, residue_idx, residue_coord = best
            pair_key = (residue_idx, metal_idx)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            geometries.append(
                ResidueMetalLigandGeometry(
                    residue_idx=residue_idx,
                    metal_idx=metal_idx,
                    residue_coord=residue_coord,
                    metal_coord=metal_coord.float(),
                    distance=distance,
                )
            )

    return geometries


def compute_site_ligand_angle_stats(
    pocket: PocketRecord,
    shell_roles: list[tuple[bool, bool]] | None = None,
    *,
    ligand_cutoff: float = DEFAULT_FIRST_SHELL_CUTOFF,
) -> Tensor:
    geometries = residue_metal_ligand_geometries(
        pocket,
        shell_roles=shell_roles,
        ligand_cutoff=ligand_cutoff,
        ensure_each_metal=False,
    )
    angles: list[float] = []
    metal_to_vectors: dict[int, list[Tensor]] = {}
    for geometry in geometries:
        metal_to_vectors.setdefault(geometry.metal_idx, []).append(geometry.metal_to_ligand_vector)

    for vectors in metal_to_vectors.values():
        if len(vectors) < 2:
            continue
        for i, vec_i in enumerate(vectors):
            for vec_j in vectors[i + 1 :]:
                denom = safe_norm(vec_i, dim=-1) * safe_norm(vec_j, dim=-1)
                cosine = torch.clamp(torch.dot(vec_i, vec_j) / denom.clamp_min(1e-8), min=-1.0, max=1.0)
                angles.append(float(torch.rad2deg(torch.acos(cosine)).item()))

    ligand_count = float(len(geometries))
    angle_pair_count = float(len(angles))
    if not angles:
        values = [ligand_count, angle_pair_count, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        return torch.tensor(values, dtype=torch.float32)

    angle_tensor = torch.tensor(angles, dtype=torch.float32)
    tetrahedral_target = angle_tensor.new_tensor(109.47)
    octahedral_targets = torch.stack(
        [
            (angle_tensor - 90.0).abs(),
            (angle_tensor - 180.0).abs(),
        ],
        dim=-1,
    )
    values = [
        ligand_count,
        angle_pair_count,
        float(angle_tensor.min().item()),
        float(angle_tensor.mean().item()),
        float(angle_tensor.max().item()),
        float(angle_tensor.std(unbiased=False).item()) if angle_tensor.numel() > 1 else 0.0,
        float((angle_tensor - tetrahedral_target).abs().mean().item()),
        float(octahedral_targets.min(dim=-1).values.mean().item()),
    ]
    if len(values) != DEFAULT_SITE_LIGAND_ANGLE_FEATURE_DIM:
        raise AssertionError("Site ligand angle feature dimension changed without updating the schema.")
    return torch.tensor(values, dtype=torch.float32)


def sidechain_atoms(residue: ResidueRecord) -> List[Tensor]:
    sidechain = []
    for atom_name, coord in residue.atoms.items():
        if atom_name not in BACKBONE_ATOMS:
            sidechain.append(coord.float())
    return sidechain


def centroid(coords: List[Tensor]) -> Optional[Tensor]:
    if len(coords) == 0:
        return None
    return torch.stack(coords, dim=0).mean(dim=0)


def functional_group_centroid(residue: ResidueRecord) -> Tensor:
    donor_coords, donor_mask = donor_coords_and_mask(residue, max_donors=2)
    if donor_mask.any():
        return donor_coords[donor_mask].mean(dim=0)

    sc = sidechain_atoms(residue)
    sc_cent = centroid(sc)
    if sc_cent is not None:
        return sc_cent

    ca = residue.ca()
    if ca is None:
        raise ValueError(f"Residue {residue.residue_id()} has no CA and no usable centroid.")
    return ca.float()


def min_distance_to_point(coords: Tensor, point: Tensor, mask: Optional[Tensor] = None) -> float:
    if coords.numel() == 0:
        return 999.0
    if mask is not None:
        coords = coords[mask]
    if coords.numel() == 0:
        return 999.0
    return float(safe_norm(coords - point.unsqueeze(0), dim=-1).min().item())


def second_min_distance_to_point(coords: Tensor, point: Tensor, mask: Optional[Tensor] = None) -> float:
    if coords.numel() == 0:
        return 999.0
    if mask is not None:
        coords = coords[mask]
    if coords.numel() == 0:
        return 999.0
    d = safe_norm(coords - point.unsqueeze(0), dim=-1)
    vals, _ = torch.sort(d)
    if vals.numel() == 1:
        return float(vals[0].item())
    return float(vals[1].item())


def build_external_feature_vector(rr: ResidueRecord, feature_names: Tuple[str, ...]) -> Tensor:
    return torch.tensor(
        [rr.get_external_feature(name, 0.0) for name in feature_names],
        dtype=torch.float32,
    )


def build_external_feature_groups(rr: ResidueRecord) -> Dict[str, Tensor]:
    return {
        "burial": build_external_feature_vector(rr, BURIAL_FEATURE_NAMES),
        "electrostatics": build_external_feature_vector(rr, ELECTROSTATIC_FEATURE_NAMES),
    }


def apply_node_feature_omissions(
    features: Dict[str, Tensor],
    omitted_features: tuple[str, ...],
) -> Dict[str, Tensor]:
    if not omitted_features:
        return features

    for feature_name in omitted_features:
        if feature_name == "aa_one_hot":
            features["x_reschem"][: len(AA_ORDER)] = 0.0
        elif feature_name == "hydrophobicity_kd":
            features["hydrophobicity_kd"].zero_()
        elif feature_name == "donor_flag":
            features["x_reschem"][len(AA_ORDER) + 0] = 0.0
        elif feature_name == "acceptor_flag":
            features["x_reschem"][len(AA_ORDER) + 1] = 0.0
        elif feature_name == "aromatic_flag":
            features["x_reschem"][len(AA_ORDER) + 2] = 0.0
        elif feature_name == "acidic_flag":
            features["x_reschem"][len(AA_ORDER) + 3] = 0.0
        elif feature_name == "basic_flag":
            features["x_reschem"][len(AA_ORDER) + 4] = 0.0
        elif feature_name == "is_first_shell":
            features["x_role"][0] = 0.0
        elif feature_name == "is_second_shell":
            features["x_role"][1] = 0.0
        elif feature_name == "ca_to_metal":
            features["x_dist_raw"][0] = 0.0
        elif feature_name == "fg_to_metal":
            features["x_dist_raw"][1] = 0.0
        elif feature_name == "min_donor_to_metal":
            features["x_dist_raw"][2] = 0.0
        elif feature_name == EXTERNAL_FEATURE_RESIDUE_SASA:
            features["x_env_burial"][0] = 0.0
        elif feature_name == EXTERNAL_FEATURE_CUSTOM_CHARGE_DISTANCE_PROXY:
            features["x_env_electrostatics"][0] = 0.0
        elif feature_name == EXTERNAL_FEATURE_DPKA_TITR:
            features["x_env_electrostatics"][1] = 0.0
        elif feature_name == "v_cb_to_fg":
            features["x_vec"][0].zero_()
        elif feature_name == "v_res_to_metal":
            features["x_vec"][1].zero_()
        elif feature_name == "cos_theta_between_vnetligand_to_vrestometal":
            features["x_misc"][0] = 0.0
        else:
            raise ValueError(f"Unmapped conservative node feature omission {feature_name!r}.")
    return features


class MultinuclearSiteHandler:
    @staticmethod
    def metal_coords_for_pocket(pocket: PocketRecord) -> Tensor:
        metal_coords = pocket.metal_coords
        return torch.stack([coord.float() for coord in metal_coords], dim=0)

    @staticmethod
    def nearest_metal_for_points(points: Tensor, metal_coords: Tensor) -> Tuple[Tensor, Tensor]:
        diff = points[:, None, :] - metal_coords[None, :, :]
        dists = safe_norm(diff, dim=-1)
        min_dists, metal_idx = torch.min(dists, dim=1)
        nearest = metal_coords[metal_idx]
        return nearest, min_dists

    @staticmethod
    def nearest_metal_for_point(point: Tensor, metal_coords: Tensor) -> Tuple[Tensor, float]:
        nearest, min_dists = MultinuclearSiteHandler.nearest_metal_for_points(
            point.unsqueeze(0),
            metal_coords,
        )
        return nearest[0], float(min_dists[0].item())

    @staticmethod
    def min_distance_to_metals(coords: Tensor, metal_coords: Tensor, mask: Optional[Tensor] = None) -> float:
        if coords.numel() == 0:
            return 999.0
        if mask is not None:
            coords = coords[mask]
        if coords.numel() == 0:
            return 999.0
        _, min_dists = MultinuclearSiteHandler.nearest_metal_for_points(coords, metal_coords)
        return float(min_dists.min().item())

    @staticmethod
    def site_metal_stats(pocket: PocketRecord) -> Tensor:
        # Pocket-level metal-site summary used later in late fusion.
        metal_coords = MultinuclearSiteHandler.metal_coords_for_pocket(pocket)
        metal_count = float(metal_coords.size(0))
        is_multinuclear = float(metal_count > 1.0)

        if metal_coords.size(0) <= 1:
            min_dist = 0.0
            mean_dist = 0.0
        else:
            dmat = pairwise_distances(metal_coords)
            mask = torch.triu(torch.ones_like(dmat, dtype=torch.bool), diagonal=1)
            pair_dists = dmat[mask]
            min_dist = float(pair_dists.min().item())
            mean_dist = float(pair_dists.mean().item())

        return torch.tensor(
            [is_multinuclear, metal_count, min_dist, mean_dist],
            dtype=torch.float32,
        )


def compute_net_ligand_vector(
    pocket: PocketRecord,
    ligand_cutoff: float = DEFAULT_FIRST_SHELL_CUTOFF,
    max_donors_per_residue: int = 2,
) -> Tensor:
    metal_coords = MultinuclearSiteHandler.metal_coords_for_pocket(pocket)
    v_net = torch.zeros(3, dtype=torch.float32)

    for rr in pocket.residues:
        donor_coords, donor_mask = donor_coords_and_mask(rr, max_donors=max_donors_per_residue)
        if not donor_mask.any():
            continue

        coords = donor_coords[donor_mask]
        nearest_metals, min_dists = MultinuclearSiteHandler.nearest_metal_for_points(coords, metal_coords)
        keep = min_dists <= ligand_cutoff
        if keep.any():
            # Sum ligand-to-metal directions over direct binders to get one site-level orientation vector.
            v_net = v_net + (coords[keep] - nearest_metals[keep]).sum(dim=0)

    return v_net


def residue_to_stage1_node_features(
    rr: ResidueRecord,
    pocket: PocketRecord,
    esm_dim: int,
    v_net: Tensor,
    node_feature_set: str = "conservative",
    omit_node_features: tuple[str, ...] | list[str] = (),
    *,
    is_first_shell: bool | None = None,
    is_second_shell: bool | None = None,
) -> Dict[str, Tensor]:
    if node_feature_set not in NODE_FEATURES_BY_SET:
        raise ValueError(
            f"Unsupported node feature set {node_feature_set!r}. "
            f"Expected one of {sorted(NODE_FEATURES_BY_SET)}."
        )
    omitted_features = validate_node_feature_omissions(node_feature_set, omit_node_features)
    esm_embedding = rr.esm_embedding
    if esm_embedding is None:
        esm_embedding = torch.zeros(esm_dim, dtype=torch.float32)

    metal_coords = MultinuclearSiteHandler.metal_coords_for_pocket(pocket)
    ca = rr.ca()
    cb = rr.get_atom("CB")
    if cb is None:
        # Keep feature construction defined for GLY or incomplete residues by
        # collapsing the scaffold vector to zero and anchoring chemistry at CA.
        cb = ca
    fg = functional_group_centroid(rr)
    donor_coords, donor_mask = donor_coords_and_mask(rr, max_donors=2)

    nearest_metal_to_ca, ca_to_metal = MultinuclearSiteHandler.nearest_metal_for_point(ca.float(), metal_coords)
    _, fg_to_metal = MultinuclearSiteHandler.nearest_metal_for_point(fg.float(), metal_coords)
    min_donor_to_metal = MultinuclearSiteHandler.min_distance_to_metals(donor_coords, metal_coords, donor_mask)

    x_role = torch.tensor(
        [
            float(rr.is_first_shell if is_first_shell is None else is_first_shell),
            float(rr.is_second_shell if is_second_shell is None else is_second_shell),
        ],
        dtype=torch.float32,
    )
    x_dist_raw = torch.tensor(
        [ca_to_metal, fg_to_metal, min_donor_to_metal],
        dtype=torch.float32,
    )

    # v_res anchors the residue to its nearest metal; x_misc keeps only the angle proxy retained by the conservative feature set.
    v_res = (ca.float() - nearest_metal_to_ca).float()
    v_net = v_net.float()
    denom = float(safe_norm(v_net, dim=-1).item()) * float(safe_norm(v_res, dim=-1).item()) + 1e-8
    cos_theta = float(torch.clamp(torch.dot(v_net, v_res) / denom, min=-1.0, max=1.0).item())

    x_misc = torch.tensor([cos_theta], dtype=torch.float32)
    env_groups = build_external_feature_groups(rr)
    # Two node vector channels retained by the conservative feature set: sidechain chemistry and residue-to-metal direction.
    x_vec = torch.stack([(fg - cb).float(), v_res], dim=0)

    features = {
        "x_esm": esm_embedding.float(),
        "hydrophobicity_kd": residue_hydrophobicity_kd(rr.resname),
        "x_reschem": build_x_reschem(rr).float(),
        "x_role": x_role,
        "x_dist_raw": x_dist_raw,
        "x_misc": x_misc,
        "x_env_burial": env_groups["burial"],
        "x_env_electrostatics": env_groups["electrostatics"],
        "x_vec": x_vec,
        "donor_coords": donor_coords.float(),
        "donor_mask": donor_mask,
        "fg_centroid": fg.float(),
        "pos": ca.float(),
    }
    return apply_node_feature_omissions(features, omitted_features)
