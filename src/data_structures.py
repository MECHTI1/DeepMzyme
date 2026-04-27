from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

AA_ORDER = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
]
AA_TO_INDEX = {aa: i for i, aa in enumerate(AA_ORDER)}
HYDROPHOBICITY_KD = {
    "ALA": 1.8,
    "ARG": -4.5,
    "ASN": -3.5,
    "ASP": -3.5,
    "CYS": 2.5,
    "GLN": -3.5,
    "GLU": -3.5,
    "GLY": -0.4,
    "HIS": -3.2,
    "ILE": 4.5,
    "LEU": 3.8,
    "LYS": -3.9,
    "MET": 1.9,
    "PHE": 2.8,
    "PRO": -1.6,
    "SER": -0.8,
    "THR": -0.7,
    "TRP": -0.9,
    "TYR": -1.3,
    "VAL": 4.2,
}

NEGATIVE = {"ASP", "GLU"}
POSITIVE = {"ARG", "LYS", "HIS"}
AROMATIC = {"PHE", "TYR", "TRP", "HIS"}
DONOR_CAPABLE = {"ARG", "ASN", "GLN", "HIS", "LYS", "SER", "THR", "TYR", "TRP", "CYS"}
ACCEPTOR_CAPABLE = {"ASP", "GLU", "ASN", "GLN", "HIS", "SER", "THR", "TYR", "CYS"}

BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}

DONOR_ATOMS_BY_RESIDUE = {
    "ASP": ["OD1", "OD2"],
    "GLU": ["OE1", "OE2"],
    "HIS": ["ND1", "NE2"],
    "CYS": ["SG"],
    "TYR": ["OH"],
    "SER": ["OG"],
    "THR": ["OG1"],
    "LYS": ["NZ"],
    "ARG": ["NH1", "NH2"],
    "ASN": ["OD1", "ND2"],
    "GLN": ["OE1", "NE2"],
    "TRP": ["NE1"],
}

# Stricter default for direct metal coordination; broader thresholds can pull in
# weak or merely nearby contacts rather than true first-shell binders.
DEFAULT_FIRST_SHELL_CUTOFF = 2.7
DEFAULT_POCKET_RADIUS = 10.0 #Try also 8-12, or even more
DEFAULT_EDGE_RADIUS = 8.0 #Try also 6-8, or even more
DEFAULT_MULTINUCLEAR_MERGE_DISTANCE = 4.5
GENERIC_METAL_ELEMENT = "METAL"
MISSING_CLASS_LABEL = -1

# Used only to detect the metal-centered sites currently supported by the
# training label policy; the true metal identity should stay out of the model
# inputs when metal type is a prediction target.
SUPPORTED_SITE_METAL_ELEMENTS = {
    "MN", "FE", "CO", "NI", "CU", "ZN",
}

NODE_FEATURES_CONSERVATIVE = [
    "aa_one_hot",
    "hydrophobicity_kd",
    "donor_flag",
    "acceptor_flag",
    "aromatic_flag",
    "acidic_flag",
    "basic_flag",
    "is_first_shell",
    "is_second_shell",
    "ca_to_metal",
    "fg_to_metal",
    "min_donor_to_metal",
    "SASA",
    "fa_elec",
    "v_cb_to_fg",
    "v_res_to_metal",
    "cos_theta_between_vnetligand_to_vrestometal",
]
NODE_FEATURE_SET_CHOICES = ("conservative",)
NODE_FEATURES_BY_SET = {
    "conservative": NODE_FEATURES_CONSERVATIVE,
}

EDGE_FEATURES_RECOMMENDED_RING = [
    "ring_contact_type_one_hot",
    "contact_distance",
    "sequence_separation",
]

INTERACTION_SUMMARIES_OPTIONAL_WITH_RING = [
    "HBOND:MC_MC",
    "HBOND:SC_SC",
    "HBOND:MIXED_MC_SC",
    "VDW:SC_SC",
    "VDW:MC_MC",
    "VDW:MIXED_MC_SC",
    "IONIC:SC_SC",
    "PIPISTACK:SC_SC",
    "PICATION:SC_SC",
    "METAL_ION:SC_LIG",
]
RING_INTERACTION_TYPES = set(INTERACTION_SUMMARIES_OPTIONAL_WITH_RING)
RING_INTERACTION_TO_INDEX = {
    interaction: i for i, interaction in enumerate(INTERACTION_SUMMARIES_OPTIONAL_WITH_RING)
}
RING_INTERACTION_ALIASES = {
    "HBOND:MC_SC": "HBOND:MIXED_MC_SC",
    "HBOND:SC_MC": "HBOND:MIXED_MC_SC",
    "VDW:MC_SC": "VDW:MIXED_MC_SC",
    "VDW:SC_MC": "VDW:MIXED_MC_SC",
}

EDGE_SOURCE_TYPES = ["radius", "ring"]
EDGE_SOURCE_TO_INDEX = {source: i for i, source in enumerate(EDGE_SOURCE_TYPES)}

NORMALIZABLE_FEATURE_NAMES = (
    "hydrophobicity_kd",
    "x_dist_raw",
    "x_misc",
    "x_env_burial",
    "x_env_pka",
    "x_env_conf",
    "x_env_interactions",
    "edge_dist_raw",
    "edge_seqsep",
    "site_metal_stats",
)


@dataclass
class ResidueRecord:
    chain_id: str
    resseq: int
    icode: str
    resname: str
    atoms: Dict[str, Tensor]
    esm_embedding: Optional[Tensor] = None
    has_esm_embedding: bool = False
    is_first_shell: bool = False
    is_second_shell: bool = False
    external_features: Dict[str, float] = field(default_factory=dict)
    has_external_features: bool = False

    def residue_id(self) -> Tuple[str, int, str]:
        return (self.chain_id, self.resseq, self.icode)

    def get_atom(self, name: str) -> Optional[Tensor]:
        return self.atoms.get(name)

    def ca(self) -> Optional[Tensor]:
        return self.get_atom("CA")

    def get_external_feature(self, name: str, default: float = 0.0) -> float:
        return float(self.external_features.get(name, default))


@dataclass
class PocketRecord:
    structure_id: str
    pocket_id: str
    metal_element: str  # Single pocket-level summary; mixed-metal pockets use GENERIC_METAL_ELEMENT, while full observed symbols live in metadata["metal_symbols_observed"].
    metal_coords: List[Tensor]  # Source of truth for metal positions; one tensor for mononuclear pockets, several for multinuclear pockets.
    residues: List[ResidueRecord]
    y_metal: Optional[int] = None
    y_ec: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.metal_coords:
            raise ValueError("PocketRecord requires at least one metal coordinate.")

    @property
    def metal_coord(self) -> Tensor:
        # Expose a single representative site center for code that wants one point,
        # while keeping metal_coords as the underlying source of truth.
        if len(self.metal_coords) == 1:
            return self.metal_coords[0]
        return torch.stack([coord.float() for coord in self.metal_coords], dim=0).mean(dim=0)

    def metal_count(self) -> int:
        return len(self.metal_coords)

    def is_multinuclear(self) -> bool:
        return self.metal_count() > 1
