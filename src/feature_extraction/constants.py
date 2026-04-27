from __future__ import annotations

from data_structures import (
    EXTERNAL_FEATURE_CUSTOM_CHARGE_DISTANCE_PROXY,
    EXTERNAL_FEATURE_DPKA_TITR,
    EXTERNAL_FEATURE_RESIDUE_SASA,
    SUPPORTED_SITE_METAL_ELEMENTS,
)

FEATURE_NAMES = (
    EXTERNAL_FEATURE_RESIDUE_SASA,
    EXTERNAL_FEATURE_CUSTOM_CHARGE_DISTANCE_PROXY,
    EXTERNAL_FEATURE_DPKA_TITR,
)

# Maximum solvent-accessible surface areas for amino acids.
# Values are from a common Tien et al.-style reference table and are used
# only to build bounded burial proxies from observed SASA.
MAX_RESIDUE_SASA = {
    "ALA": 121.0,
    "ARG": 265.0,
    "ASN": 187.0,
    "ASP": 187.0,
    "CYS": 148.0,
    "GLN": 214.0,
    "GLU": 214.0,
    "GLY": 97.0,
    "HIS": 216.0,
    "ILE": 195.0,
    "LEU": 191.0,
    "LYS": 230.0,
    "MET": 203.0,
    "PHE": 228.0,
    "PRO": 154.0,
    "SER": 143.0,
    "THR": 163.0,
    "TRP": 264.0,
    "TYR": 255.0,
    "VAL": 165.0,
}

# These are coarse electrostatic weights for the lightweight electrostatic
# proxy, not rigorous formal charges at all pH values.
RESIDUE_CHARGE_PROXIES = {
    "ARG": 1.0,
    "ASP": -1.0,
    "GLU": -1.0,
    "HIS": 0.5,
    "LYS": 1.0,
}

# Restrict the metal proxy table to the catalytic site metals supported by the
# training pipeline. Unsupported metals should not influence the electrostatic
# proxy or PROPKA input sanitization.
METAL_CHARGE_PROXIES = {
    "CO": 2.0,
    "CU": 2.0,
    "FE": 2.0,
    "MN": 2.0,
    "NI": 2.0,
    "ZN": 2.0,
}

if set(METAL_CHARGE_PROXIES) != set(SUPPORTED_SITE_METAL_ELEMENTS):
    raise ValueError(
        "Updated feature extraction metal proxy table drifted from SUPPORTED_SITE_METAL_ELEMENTS."
    )
