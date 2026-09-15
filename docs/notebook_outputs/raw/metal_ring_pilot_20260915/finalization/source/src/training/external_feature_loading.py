from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

from data_structures import (
    EXTERNAL_FEATURE_CUSTOM_CHARGE_DISTANCE_PROXY,
    EXTERNAL_FEATURE_DPKA_TITR,
    EXTERNAL_FEATURE_NAMES,
    EXTERNAL_FEATURE_NAME_ALIASES,
)
from training.esm_feature_loading import ResidueKey


RELEVANT_FEATURE_NAMES = EXTERNAL_FEATURE_NAMES


def default_feature_dict() -> Dict[str, float]:
    features: Dict[str, float] = {}
    for name in RELEVANT_FEATURE_NAMES:
        features[name] = 0.0
        features[f"{name}_missing"] = 1.0
    return features


def ensure_residue_entry(
    residue_features: Dict[ResidueKey, Dict[str, float]],
    key: ResidueKey,
) -> Dict[str, float]:
    if key not in residue_features:
        residue_features[key] = default_feature_dict()
    return residue_features[key]


def set_feature_value(
    residue_features: Dict[ResidueKey, Dict[str, float]],
    key: ResidueKey,
    name: str,
    value: float,
) -> None:
    name = EXTERNAL_FEATURE_NAME_ALIASES.get(name, name)
    entry = ensure_residue_entry(residue_features, key)
    entry[name] = float(value)
    entry[f"{name}_missing"] = 0.0


def iter_structure_dirs(root_dir: str | Path) -> Iterable[Path]:
    root = Path(root_dir)
    for job_dir in sorted(root.glob("job_*")):
        if not job_dir.is_dir():
            continue
        for structure_dir in sorted(job_dir.iterdir()):
            if structure_dir.is_dir():
                yield structure_dir


def structure_dir_to_feature_lookup(structure_dir: str | Path) -> Dict[ResidueKey, Dict[str, float]]:
    structure_path = Path(structure_dir)
    updated_feature_json_path = structure_path / "residue_features.json"

    if updated_feature_json_path.is_file():
        payload = json.loads(updated_feature_json_path.read_text(encoding="utf-8"))
        residue_features: Dict[ResidueKey, Dict[str, float]] = {}
        for residue_payload in payload.get("residues", []):
            key = (
                str(residue_payload["chain_id"]),
                int(residue_payload["resseq"]),
                str(residue_payload.get("icode", "")).strip(),
            )
            entry = default_feature_dict()
            for feature_name, value in residue_payload.get("features", {}).items():
                if feature_name in {"fa_elec", "biotite_propka_electrostatic_proxy"}:
                    raise ValueError(
                        f"{updated_feature_json_path} contains legacy mixed electrostatic feature "
                        f"{feature_name!r}. Regenerate external features so "
                        f"{EXTERNAL_FEATURE_CUSTOM_CHARGE_DISTANCE_PROXY!r} and "
                        f"{EXTERNAL_FEATURE_DPKA_TITR!r} are stored separately."
                    )
                feature_name = EXTERNAL_FEATURE_NAME_ALIASES.get(feature_name, feature_name)
                entry[feature_name] = float(value)
                if f"{feature_name}_missing" in entry:
                    entry[f"{feature_name}_missing"] = 0.0
            residue_features[key] = entry
        if residue_features:
            return residue_features

    raise FileNotFoundError(
        f"Missing updated residue_features.json in {structure_path}. "
        "Generate updated external features with the feature_extraction package."
    )
