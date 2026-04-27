from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import biotite.structure as struc
import numpy as np
from biotite.structure.io import load_structure

from data_structures import (
    EXTERNAL_FEATURE_CUSTOM_CHARGE_DISTANCE_PROXY,
    EXTERNAL_FEATURE_DPKA_TITR,
    EXTERNAL_FEATURE_RESIDUE_SASA,
)
from .constants import (
    FEATURE_NAMES,
    MAX_RESIDUE_SASA,
    METAL_CHARGE_PROXIES,
    RESIDUE_CHARGE_PROXIES,
)
from .propka_support import PropkaRunResult, run_propka_for_structure

ResidueKey = tuple[str, int, str]


@dataclass(frozen=True)
class ResidueGeometry:
    key: ResidueKey
    resname: str
    chain_id: str
    resseq: int
    icode: str
    heavy_coords: np.ndarray
    ca_coord: np.ndarray


def default_feature_dict() -> dict[str, float]:
    features: dict[str, float] = {}
    for name in FEATURE_NAMES:
        features[name] = 0.0
        features[f"{name}_missing"] = 1.0
    return features


def _set_feature(entry: dict[str, float], name: str, value: float) -> None:
    entry[name] = float(value)
    missing_name = f"{name}_missing"
    if missing_name in entry:
        entry[missing_name] = 0.0


def _residue_identifier(residue) -> ResidueKey:
    return (
        str(residue.chain_id[0]),
        int(residue.res_id[0]),
        str(residue.ins_code[0]).strip(),
    )


def _residue_geometry_from_atom_array(atom_array) -> list[ResidueGeometry]:
    residues: list[ResidueGeometry] = []
    for residue in struc.residue_iter(atom_array):
        resname = str(residue.res_name[0]).strip()
        coords = residue.coord.astype(np.float64, copy=False)
        heavy_mask = residue.element != "H"
        heavy_coords = coords[heavy_mask]
        if heavy_coords.size == 0:
            continue
        ca_indices = np.where(residue.atom_name == "CA")[0]
        if ca_indices.size == 0:
            continue
        residues.append(
            ResidueGeometry(
                key=_residue_identifier(residue),
                resname=resname,
                chain_id=str(residue.chain_id[0]),
                resseq=int(residue.res_id[0]),
                icode=str(residue.ins_code[0]).strip(),
                heavy_coords=heavy_coords,
                ca_coord=coords[int(ca_indices[0])],
            )
        )
    return residues


def _minimum_pair_distance(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    deltas = coords_a[:, np.newaxis, :] - coords_b[np.newaxis, :, :]
    return float(np.linalg.norm(deltas, axis=-1).min())


def _iter_metal_sites(atom_array) -> Iterable[tuple[str, np.ndarray]]:
    hetero_metals = atom_array[
        np.isin(atom_array.element, np.array(list(METAL_CHARGE_PROXIES)))
    ]
    for atom in hetero_metals:
        yield str(atom.element), atom.coord.astype(np.float64, copy=False)


def _apply_sasa_and_burial(
    atom_array,
    residues: list[ResidueGeometry],
    feature_map: dict[ResidueKey, dict[str, float]],
) -> None:
    protein_mask = struc.filter_canonical_amino_acids(atom_array)
    protein_atoms = atom_array[protein_mask]
    sasa = struc.sasa(
        protein_atoms,
        ignore_ions=True,
    )
    atom_starts = struc.get_residue_starts(protein_atoms)
    atom_stops = list(atom_starts[1:]) + [protein_atoms.array_length()]

    for residue, start, stop in zip(residues, atom_starts, atom_stops):
        residue_sasa = float(np.nansum(sasa[start:stop]))
        max_sasa = MAX_RESIDUE_SASA.get(residue.resname, max(MAX_RESIDUE_SASA.values()))
        _set_feature(feature_map[residue.key], EXTERNAL_FEATURE_RESIDUE_SASA, residue_sasa)


def _apply_pairwise_interaction_proxies(
    residues: list[ResidueGeometry],
    atom_array,
    feature_map: dict[ResidueKey, dict[str, float]],
) -> None:
    # Heuristic q1*q2/r-style charge-distance feature derived from Biotite
    # geometry and coarse residue/metal charge proxies.
    for residue in residues:
        _set_feature(
            feature_map[residue.key],
            EXTERNAL_FEATURE_CUSTOM_CHARGE_DISTANCE_PROXY,
            feature_map[residue.key][EXTERNAL_FEATURE_CUSTOM_CHARGE_DISTANCE_PROXY],
        )

    for left_index, residue_left in enumerate(residues):
        for right_index in range(left_index + 1, len(residues)):
            residue_right = residues[right_index]
            if (
                residue_left.chain_id == residue_right.chain_id
                and abs(residue_left.resseq - residue_right.resseq) <= 1
            ):
                continue
            ca_distance = float(np.linalg.norm(residue_left.ca_coord - residue_right.ca_coord))
            if ca_distance > 12.0:
                continue

            min_distance = _minimum_pair_distance(residue_left.heavy_coords, residue_right.heavy_coords)
            if min_distance <= 0.0:
                continue

            electrostatics = 0.0

            charge_left = RESIDUE_CHARGE_PROXIES.get(residue_left.resname, 0.0)
            charge_right = RESIDUE_CHARGE_PROXIES.get(residue_right.resname, 0.0)
            if charge_left and charge_right:
                electrostatics = (charge_left * charge_right) / max(min_distance, 2.5)

            for key in (residue_left.key, residue_right.key):
                _set_feature(
                    feature_map[key],
                    EXTERNAL_FEATURE_CUSTOM_CHARGE_DISTANCE_PROXY,
                    feature_map[key][EXTERNAL_FEATURE_CUSTOM_CHARGE_DISTANCE_PROXY] + electrostatics,
                )

    metal_sites = list(_iter_metal_sites(atom_array))
    if not metal_sites:
        return

    for residue in residues:
        charge = RESIDUE_CHARGE_PROXIES.get(residue.resname, 0.0)
        if not charge:
            continue
        for metal_element, metal_coord in metal_sites:
            min_distance = float(np.linalg.norm(residue.heavy_coords - metal_coord, axis=1).min())
            metal_charge = METAL_CHARGE_PROXIES.get(metal_element, 0.0)
            if not metal_charge:
                continue
            _set_feature(
                feature_map[residue.key],
                EXTERNAL_FEATURE_CUSTOM_CHARGE_DISTANCE_PROXY,
                feature_map[residue.key][EXTERNAL_FEATURE_CUSTOM_CHARGE_DISTANCE_PROXY] + (charge * metal_charge) / max(min_distance, 2.0),
            )


def _apply_propka_features(
    residues: list[ResidueGeometry],
    feature_map: dict[ResidueKey, dict[str, float]],
    propka_result: PropkaRunResult | None,
) -> list[str]:
    if propka_result is None:
        return []

    residue_index_by_chain_resseq_resname: dict[tuple[str, int, str], list[ResidueKey]] = {}
    for residue in residues:
        residue_index_by_chain_resseq_resname.setdefault(
            (residue.chain_id, residue.resseq, residue.resname),
            [],
        ).append(residue.key)

    for compact_key, propka_features in propka_result.residues.items():
        chain_id, resseq, resname = compact_key
        for residue_key in residue_index_by_chain_resseq_resname.get((chain_id, resseq, resname), []):
            # PROPKA-derived titration/electrostatic contribution. This is kept
            # separate from the geometric charge-distance proxy because the two
            # quantities are not assumed to share a common physical scale.
            entry = feature_map[residue_key]
            _set_feature(
                entry,
                EXTERNAL_FEATURE_DPKA_TITR,
                propka_features.dpka_titr,
            )
    return propka_result.warnings


def generate_feature_map_for_structure(
    structure_path: str | Path,
    *,
    propka_ph: float = 7.0,
    include_propka: bool = True,
) -> tuple[dict[ResidueKey, dict[str, float]], dict[str, object]]:
    structure_path = Path(structure_path)
    atom_array = load_structure(str(structure_path))
    protein_atoms = atom_array[struc.filter_canonical_amino_acids(atom_array)]
    residues = _residue_geometry_from_atom_array(protein_atoms)
    feature_map = {residue.key: default_feature_dict() for residue in residues}

    _apply_sasa_and_burial(atom_array, residues, feature_map)
    _apply_pairwise_interaction_proxies(residues, atom_array, feature_map)

    warnings: list[str] = []
    propka_used = False
    if include_propka:
        try:
            propka_result = run_propka_for_structure(structure_path, ph=propka_ph)
        except Exception as exc:
            warnings.append(str(exc))
            propka_result = None
        else:
            propka_used = True
        warnings.extend(_apply_propka_features(residues, feature_map, propka_result))

    metadata = {
        "structure_id": structure_path.stem,
        "source_path": str(structure_path),
        "feature_names": list(FEATURE_NAMES),
        "tooling": {
            "geometry": "biotite",
            "pka": "propka" if propka_used else "unavailable",
        },
        "warnings": warnings,
        "n_residues": len(residues),
    }
    return feature_map, metadata


def build_structure_feature_payload(
    structure_path: str | Path,
    *,
    propka_ph: float = 7.0,
    include_propka: bool = True,
) -> dict[str, object]:
    feature_map, metadata = generate_feature_map_for_structure(
        structure_path,
        propka_ph=propka_ph,
        include_propka=include_propka,
    )
    residues_payload = [
        {
            "chain_id": chain_id,
            "resseq": resseq,
            "icode": icode,
            "features": feature_map[(chain_id, resseq, icode)],
        }
        for chain_id, resseq, icode in sorted(feature_map)
    ]
    return {
        "schema_version": 1,
        **metadata,
        "residues": residues_payload,
    }
