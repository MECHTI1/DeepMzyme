"""Frozen source-row cohort binding for ion-level metal examples.

A cohort CSV binds every retained source row (``source_uid``) to exactly one
physical ion atom: structure file and content hash, model, chain, residue
number, insertion code, atom name, alternate location and coordinate. The
loader builds one example per cohort row from that binding alone; it never
matches ions by summary keys, nearest distance or residue number without an
insertion code. A binding that cannot be reproduced exactly is a fatal error,
not a silently skipped example.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from data_structures import DEFAULT_MULTINUCLEAR_MERGE_DISTANCE, DEFAULT_POCKET_RADIUS, PocketRecord
from graph.structure_parsing import (
    MetalAtomRecord,
    canonicalize_site_metal_resname,
    cluster_metal_records,
    collect_structure_residues_and_metals,
    find_pocket_residues_near_metal_cluster,
)

COHORT_SCHEMA_VERSION = 1
COORDINATE_MATCH_TOLERANCE = 2.0e-3
COHORT_COLUMNS = (
    "source_uid",
    "source_row",
    "pdbid",
    "group_id",
    "structure_name",
    "structure_sha256",
    "model_index",
    "chain",
    "resseq",
    "icode",
    "resname",
    "atom_name",
    "altloc",
    "coord_x",
    "coord_y",
    "coord_z",
    "native_element",
    "source_label_code",
    "source_residueid_ion",
    "source_metalid",
    "physical_ion_id",
    "alias_source_uids",
    "context_chains",
    "n_context_residues",
    "n_first_shell_residues",
)


class CohortBindingError(RuntimeError):
    """A frozen cohort binding could not be reproduced from its structure file."""


@dataclass(frozen=True)
class CohortBinding:
    source_uid: str
    source_row: int
    pdbid: str
    group_id: str
    structure_name: str
    structure_sha256: str
    model_index: int
    chain: str
    resseq: int
    icode: str
    resname: str
    atom_name: str
    altloc: str
    coord: tuple[float, float, float]
    native_element: str
    physical_ion_id: str
    alias_source_uids: tuple[str, ...]

    @property
    def structure_stem(self) -> str:
        return Path(self.structure_name).stem

    @property
    def site_id(self) -> tuple[str, int, str]:
        return (self.chain, self.resseq, self.icode)

    def example_id(self) -> str:
        return f"{self.structure_stem}__SRC_{self.source_row}"


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def physical_ion_id(pdbid: str, model_index: int, chain: str, resseq: int, icode: str, atom_name: str, altloc: str) -> str:
    return f"{pdbid.lower()}|m{model_index}|{chain}|{resseq}|{icode or '-'}|{atom_name}|{altloc or '-'}"


def format_coord(value: float) -> str:
    return f"{float(value):.3f}"


def binding_from_row(row: dict[str, str]) -> CohortBinding:
    return CohortBinding(
        source_uid=row["source_uid"],
        source_row=int(row["source_row"]),
        pdbid=row["pdbid"].lower(),
        group_id=row["group_id"],
        structure_name=row["structure_name"],
        structure_sha256=row["structure_sha256"],
        model_index=int(row["model_index"]),
        chain=row["chain"],
        resseq=int(row["resseq"]),
        icode=row["icode"],
        resname=row["resname"],
        atom_name=row["atom_name"],
        altloc=row["altloc"],
        coord=(float(row["coord_x"]), float(row["coord_y"]), float(row["coord_z"])),
        native_element=row["native_element"].upper(),
        physical_ion_id=row["physical_ion_id"],
        alias_source_uids=tuple(uid for uid in row.get("alias_source_uids", "").split(";") if uid),
    )


def read_cohort_csv(path: Path, expected_sha256: str | None = None) -> list[CohortBinding]:
    """Read a frozen cohort in file order; reject hash drift and repeated identities."""
    path = Path(path)
    if expected_sha256 is not None:
        actual = sha256_file(path)
        if actual != expected_sha256.lower():
            raise CohortBindingError(f"Cohort file {path} has SHA256 {actual}, expected {expected_sha256}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in COHORT_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise CohortBindingError(f"Cohort file {path} is missing columns {missing}")
        bindings = [binding_from_row(row) for row in reader]
    seen_uids: set[str] = set()
    seen_ions: set[str] = set()
    for binding in bindings:
        if binding.source_uid in seen_uids:
            raise CohortBindingError(f"Repeated source_uid in cohort: {binding.source_uid}")
        if binding.physical_ion_id in seen_ions:
            raise CohortBindingError(
                f"Physical ion {binding.physical_ion_id} appears twice; aliases must be merged into one row"
            )
        seen_uids.add(binding.source_uid)
        seen_ions.add(binding.physical_ion_id)
    if not bindings:
        raise CohortBindingError(f"Cohort file {path} has no retained rows")
    return bindings


def bindings_by_structure(bindings: Iterable[CohortBinding]) -> dict[str, list[CohortBinding]]:
    grouped: dict[str, list[CohortBinding]] = {}
    for binding in bindings:
        grouped.setdefault(binding.structure_stem, []).append(binding)
    return grouped


def cohort_load_payload(bindings: Iterable[CohortBinding]) -> dict[str, tuple[tuple[Any, ...], ...]]:
    """Picklable, hashable per-structure binding tuples for (parallel) loaders."""
    payload: dict[str, list[tuple[Any, ...]]] = {}
    for binding in bindings:
        payload.setdefault(binding.structure_stem, []).append(
            (
                binding.source_uid, binding.source_row, binding.pdbid, binding.group_id,
                binding.structure_name, binding.structure_sha256, binding.model_index,
                binding.chain, binding.resseq, binding.icode, binding.resname,
                binding.atom_name, binding.altloc, binding.coord, binding.native_element,
                binding.physical_ion_id, binding.alias_source_uids,
            )
        )
    return {stem: tuple(rows) for stem, rows in payload.items()}


def binding_from_payload(item: tuple[Any, ...]) -> CohortBinding:
    return CohortBinding(*item)


@dataclass(frozen=True)
class TargetAtomCandidate:
    model_index: int
    chain: str
    resseq: int
    icode: str
    resname: str
    symbol: str | None
    atom_name: str
    selected_altloc: str
    altloc_coords: tuple[tuple[str, tuple[float, float, float], float], ...]
    coord: tuple[float, float, float]
    n_atoms_in_residue: int

    @property
    def is_disordered(self) -> bool:
        return len(self.altloc_coords) > 1


def _atom_coord(atom) -> tuple[float, float, float]:
    return tuple(float(value) for value in atom.coord)  # type: ignore[return-value]


def target_atom_candidates(structure, *, model_index: int, chain: str, resseq: int) -> list[TargetAtomCandidate]:
    """All hetero atoms at ``chain``/``resseq`` (any insertion code) in one model."""
    models = list(structure)
    if model_index >= len(models):
        return []
    candidates: list[TargetAtomCandidate] = []
    for current_chain in models[model_index]:
        if str(current_chain.id) != str(chain):
            continue
        for residue in current_chain:
            hetflag, current_resseq, icode = residue.id
            if int(current_resseq) != int(resseq) or not str(hetflag).strip():
                continue
            atoms = list(residue.get_atoms())
            for atom in atoms:
                children = atom.disordered_get_list() if atom.is_disordered() else [atom]
                altloc_coords = tuple(
                    (str(child.get_altloc()).strip(), _atom_coord(child), float(child.get_occupancy() or 0.0))
                    for child in children
                )
                candidates.append(
                    TargetAtomCandidate(
                        model_index=model_index,
                        chain=str(current_chain.id),
                        resseq=int(current_resseq),
                        icode=str(icode).strip(),
                        resname=residue.resname.strip(),
                        symbol=canonicalize_site_metal_resname(residue.resname),
                        atom_name=atom.get_name().strip(),
                        selected_altloc=str(atom.get_altloc()).strip(),
                        altloc_coords=altloc_coords,
                        coord=_atom_coord(atom),
                        n_atoms_in_residue=len(atoms),
                    )
                )
    return candidates


def _coord_distance(a: Iterable[float], b: Iterable[float]) -> float:
    return float(torch.linalg.vector_norm(torch.tensor(list(a)) - torch.tensor(list(b))).item())


def build_cohort_ion_examples(
    structure,
    *,
    structure_id: str,
    structure_path: Path,
    bindings: list[CohortBinding],
    pocket_radius: float = DEFAULT_POCKET_RADIUS,
) -> list[PocketRecord]:
    """Create exactly one ion example per binding, in binding order.

    Residues come from the bound model only. The parent pocket (the existing
    4.5 Å metal cluster) is retained as context metadata; it never merges two
    bound ions into one example.
    """
    if not bindings:
        return []
    model_indices = {binding.model_index for binding in bindings}
    if len(model_indices) != 1:
        raise CohortBindingError(f"{structure_id}: bindings span several models {sorted(model_indices)}")
    model_index = next(iter(model_indices))
    residues, metal_records = collect_structure_residues_and_metals(structure, model_index=model_index)
    residue_coords = [torch.stack([coord.float() for coord in residue.atoms.values()], dim=0) for residue in residues]
    clusters = cluster_metal_records(metal_records, merge_distance=DEFAULT_MULTINUCLEAR_MERGE_DISTANCE)
    cluster_of_site: dict[tuple[str, int, str], int] = {}
    for cluster_index, cluster in enumerate(clusters):
        for record in cluster:
            cluster_of_site.setdefault(record.site_id, cluster_index)

    examples: list[PocketRecord] = []
    for binding in bindings:
        candidates = [
            candidate
            for candidate in target_atom_candidates(
                structure, model_index=model_index, chain=binding.chain, resseq=binding.resseq,
            )
            if candidate.icode == binding.icode and candidate.atom_name == binding.atom_name
        ]
        if len(candidates) != 1:
            raise CohortBindingError(
                f"{structure_id}: binding {binding.source_uid} resolves to {len(candidates)} atoms at "
                f"{binding.site_id}/{binding.atom_name}"
            )
        candidate = candidates[0]
        altloc_positions = {altloc: coord for altloc, coord, _occ in candidate.altloc_coords}
        bound_coord = altloc_positions.get(binding.altloc, candidate.coord if not binding.altloc else None)
        if bound_coord is None or _coord_distance(bound_coord, binding.coord) > COORDINATE_MATCH_TOLERANCE:
            raise CohortBindingError(
                f"{structure_id}: binding {binding.source_uid} coordinate differs from the frozen cohort"
            )
        if candidate.symbol != binding.native_element:
            raise CohortBindingError(
                f"{structure_id}: binding {binding.source_uid} element {candidate.symbol} != {binding.native_element}"
            )
        coord = torch.tensor(binding.coord, dtype=torch.float32)
        target = MetalAtomRecord(coord=coord, symbol=binding.native_element, site_id=binding.site_id)
        nearby = find_pocket_residues_near_metal_cluster(
            residues, [target], pocket_radius=pocket_radius, residue_coord_tensors=residue_coords,
        )
        if not nearby:
            raise CohortBindingError(f"{structure_id}: binding {binding.source_uid} has no residues within {pocket_radius} Å")
        cluster_index = cluster_of_site.get(binding.site_id)
        parent_pocket_id = (
            f"{structure_id}_METAL_{cluster_index}" if cluster_index is not None else f"{structure_id}_METAL_unclustered"
        )
        example = PocketRecord(
            structure_id=structure_id,
            pocket_id=binding.example_id(),
            metal_element=binding.native_element,
            metal_coords=[coord],
            residues=nearby,
            metadata={
                "source_path": str(structure_path),
                "source_uid": binding.source_uid,
                "source_row": binding.source_row,
                "alias_source_uids": list(binding.alias_source_uids),
                "physical_ion_id": binding.physical_ion_id,
                "cohort_group_id": binding.group_id,
                "parent_pocket_id": parent_pocket_id,
                "ion_index": binding.source_row,
                "ion_site_id": binding.site_id,
                "ion_symbol_observed": binding.native_element,
                "ion_symbol_target": binding.native_element,
                "metal_site_ids": [binding.site_id],
                "metal_site_symbols": [binding.native_element],
                "metal_symbols_observed": [binding.native_element],
                "matched_summary_site_metal_types": [binding.native_element],
                "metal_site_coord_map": {binding.site_id: coord},
                "model_index": model_index,
            },
        )
        examples.append(example)
    return examples
