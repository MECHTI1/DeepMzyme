from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from Bio.PDB import MMCIFParser, PDBParser
from torch import Tensor

from data_structures import (
    AA_ORDER,
    DEFAULT_MULTINUCLEAR_MERGE_DISTANCE,
    DEFAULT_POCKET_RADIUS,
    GENERIC_METAL_ELEMENT,
    PocketRecord,
    ResidueRecord,
    SUPPORTED_SITE_METAL_ELEMENTS,
)
from featurization import pairwise_distances, safe_norm


@dataclass(frozen=True)
class MetalAtomRecord:
    coord: Tensor
    symbol: str
    site_id: Tuple[str, int, str]


def parse_structure_file(filepath: str, structure_id: Optional[str] = None):
    path = Path(filepath)
    sid = structure_id or path.stem
    parser = MMCIFParser(QUIET=True) if path.suffix.lower() in {".cif", ".mmcif"} else PDBParser(QUIET=True)
    return parser.get_structure(sid, str(path))


def residue_record_from_biopython_residue(residue) -> Optional[ResidueRecord]:
    hetflag, resseq, icode = residue.id
    if hetflag.strip():
        return None
    if residue.resname.strip() not in AA_ORDER:
        return None

    atoms = {}
    selected_altloc_ids = set()
    for atom in residue.get_atoms():
        altloc = str(getattr(atom, "get_altloc", lambda: " ")()).strip()
        if altloc:
            selected_altloc_ids.add(altloc)
        atoms[atom.get_name().strip()] = torch.tensor(atom.coord, dtype=torch.float32)

    if len(selected_altloc_ids) > 1:
        return None

    if "CA" not in atoms:
        return None

    return ResidueRecord(
        chain_id=str(residue.get_parent().id),
        resseq=int(resseq),
        icode=str(icode).strip() if str(icode).strip() else "",
        resname=residue.resname.strip(),
        atoms=atoms,
    )


def normalize_site_metal_resname(resname: str) -> str:
    return "".join(ch for ch in resname.strip().upper() if ch.isalpha())


def canonicalize_site_metal_resname(resname: str) -> Optional[str]:
    raw = resname.strip().upper()
    letters_only = normalize_site_metal_resname(raw)

    for symbol in sorted(SUPPORTED_SITE_METAL_ELEMENTS, key=len, reverse=True):
        if symbol in raw and letters_only == symbol:
            return symbol
    return None


def biopython_residue_site_id(residue, chain_id: str) -> Tuple[str, int, str]:
    _, resseq, icode = residue.id
    return str(chain_id), int(resseq), str(icode).strip() if str(icode).strip() else ""


def metal_records_from_biopython_residue(residue, chain_id: str) -> List[MetalAtomRecord]:
    metal_symbol = canonicalize_site_metal_resname(residue.resname)
    if metal_symbol is None:
        return []

    site_id = biopython_residue_site_id(residue, chain_id)
    return [
        MetalAtomRecord(
            coord=torch.tensor(atom.coord, dtype=torch.float32),
            symbol=metal_symbol,
            site_id=site_id,
        )
        for atom in residue.get_atoms()
    ]


def collect_structure_residues_and_metals(
    structure,
    model_index: Optional[int] = None,
) -> Tuple[List[ResidueRecord], List[MetalAtomRecord]]:
    """Collect protein residues and metal atoms.

    ``model_index=None`` keeps the historical traversal of every model. A
    multi-model (e.g. NMR) file then contributes duplicate residues and ions, so
    callers that need one physical coordinate set pass the model position.
    """
    all_residues: List[ResidueRecord] = []
    metal_records: List[MetalAtomRecord] = []

    models = list(structure) if model_index is None else [list(structure)[model_index]]
    for model in models:
        for chain in model:
            for residue in chain:
                residue_metal_records = metal_records_from_biopython_residue(residue, chain.id)
                if residue_metal_records:
                    metal_records.extend(residue_metal_records)
                    continue

                residue_record = residue_record_from_biopython_residue(residue)
                if residue_record is not None:
                    all_residues.append(residue_record)

    return all_residues, metal_records


def cluster_metal_records(
    metal_records: List[MetalAtomRecord],
    merge_distance: float,
) -> List[List[MetalAtomRecord]]:
    if not metal_records:
        return []

    coords = torch.stack([record.coord.float() for record in metal_records], dim=0)
    dmat = pairwise_distances(coords)

    clusters: List[List[MetalAtomRecord]] = []
    visited = set()
    for start_idx in range(len(metal_records)):
        if start_idx in visited:
            continue

        stack = [start_idx]
        component = []
        visited.add(start_idx)
        while stack:
            idx = stack.pop()
            component.append(metal_records[idx])
            neighbors = torch.where(dmat[idx] <= merge_distance)[0].tolist()
            for neighbor_idx in neighbors:
                if neighbor_idx in visited:
                    continue
                visited.add(neighbor_idx)
                stack.append(neighbor_idx)

        clusters.append(component)

    return clusters


def find_pocket_residues_near_metal_cluster(
    all_residues: List[ResidueRecord],
    metal_cluster: List[MetalAtomRecord],
    pocket_radius: float,
    residue_coord_tensors: Optional[List[Tensor]] = None,
) -> List[ResidueRecord]:
    cluster_tensor = torch.stack([record.coord.float() for record in metal_cluster], dim=0)
    pocket_residues = []
    if residue_coord_tensors is None:
        residue_coord_tensors = [
            torch.stack([coord.float() for coord in residue_record.atoms.values()], dim=0)
            for residue_record in all_residues
        ]

    for residue_record, residue_coords in zip(all_residues, residue_coord_tensors):
        diff = residue_coords[:, None, :] - cluster_tensor[None, :, :]
        if safe_norm(diff, dim=-1).min().item() <= pocket_radius:
            pocket_residues.append(residue_record)

    return pocket_residues


def pocket_record_from_metal_cluster(
    structure_id: str,
    cluster_index: int,
    metal_cluster: List[MetalAtomRecord],
    all_residues: List[ResidueRecord],
    pocket_radius: float,
    residue_coord_tensors: Optional[List[Tensor]] = None,
) -> Optional[PocketRecord]:
    pocket_residues = find_pocket_residues_near_metal_cluster(
        all_residues,
        metal_cluster,
        pocket_radius=pocket_radius,
        residue_coord_tensors=residue_coord_tensors,
    )
    if not pocket_residues:
        return None

    cluster_coords = [record.coord.float() for record in metal_cluster]
    cluster_symbols = sorted({record.symbol for record in metal_cluster})
    return PocketRecord(
        structure_id=structure_id,
        pocket_id=f"{structure_id}_METAL_{cluster_index}",
        metal_element=cluster_symbols[0] if len(cluster_symbols) == 1 else GENERIC_METAL_ELEMENT,
        metal_coords=cluster_coords,
        residues=pocket_residues,
        metadata={
            "metal_symbols_observed": cluster_symbols,
            "metal_site_symbols": [record.symbol for record in metal_cluster],
            "metal_site_ids": [record.site_id for record in metal_cluster],
            "metal_site_coord_map": {
                record.site_id: record.coord.float()
                for record in metal_cluster
            },
        },
    )


def extract_metal_pockets_from_structure(
    structure,
    structure_id: Optional[str] = None,
    pocket_radius: float = DEFAULT_POCKET_RADIUS,
    multinuclear_merge_distance: float = DEFAULT_MULTINUCLEAR_MERGE_DISTANCE,
    model_index: Optional[int] = None,
) -> List[PocketRecord]:
    sid = structure_id or getattr(structure, "id", "unknown_structure")
    all_residues, metal_records = collect_structure_residues_and_metals(structure, model_index=model_index)
    residue_coord_tensors = [
        torch.stack([coord.float() for coord in residue_record.atoms.values()], dim=0)
        for residue_record in all_residues
    ]
    metal_clusters = cluster_metal_records(metal_records, merge_distance=multinuclear_merge_distance)

    pockets: List[PocketRecord] = []
    for idx, metal_cluster in enumerate(metal_clusters):
        pocket = pocket_record_from_metal_cluster(
            structure_id=sid,
            cluster_index=idx,
            metal_cluster=metal_cluster,
            all_residues=all_residues,
            pocket_radius=pocket_radius,
            residue_coord_tensors=residue_coord_tensors,
        )
        if pocket is not None:
            pockets.append(pocket)
    return pockets


# Selected-site extraction: pockets are built only from the metal sites named by a
# site summary, instead of from every metal component in the file.
SELECTED_SITE_EXTRACTION = "selected_sites"
NO_SELECTED_SITES_REASON = "no_selected_sites"
SelectedSiteId = Tuple[str, int]


class UnresolvedSelectedSiteError(ValueError):
    """A selected site has no metal atom record in the structure."""


class AmbiguousSelectedSiteError(ValueError):
    """A selected site resolves to more than one metal atom record."""


def metal_record_selected_site_id(record: MetalAtomRecord) -> SelectedSiteId:
    # Same normalization as training.site_filter.matched_site_keys_for_pocket:
    # stripped chain, integer residue number, insertion code ignored.
    chain_id, resseq, _icode = record.site_id
    return str(chain_id).strip(), int(str(resseq).strip())


def _normalize_selected_site_ids(selected_site_ids) -> frozenset[SelectedSiteId]:
    normalized: set[SelectedSiteId] = set()
    for site in selected_site_ids:
        if not isinstance(site, tuple) or len(site) != 2:
            raise ValueError(f"Selected site ids must be (chain, resseq) tuples, got {site!r}.")
        chain_id, resseq = site
        if not str(chain_id).strip() or isinstance(resseq, bool) or not isinstance(resseq, int):
            raise ValueError(f"Selected site ids must be (chain, resseq) tuples, got {site!r}.")
        normalized.add((str(chain_id).strip(), resseq))
    return frozenset(normalized)


def _residue_coordinate_tensors(all_residues: List[ResidueRecord]) -> List[Tensor]:
    return [
        torch.stack([coord.float() for coord in residue_record.atoms.values()], dim=0)
        for residue_record in all_residues
    ]


def extract_selected_site_pockets_from_structure(
    structure,
    *,
    structure_id: str,
    selected_site_ids: frozenset[SelectedSiteId] | set[SelectedSiteId],
    pocket_radius: float = DEFAULT_POCKET_RADIUS,
    multinuclear_merge_distance: float = DEFAULT_MULTINUCLEAR_MERGE_DISTANCE,
    model_index: Optional[int] = None,
) -> tuple[List[PocketRecord], dict[str, object]]:
    """Extract pockets defined only by the selected metal sites.

    Unlike ``extract_metal_pockets_from_structure``, metal atom records are
    filtered to ``selected_site_ids`` before clustering, so pocket residues,
    ``metal_coords`` and everything derived from them see only selected ions.
    Selected ions within ``multinuclear_merge_distance`` still form one
    multinuclear pocket; an unselected ion never bridges two selected ones.

    Each selected ``(chain, resseq)`` must resolve to exactly one metal atom
    record. Insertion codes are ignored, as in the summary-site matcher, so two
    records for one site (insertion-code variants, several atoms in one metal
    residue, or every model of a multi-model file when ``model_index`` is None)
    raise ``AmbiguousSelectedSiteError``; a missing one raises
    ``UnresolvedSelectedSiteError``.

    Pocket ids number the filtered clusters (``{structure_id}_METAL_{i}``), so
    they need not equal component-mode ids; ``metadata["site_extraction"]``
    records the mode. An empty selection returns no pockets and sets
    ``report["no_selected_sites"]``; the caller then records a skipped row
    with reason ``NO_SELECTED_SITES_REASON``.

    The report compares against the legacy components (all metals clustered)
    that hold a selected site: ``split_components`` lists those that yield
    more than one selected pocket, and ``residue_count_delta_total`` sums, per
    such component, the residues covered by its selected pockets minus its
    legacy pocket residues. Selected sites whose cluster has no residue within
    ``pocket_radius`` appear only in ``selected_sites_without_pocket``.
    """
    if not structure_id:
        raise ValueError("structure_id is required for selected-site extraction.")
    selected = _normalize_selected_site_ids(selected_site_ids)
    all_residues, metal_records = collect_structure_residues_and_metals(structure, model_index=model_index)

    records_by_site: dict[SelectedSiteId, List[MetalAtomRecord]] = {}
    for record in metal_records:
        records_by_site.setdefault(metal_record_selected_site_id(record), []).append(record)
    unresolved = sorted(site for site in selected if site not in records_by_site)
    if unresolved:
        raise UnresolvedSelectedSiteError(
            f"{structure_id}: selected sites without a metal atom record: {unresolved}"
        )
    ambiguous = {
        site: [record.site_id for record in records_by_site[site]]
        for site in sorted(selected)
        if len(records_by_site[site]) > 1
    }
    if ambiguous:
        raise AmbiguousSelectedSiteError(
            f"{structure_id}: selected sites with more than one metal atom record: {ambiguous}"
        )

    selected_records = [record for record in metal_records if metal_record_selected_site_id(record) in selected]
    selected_record_ids = {id(record) for record in selected_records}
    legacy_clusters = cluster_metal_records(metal_records, merge_distance=multinuclear_merge_distance)
    legacy_index_by_record = {
        id(record): component_index
        for component_index, component in enumerate(legacy_clusters)
        for record in component
    }
    selected_clusters = cluster_metal_records(selected_records, merge_distance=multinuclear_merge_distance)
    residue_coord_tensors = _residue_coordinate_tensors(all_residues)

    pockets: List[PocketRecord] = []
    pocket_rows: list[dict[str, object]] = []
    sites_without_pocket: list[SelectedSiteId] = []
    clusters_by_component: dict[int, list[int]] = {}
    pocket_ids_by_cluster: dict[int, str] = {}
    legacy_residue_counts: dict[int, int] = {}
    # Residue records are shared objects of all_residues, so the union over a
    # split component's pockets counts each residue once.
    pocket_residue_ids_by_component: dict[int, set[int]] = {}
    for cluster_index, metal_cluster in enumerate(selected_clusters):
        # A filtered cluster is a subset of exactly one legacy component.
        component_indices = {legacy_index_by_record[id(record)] for record in metal_cluster}
        if len(component_indices) != 1:
            raise RuntimeError(f"{structure_id}: selected cluster spans legacy components {component_indices}.")
        component_index = component_indices.pop()
        clusters_by_component.setdefault(component_index, []).append(cluster_index)
        legacy_component = legacy_clusters[component_index]
        if component_index not in legacy_residue_counts:
            legacy_residue_counts[component_index] = len(
                find_pocket_residues_near_metal_cluster(
                    all_residues,
                    legacy_component,
                    pocket_radius=pocket_radius,
                    residue_coord_tensors=residue_coord_tensors,
                )
            )
        cluster_site_keys = [metal_record_selected_site_id(record) for record in metal_cluster]
        pocket = pocket_record_from_metal_cluster(
            structure_id=structure_id,
            cluster_index=cluster_index,
            metal_cluster=metal_cluster,
            all_residues=all_residues,
            pocket_radius=pocket_radius,
            residue_coord_tensors=residue_coord_tensors,
        )
        if pocket is None:
            sites_without_pocket.extend(cluster_site_keys)
            continue

        legacy_component_site_ids = [record.site_id for record in legacy_component]
        pocket.metadata["site_extraction"] = SELECTED_SITE_EXTRACTION
        pocket.metadata["selected_site_keys"] = cluster_site_keys
        pocket.metadata["legacy_component_site_ids"] = legacy_component_site_ids
        pockets.append(pocket)
        pocket_ids_by_cluster[cluster_index] = pocket.pocket_id
        pocket_residue_ids_by_component.setdefault(component_index, set()).update(
            id(residue) for residue in pocket.residues
        )

        legacy_residue_count = legacy_residue_counts[component_index]
        pocket_rows.append(
            {
                "pocket_id": pocket.pocket_id,
                "selected_site_keys": cluster_site_keys,
                "metal_count": pocket.metal_count(),
                "legacy_pocket_id": f"{structure_id}_METAL_{component_index}",
                "legacy_component_site_ids": legacy_component_site_ids,
                "legacy_component_metal_count": len(legacy_component),
                "residue_count": len(pocket.residues),
                "legacy_residue_count": legacy_residue_count,
                "residue_count_delta": len(pocket.residues) - legacy_residue_count,
            }
        )

    components_with_unselected_metals = []
    legacy_components_without_selected_sites = 0
    for component_index, component in enumerate(legacy_clusters):
        unselected_site_ids = [record.site_id for record in component if id(record) not in selected_record_ids]
        if len(unselected_site_ids) == len(component):
            legacy_components_without_selected_sites += 1
        elif unselected_site_ids:
            components_with_unselected_metals.append(
                {
                    "legacy_pocket_id": f"{structure_id}_METAL_{component_index}",
                    "legacy_component_site_ids": [record.site_id for record in component],
                    "unselected_site_ids": unselected_site_ids,
                }
            )
    split_components = []
    for component_index, cluster_indices in sorted(clusters_by_component.items()):
        component_pocket_ids = [
            pocket_ids_by_cluster[cluster_index]
            for cluster_index in cluster_indices
            if cluster_index in pocket_ids_by_cluster
        ]
        if len(component_pocket_ids) > 1:
            split_components.append(
                {
                    "legacy_pocket_id": f"{structure_id}_METAL_{component_index}",
                    "legacy_component_site_ids": [record.site_id for record in legacy_clusters[component_index]],
                    "selected_cluster_count": len(cluster_indices),
                    "pocket_ids": component_pocket_ids,
                }
            )
    residue_count_delta_total = sum(
        len(pocket_residue_ids_by_component.get(component_index, ())) - legacy_residue_count
        for component_index, legacy_residue_count in legacy_residue_counts.items()
    )

    report: dict[str, object] = {
        "site_extraction": SELECTED_SITE_EXTRACTION,
        "structure_id": structure_id,
        "model_index": model_index,
        "pocket_radius": float(pocket_radius),
        "multinuclear_merge_distance": float(multinuclear_merge_distance),
        "selected_site_keys": sorted(selected),
        "no_selected_sites": not selected,
        "metal_records_total": len(metal_records),
        "metal_records_selected": len(selected_records),
        "metal_records_dropped": len(metal_records) - len(selected_records),
        "legacy_component_count": len(legacy_clusters),
        "legacy_components_without_selected_sites": legacy_components_without_selected_sites,
        "selected_cluster_count": len(selected_clusters),
        "pocket_count": len(pockets),
        "pockets": pocket_rows,
        "components_with_unselected_metals": components_with_unselected_metals,
        "components_with_unselected_metals_count": len(components_with_unselected_metals),
        "split_components": split_components,
        "split_component_count": len(split_components),
        "selected_sites_without_pocket": sorted(sites_without_pocket),
        "residue_count_delta_total": residue_count_delta_total,
    }
    return pockets, report
