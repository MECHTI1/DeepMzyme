"""Create one supervised graph example per observed metal ion in a pocket."""

from __future__ import annotations

from pathlib import Path

from data_structures import DEFAULT_POCKET_RADIUS, PocketRecord
from graph.structure_parsing import (
    MetalAtomRecord, canonicalize_site_metal_resname,
    find_pocket_residues_near_metal_cluster,
)
from label_schemes import map_site_metal_symbols
from training.site_filter import AllowedSiteMetalLabels, matched_site_keys_for_pocket


def metal_ion_examples(
    parent: PocketRecord,
    structure_path: Path,
    allowed_site_metal_labels: AllowedSiteMetalLabels | None,
    *,
    unsupported_metal_policy: str,
) -> tuple[list[PocketRecord], list[dict[str, str]]]:
    """Keep each ion only when its own site has an unambiguous metal label.

    Summary labels are authoritative when a summary CSV is supplied. A site
    whose observed metal conflicts with that label is recorded and skipped.
    Shared residues are filtered to the same 10 Å radius around each ion.
    """
    site_ids = parent.metadata.get("metal_site_ids")
    symbols = parent.metadata.get("metal_site_symbols")
    if not isinstance(site_ids, list) or len(site_ids) != parent.metal_count():
        raise ValueError(f"Missing aligned metal-site IDs for {parent.pocket_id}")
    if not isinstance(symbols, list) or len(symbols) != parent.metal_count():
        if parent.metal_count() == 1:
            symbols = [parent.metal_element]
        else:
            raise ValueError(f"Missing aligned metal-site symbols for {parent.pocket_id}")

    examples: list[PocketRecord] = []
    skipped: list[dict[str, str]] = []
    for index, (site_id, observed, coord) in enumerate(zip(site_ids, symbols, parent.metal_coords)):
        example_id = f"{parent.pocket_id}__ION_{index}"
        site_id = tuple(site_id)
        metadata = dict(parent.metadata)
        metadata.update({
            "parent_pocket_id": parent.pocket_id,
            "ion_index": index,
            "ion_site_id": site_id,
            "ion_symbol_observed": observed,
            "metal_site_ids": [site_id],
            "metal_site_symbols": [observed],
            "metal_symbols_observed": [observed],
            "metal_site_coord_map": {site_id: coord},
        })
        nearby = find_pocket_residues_near_metal_cluster(
            parent.residues,
            [MetalAtomRecord(coord=coord, symbol=observed, site_id=site_id)],
            pocket_radius=DEFAULT_POCKET_RADIUS,
        )
        if not nearby:
            skipped.append({"structure_id": parent.structure_id, "pocket_id": example_id,
                            "reason": "no_residues_near_ion"})
            continue
        example = PocketRecord(
            structure_id=parent.structure_id,
            pocket_id=example_id,
            metal_element=observed,
            metal_coords=[coord],
            residues=nearby,
            metadata=metadata,
        )
        label_symbol = observed
        if allowed_site_metal_labels is not None:
            keys = matched_site_keys_for_pocket(example, structure_path, allowed_site_metal_labels)
            if len(keys) != 1:
                skipped.append({"structure_id": parent.structure_id, "pocket_id": example_id,
                                "reason": "ion_not_in_catalytic_summary"})
                continue
            summary_symbol = allowed_site_metal_labels[next(iter(keys))].strip().upper()
            label_symbol = canonicalize_site_metal_resname(summary_symbol) or summary_symbol
            if label_symbol != observed:
                skipped.append({"structure_id": parent.structure_id, "pocket_id": example_id,
                                "reason": "observed_summary_metal_mismatch"})
                continue
            example.metadata["matched_summary_site_metal_types"] = [label_symbol]
        example.metadata["ion_symbol_target"] = label_symbol
        try:
            example.y_metal = map_site_metal_symbols(
                [label_symbol], unsupported_metal_policy=unsupported_metal_policy,
            )
        except ValueError:
            if unsupported_metal_policy != "skip":
                raise
            example.y_metal = None
        if example.y_metal is None:
            skipped.append({"structure_id": parent.structure_id, "pocket_id": example_id,
                            "reason": "unsupported_metal_label"})
            continue
        examples.append(example)
    return examples, skipped
