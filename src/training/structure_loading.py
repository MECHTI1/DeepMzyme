from __future__ import annotations

from pathlib import Path
from typing import Any

from data_structures import PocketRecord
from graph.ring_edges import canonical_ring_edges_output_path, ring_edges_output_path
from graph.structure_parsing import (
    NO_SELECTED_SITES_REASON, POCKET_GROUPING_CLUSTER, POCKET_GROUPING_STRUCTURE, SELECTED_SITE_EXTRACTION,
    SELECTED_SITE_POCKET_GROUPINGS, extract_metal_pockets_from_structure, extract_selected_site_pockets_from_structure,
    parse_structure_file,
)
from label_schemes import map_site_metal_symbols, map_site_metal_symbols_for_scheme
from structure_store import STRUCTURE_MANIFEST_FILENAME, read_structure_manifest
from training.feature_sources import (
    attach_structure_features_to_pocket,
    build_feature_load_report,
    load_structure_feature_sources,
)
from training.labels import (
    infer_metal_target_class_from_pocket,
    parse_ec_label_token_from_structure_path,
    parse_structure_ec_numbers,
)
from training.metal_examples import metal_ion_examples
from training.source_cohort import CohortBindingError, binding_from_payload, build_cohort_ion_examples
from training.site_filter import (
    AllowedSiteMetalLabels, matched_site_metal_types, pocket_matches_allowed_sites, selected_site_ids_for_structure,
)


class StructureLoadError(ValueError):
    """Raised when a structure cannot be parsed or aligned to required features."""


def pocket_has_required_supervision(
    pocket: PocketRecord,
    required_targets: tuple[str, ...] = ("metal", "ec"),
    metal_eligibility_scheme: str = "active",
) -> bool:
    if metal_eligibility_scheme not in {"active", "six_class"}:
        raise ValueError(f"Unsupported metal eligibility scheme: {metal_eligibility_scheme!r}")
    if "metal" in required_targets and metal_eligibility_scheme == "six_class":
        symbols = (pocket.metadata.get("matched_summary_site_metal_types")
                   or pocket.metadata.get("metal_symbols_observed") or [pocket.metal_element])
        if map_site_metal_symbols_for_scheme(symbols, scheme_name="six_class") is None:
            return False
    for target_name in required_targets:
        if target_name == "metal" and pocket.y_metal is None:
            return False
        if target_name == "ec" and pocket.y_ec is None:
            return False
    return True


def is_auxiliary_structure_file(path: Path, structure_root: Path) -> bool:
    try:
        relative_parts = path.relative_to(structure_root).parts
    except ValueError:
        relative_parts = path.parts

    if len(relative_parts) > 2:
        return True
    return path.parent.name == path.stem


def find_structure_files(structure_dir: Path) -> list[Path]:
    manifest_path = structure_dir / STRUCTURE_MANIFEST_FILENAME
    if manifest_path.is_file():
        return [reference.path for reference in read_structure_manifest(structure_dir)]

    structure_files: list[Path] = []
    for pattern in ("*.pdb", "*.cif", "*.mmcif"):
        structure_files.extend(structure_dir.rglob(pattern))
    return sorted(
        path for path in structure_files if path.is_file() and not is_auxiliary_structure_file(path, structure_dir)
    )


def build_load_report(
    *,
    pockets: list[PocketRecord],
    structure_files: list[Path],
    feature_fallbacks: list[dict[str, str]],
    skipped_pockets: list[dict[str, str]],
    invalid_structures: list[dict[str, str]],
) -> dict[str, Any]:
    return build_feature_load_report(
        pockets=pockets,
        total_structure_files=len(structure_files),
        feature_fallbacks=feature_fallbacks,
        skipped_pockets=skipped_pockets,
        invalid_structures=invalid_structures,
    )


def load_structure_pockets(
    *,
    structure_path: Path,
    structure_root: Path,
    allowed_site_metal_labels: AllowedSiteMetalLabels | None,
    esm_dim: int,
    embeddings_dir: Path,
    require_esm_embeddings: bool,
    ring_features_dir: Path | None = None,
    feature_root_dir: Path,
    external_feature_source: str,
    require_external_features: bool,
    unsupported_metal_policy: str = "error",
    ec_label_depth: int = 1,
    metal_example_unit: str = "pocket",
    cohort_bindings: dict[str, tuple] | None = None,
    feature_inventory_sha256: str | None = None,
    site_extraction: str = "component",
    selected_site_pocket_grouping: str = POCKET_GROUPING_CLUSTER,
) -> tuple[list[PocketRecord], list[dict[str, str]], list[dict[str, str]]]:
    """Parse one structure into labeled examples.

    ``site_extraction="selected_sites"`` builds pockets only around the summary's
    selected sites, so unselected metals in the file shape no pocket (see
    ``extract_selected_site_pockets_from_structure``); the default keeps the
    component extraction followed by the summary filter.
    ``selected_site_pocket_grouping="structure"`` (selected sites only) makes the
    selected sites of the structure one pocket instead of one per cluster.

    ``cohort_bindings`` (structure stem -> frozen binding tuples) switches to the
    source-row cohort path: one example per bound ion, no summary-key matching.
    ``feature_inventory_sha256`` is not used here; it only makes parse-cache
    entries specific to one immutable feature inventory.
    """
    if metal_example_unit not in {"pocket", "ion"}:
        raise ValueError(f"Unsupported metal example unit: {metal_example_unit!r}")
    if site_extraction not in {"component", SELECTED_SITE_EXTRACTION}:
        raise ValueError(f"Unsupported site extraction: {site_extraction!r}")
    selected_mode = site_extraction == SELECTED_SITE_EXTRACTION
    if selected_mode and (
        allowed_site_metal_labels is None or metal_example_unit != "pocket" or cohort_bindings is not None
    ):
        raise ValueError("Selected-site extraction needs a site summary, pocket examples and no source cohort")
    if selected_site_pocket_grouping not in SELECTED_SITE_POCKET_GROUPINGS:
        raise ValueError(f"Unsupported selected-site pocket grouping: {selected_site_pocket_grouping!r}")
    structure_grouping = selected_site_pocket_grouping == POCKET_GROUPING_STRUCTURE
    if structure_grouping and not selected_mode:
        raise ValueError("Structure pocket grouping needs selected-site extraction")
    if cohort_bindings is not None:
        return _load_cohort_structure_pockets(
            structure_path=structure_path,
            structure_root=structure_root,
            bindings=cohort_bindings.get(structure_path.stem, ()),
            esm_dim=esm_dim,
            embeddings_dir=embeddings_dir,
            require_esm_embeddings=require_esm_embeddings,
            feature_root_dir=feature_root_dir,
            external_feature_source=external_feature_source,
            require_external_features=require_external_features,
            ec_label_depth=ec_label_depth,
        )
    selected_report = None
    try:
        structure = parse_structure_file(str(structure_path), structure_id=structure_path.stem)
        if selected_mode:
            selected_ids = selected_site_ids_for_structure(structure_path, allowed_site_metal_labels)
            if not selected_ids:
                return [], [], [{"structure_id": structure_path.stem, "pocket_id": "", "reason": NO_SELECTED_SITES_REASON}]
            extracted_pockets, selected_report = extract_selected_site_pockets_from_structure(
                structure, structure_id=structure_path.stem, selected_site_ids=selected_ids,
                pocket_grouping=selected_site_pocket_grouping,
            )
        else:
            extracted_pockets = extract_metal_pockets_from_structure(structure, structure_id=structure_path.stem)
    except Exception as exc:
        raise StructureLoadError(f"Failed to parse structure {structure_path}: {exc}") from exc
    if not extracted_pockets:
        if selected_report is not None:
            return [], [], [{"structure_id": structure_path.stem, "pocket_id": "",
                             "reason": "selected_sites_without_pocket"}]
        return [], [], []
    if selected_report is not None:
        compact_report = {key: selected_report[key] for key in (
            "metal_records_total", "metal_records_selected", "metal_records_dropped", "selected_cluster_count",
            "components_with_unselected_metals_count", "split_component_count", "residue_count_delta_total",
        )}
        compact_report["selected_sites_without_pocket"] = len(selected_report["selected_sites_without_pocket"])
        if structure_grouping:
            compact_report["cluster_grouping_cluster_count"] = selected_report["cluster_grouping_cluster_count"]
            compact_report["selected_sites_without_residues"] = len(selected_report["selected_sites_without_residues"])
        for pocket in extracted_pockets:
            pocket.metadata["site_extraction_report"] = compact_report

    feature_fallbacks: list[dict[str, str]] = []
    skipped_pockets: list[dict[str, str]] = []
    try:
        feature_sources = load_structure_feature_sources(
            structure=structure,
            structure_path=structure_path,
            structure_root=structure_root,
            embeddings_dir=embeddings_dir,
            require_esm_embeddings=require_esm_embeddings,
            feature_root_dir=feature_root_dir,
            external_feature_source=external_feature_source,
            require_external_features=require_external_features,
            feature_fallbacks=feature_fallbacks,
        )
    except ValueError as exc:
        raise StructureLoadError(str(exc)) from exc
    ec_label_token = parse_ec_label_token_from_structure_path(structure_path, depth=ec_label_depth)
    ec_numbers = list(parse_structure_ec_numbers(structure_path.stem))

    kept_pockets: list[PocketRecord] = []
    for pocket in extracted_pockets:
        pocket.metadata["source_path"] = str(structure_path)
        expected_ring_path = (
            ring_edges_output_path(ring_features_dir, structure_path)
            if ring_features_dir is not None
            else canonical_ring_edges_output_path(structure_path)
        )
        pocket.metadata.setdefault(
            "ring_edges_expected_path",
            str(expected_ring_path),
        )
        try:
            attach_structure_features_to_pocket(
                pocket,
                feature_sources=feature_sources,
                esm_dim=esm_dim,
                require_esm_embeddings=require_esm_embeddings,
                require_external_features=require_external_features,
                structure_path=structure_path,
            )
        except ValueError as exc:
            raise StructureLoadError(str(exc)) from exc

        if metal_example_unit == "ion":
            try:
                ion_pockets, ion_skipped = metal_ion_examples(
                    pocket, structure_path, allowed_site_metal_labels,
                    unsupported_metal_policy=unsupported_metal_policy,
                )
            except ValueError as exc:
                raise StructureLoadError(str(exc)) from exc
            skipped_pockets.extend(ion_skipped)
            for ion in ion_pockets:
                ion.metadata["ec_label_depth"] = ec_label_depth
                ion.metadata["ec_numbers"] = ec_numbers
                if ec_label_token is not None:
                    ion.metadata["ec_label_token"] = ec_label_token
            kept_pockets.extend(ion_pockets)
            continue

        matched_summary_metal_types: set[str] = set()
        if selected_mode and not pocket_matches_allowed_sites(pocket, structure_path, allowed_site_metal_labels):
            raise StructureLoadError(f"Selected-site pocket {pocket.pocket_id} matches no summary site")
        if allowed_site_metal_labels is not None and not pocket_matches_allowed_sites(
            pocket,
            structure_path,
            allowed_site_metal_labels,
        ):
            skipped_pockets.append(
                {
                    "structure_id": pocket.structure_id,
                    "pocket_id": pocket.pocket_id,
                    "reason": "filtered_by_summary_sites",
                }
            )
            continue
        if allowed_site_metal_labels is not None:
            matched_summary_metal_types = matched_site_metal_types(
                pocket,
                structure_path,
                allowed_site_metal_labels,
            )
            if matched_summary_metal_types:
                pocket.metadata["matched_summary_site_metal_types"] = sorted(matched_summary_metal_types)

        try:
            if matched_summary_metal_types:
                pocket.y_metal = map_site_metal_symbols(
                    sorted(matched_summary_metal_types),
                    unsupported_metal_policy=unsupported_metal_policy,
                )
            else:
                pocket.y_metal = infer_metal_target_class_from_pocket(
                    pocket,
                    unsupported_metal_policy=unsupported_metal_policy,
                )
        except ValueError:
            skipped_pockets.append(
                {
                    "structure_id": pocket.structure_id,
                    "pocket_id": pocket.pocket_id,
                    "reason": "unsupported_metal_label",
                }
            )
            raise
        if pocket.y_metal is None and unsupported_metal_policy == "skip":
            skipped_pockets.append(
                {
                    "structure_id": pocket.structure_id,
                    "pocket_id": pocket.pocket_id,
                    "reason": "unsupported_metal_label",
                }
            )
            continue
        pocket.metadata["ec_label_depth"] = ec_label_depth
        pocket.metadata["ec_numbers"] = ec_numbers
        if ec_label_token is not None:
            pocket.metadata["ec_label_token"] = ec_label_token
        kept_pockets.append(pocket)

    return kept_pockets, feature_fallbacks, skipped_pockets


def _load_cohort_structure_pockets(
    *,
    structure_path: Path,
    structure_root: Path,
    bindings: tuple,
    esm_dim: int,
    embeddings_dir: Path,
    require_esm_embeddings: bool,
    feature_root_dir: Path,
    external_feature_source: str,
    require_external_features: bool,
    ec_label_depth: int,
) -> tuple[list[PocketRecord], list[dict[str, str]], list[dict[str, str]]]:
    """Frozen-cohort ion examples. Every failure is fatal (no silent skips)."""
    if not bindings:
        return [], [], []
    resolved = [binding_from_payload(item) for item in bindings]
    try:
        structure = parse_structure_file(str(structure_path), structure_id=structure_path.stem)
    except Exception as exc:
        raise CohortBindingError(f"Failed to parse cohort structure {structure_path}: {exc}") from exc
    examples = build_cohort_ion_examples(
        structure,
        structure_id=structure_path.stem,
        structure_path=structure_path,
        bindings=resolved,
    )
    feature_fallbacks: list[dict[str, str]] = []
    try:
        feature_sources = load_structure_feature_sources(
            structure=structure,
            structure_path=structure_path,
            structure_root=structure_root,
            embeddings_dir=embeddings_dir,
            require_esm_embeddings=require_esm_embeddings,
            feature_root_dir=feature_root_dir,
            external_feature_source=external_feature_source,
            require_external_features=require_external_features,
            feature_fallbacks=feature_fallbacks,
        )
        # Features are attached to retained ion examples only, never to other
        # pockets of the structure, so an unused remote chain cannot fail a load.
        for example in examples:
            attach_structure_features_to_pocket(
                example,
                feature_sources=feature_sources,
                esm_dim=esm_dim,
                require_esm_embeddings=require_esm_embeddings,
                require_external_features=require_external_features,
                structure_path=structure_path,
            )
    except ValueError as exc:
        raise CohortBindingError(f"Required features unavailable for cohort structure {structure_path}: {exc}") from exc
    ec_label_token = parse_ec_label_token_from_structure_path(structure_path, depth=ec_label_depth)
    ec_numbers = list(parse_structure_ec_numbers(structure_path.stem))
    for example in examples:
        example.y_metal = map_site_metal_symbols([example.metal_element], unsupported_metal_policy="error")
        example.metadata["ec_label_depth"] = ec_label_depth
        example.metadata["ec_numbers"] = ec_numbers
        if ec_label_token is not None:
            example.metadata["ec_label_token"] = ec_label_token
    return examples, feature_fallbacks, []
