"""Load training pockets from structure files.

This module:
- discovers structure files under a dataset directory, such as `.pdb` and `.cif`
- resolves the locations of runtime feature inputs, including ESM embeddings and external features
- filters out pockets that should not be used, such as pockets excluded by the summary CSV or missing required supervision
- returns either the loaded pockets alone or the loaded pockets together with a feature-coverage report
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_structures import PocketRecord
from graph.ring_edges import default_ring_output_dir
from project_paths import resolve_ring_features_dir
from training.defaults import DEFAULT_STRUCTURE_DIR, DEFAULT_TRAIN_SUMMARY_CSV
from training.esm_feature_loading import DEFAULT_ESMC_EMBED_DIM
from training.feature_paths import resolve_runtime_feature_paths
from training.labels import assign_ec_targets
from training.parallel_loading import iter_structure_load_results, resolve_load_workers
from training.site_filter import resolve_allowed_site_metal_labels
from training.structure_loading import (
    build_load_report,
    find_structure_files,
    pocket_has_required_supervision,
)


@dataclass(frozen=True)
class PocketLoadResult:
    pockets: List[PocketRecord]
    feature_report: Dict[str, Any]
    ec_label_to_index: Dict[str, int]
    ec_index_to_label: Dict[int, str]


def _assemble_pocket_load_result(
    *,
    pockets: List[PocketRecord],
    structure_files: List[Path],
    feature_fallbacks: List[Dict[str, str]],
    skipped_pockets: List[Dict[str, str]],
    invalid_structures: List[Dict[str, str]],
) -> PocketLoadResult:
    """Package loaded pockets together with a summarized feature-load report."""
    return PocketLoadResult(
        pockets=pockets,
        feature_report=build_load_report(
            pockets=pockets,
            structure_files=structure_files,
            feature_fallbacks=feature_fallbacks,
            skipped_pockets=skipped_pockets,
            invalid_structures=invalid_structures,
        ),
        ec_label_to_index={},
        ec_index_to_label={},
    )


def _structure_load_progress_interval(total: int) -> int:
    """Structures between progress lines; 0 disables reporting.

    Parsing a large structure set is single-threaded and can run for many minutes
    while emitting nothing, which makes a stalled job indistinguishable from a
    healthy one in a captured log. Report periodically for large sets, and stay
    silent for the small sets used by tests and smoke runs.
    """
    raw = os.environ.get("DEEPMZYME_LOAD_PROGRESS_EVERY")
    if raw is not None:
        try:
            return max(0, int(raw))
        except ValueError:
            return 0
    return 250 if total > 200 else 0


def load_labeled_pockets_with_report_from_dir(
    structure_dir: Path,
    max_cases: Optional[int] = None,
    require_full_labels: bool = True,
    required_targets: tuple[str, ...] = ("metal", "ec"),
    summary_csv: Optional[Path] = DEFAULT_TRAIN_SUMMARY_CSV,
    esm_dim: int = DEFAULT_ESMC_EMBED_DIM,
    esm_embeddings_dir: str | Path | None = None,
    require_esm_embeddings: bool = True,
    ring_features_dir: str | Path | None = None,
    external_features_root_dir: str | Path | None = None,
    external_feature_source: str = "auto",
    require_external_features: bool = True,
    unsupported_metal_policy: str = "error",
    invalid_structure_policy: str = "skip",
    ec_label_depth: int = 1,
    ec_label_to_index: Dict[str, int] | None = None,
    metal_eligibility_scheme: str = "active",
    metal_example_unit: str = "pocket",
    allowed_structure_ids: set[str] | None = None,
    load_workers: int | str | None = None,
) -> PocketLoadResult:
    """Load labeled pockets from a structure directory and return them with a load report.

    ``load_workers`` sets parallel structure parsing; see
    ``training.parallel_loading.resolve_load_workers`` for the default (every available
    core, unless overridden by environment or per-machine config). Output is identical
    to serial loading.
    """
    if metal_example_unit == "ion" and required_targets != ("metal",):
        raise ValueError("Ion examples currently require metal-only supervision")
    structure_root = Path(structure_dir)
    structure_files = find_structure_files(structure_root)
    if allowed_structure_ids is not None:
        structure_files = [p for p in structure_files if p.stem in allowed_structure_ids]
        if {p.stem for p in structure_files} != allowed_structure_ids:
            raise ValueError("Unresolved explicit membership structure IDs")
    if not structure_files:
        raise FileNotFoundError(f"No structure files found under {structure_root}")

    allowed_site_metal_labels = resolve_allowed_site_metal_labels(summary_csv)
    embeddings_dir, feature_root_dir = resolve_runtime_feature_paths(
        structure_dir=structure_root,
        esm_embeddings_dir=esm_embeddings_dir,
        external_features_root_dir=external_features_root_dir,
        external_feature_source=external_feature_source,
    )
    resolved_ring_features_dir = (
        resolve_ring_features_dir(str(ring_features_dir), create=False)
        if ring_features_dir is not None
        else default_ring_output_dir()
    )

    raw_pockets: List[PocketRecord] = []
    feature_fallbacks: List[Dict[str, str]] = []
    skipped_pockets: List[Dict[str, str]] = []
    invalid_structures: List[Dict[str, str]] = []

    workers, workers_source = resolve_load_workers(load_workers)
    progress_every = _structure_load_progress_interval(len(structure_files))
    load_started_at = time.monotonic()
    if progress_every:
        print(
            f"[LOAD] Parsing {len(structure_files)} structures from {structure_root} "
            f"(metal_example_unit={metal_example_unit}, workers={workers} from {workers_source})",
            flush=True,
        )

    load_kwargs = dict(
        structure_root=structure_root,
        allowed_site_metal_labels=allowed_site_metal_labels,
        esm_dim=esm_dim,
        embeddings_dir=embeddings_dir,
        require_esm_embeddings=require_esm_embeddings,
        ring_features_dir=resolved_ring_features_dir,
        feature_root_dir=feature_root_dir,
        external_feature_source=external_feature_source,
        require_external_features=require_external_features,
        unsupported_metal_policy=unsupported_metal_policy,
        ec_label_depth=ec_label_depth,
        metal_example_unit=metal_example_unit,
    )
    load_results = iter_structure_load_results(structure_files, load_kwargs, workers=workers)
    for loaded_count, (structure_path, (status, outcome)) in enumerate(load_results, start=1):
        if progress_every and (loaded_count % progress_every == 0 or loaded_count == len(structure_files)):
            elapsed = time.monotonic() - load_started_at
            rate = loaded_count / elapsed if elapsed > 0 else 0.0
            remaining = (len(structure_files) - loaded_count) / rate if rate > 0 else 0.0
            print(
                f"[LOAD] {loaded_count}/{len(structure_files)} structures "
                f"({rate:.1f}/s, {elapsed / 60:.1f} min elapsed, "
                f"~{remaining / 60:.1f} min remaining), {len(raw_pockets)} examples so far",
                flush=True,
            )
        if status == "invalid":
            if invalid_structure_policy != "skip":
                raise outcome
            invalid_structures.append(
                {
                    "structure_path": str(structure_path),
                    "reason": "invalid_structure",
                    "detail": str(outcome),
                }
            )
            continue
        structure_pockets, structure_fallbacks, structure_skipped_pockets = outcome
        feature_fallbacks.extend(structure_fallbacks)
        skipped_pockets.extend(structure_skipped_pockets)

        raw_pockets.extend(structure_pockets)

    if progress_every:
        print(
            f"[LOAD] Finished parsing {len(structure_files)} structures in "
            f"{(time.monotonic() - load_started_at) / 60:.1f} min: "
            f"{len(raw_pockets)} examples, {len(invalid_structures)} invalid structures",
            flush=True,
        )

    ec_label_to_index, ec_index_to_label = assign_ec_targets(
        raw_pockets,
        depth=ec_label_depth,
        token_to_index=ec_label_to_index,
    )
    pockets: List[PocketRecord] = []
    for pocket in raw_pockets:
        if require_full_labels and not pocket_has_required_supervision(
            pocket,
            required_targets=required_targets,
            metal_eligibility_scheme=metal_eligibility_scheme,
        ):
            skipped_pockets.append(
                {
                    "structure_id": pocket.structure_id,
                    "pocket_id": pocket.pocket_id,
                    "reason": "missing_required_supervision",
                }
            )
            continue
        pockets.append(pocket)
        if max_cases is not None and len(pockets) >= max_cases:
            result = _assemble_pocket_load_result(
                pockets=pockets,
                structure_files=structure_files,
                feature_fallbacks=feature_fallbacks,
                skipped_pockets=skipped_pockets,
                invalid_structures=invalid_structures,
            )
            return PocketLoadResult(
                pockets=result.pockets,
                feature_report=result.feature_report,
                ec_label_to_index=ec_label_to_index,
                ec_index_to_label=ec_index_to_label,
            )

    if not pockets:
        if require_full_labels:
            required_target_list = ", ".join(required_targets)
            raise ValueError(
                "No metal-centered pockets with required supervision "
                f"({required_target_list}) were extracted from {structure_root}"
            )
        raise ValueError(f"No metal-centered pockets were extracted from {structure_root}")

    result = _assemble_pocket_load_result(
        pockets=pockets,
        structure_files=structure_files,
        feature_fallbacks=feature_fallbacks,
        skipped_pockets=skipped_pockets,
        invalid_structures=invalid_structures,
    )
    return PocketLoadResult(
        pockets=result.pockets,
        feature_report=result.feature_report,
        ec_label_to_index=ec_label_to_index,
        ec_index_to_label=ec_index_to_label,
    )


def load_training_pockets_with_report_from_dir(
    structure_dir: Path,
    require_full_labels: bool = True,
    required_targets: tuple[str, ...] = ("metal", "ec"),
    summary_csv: Optional[Path] = DEFAULT_TRAIN_SUMMARY_CSV,
    esm_dim: int = DEFAULT_ESMC_EMBED_DIM,
    esm_embeddings_dir: str | Path | None = None,
    require_esm_embeddings: bool = True,
    ring_features_dir: str | Path | None = None,
    external_features_root_dir: str | Path | None = None,
    external_feature_source: str = "auto",
    require_external_features: bool = True,
    unsupported_metal_policy: str = "error",
    invalid_structure_policy: str = "skip",
    ec_label_depth: int = 1,
    ec_label_to_index: Dict[str, int] | None = None,
    metal_eligibility_scheme: str = "active",
    metal_example_unit: str = "pocket",
    allowed_structure_ids: set[str] | None = None,
    load_workers: int | str | None = None,
) -> PocketLoadResult:
    """Load the full training set from a structure directory with a load report."""
    return load_labeled_pockets_with_report_from_dir(
        structure_dir=structure_dir,
        max_cases=None,
        require_full_labels=require_full_labels,
        required_targets=required_targets,
        summary_csv=summary_csv,
        esm_dim=esm_dim,
        esm_embeddings_dir=esm_embeddings_dir,
        require_esm_embeddings=require_esm_embeddings,
        ring_features_dir=ring_features_dir,
        external_features_root_dir=external_features_root_dir,
        external_feature_source=external_feature_source,
        require_external_features=require_external_features,
        unsupported_metal_policy=unsupported_metal_policy,
        invalid_structure_policy=invalid_structure_policy,
        ec_label_depth=ec_label_depth,
        ec_label_to_index=ec_label_to_index,
        metal_eligibility_scheme=metal_eligibility_scheme,
        metal_example_unit=metal_example_unit,
        allowed_structure_ids=allowed_structure_ids,
        load_workers=load_workers,
    )


def load_smoke_test_pockets_from_dir(
    structure_dir: Path,
    max_cases: int = 4,
    require_full_labels: bool = True,
    required_targets: tuple[str, ...] = ("metal", "ec"),
    summary_csv: Optional[Path] = DEFAULT_TRAIN_SUMMARY_CSV,
    esm_dim: int = DEFAULT_ESMC_EMBED_DIM,
    esm_embeddings_dir: str | Path | None = None,
    require_esm_embeddings: bool = False,
    ring_features_dir: str | Path | None = None,
    external_features_root_dir: str | Path | None = None,
    external_feature_source: str = "auto",
    require_external_features: bool = False,
    unsupported_metal_policy: str = "error",
    invalid_structure_policy: str = "skip",
    ec_label_depth: int = 1,
    ec_label_to_index: Dict[str, int] | None = None,
    metal_example_unit: str = "pocket",
) -> List[PocketRecord]:
    """Load a small pocket subset for smoke tests, with optional feature requirements relaxed."""
    return load_labeled_pockets_with_report_from_dir(
        structure_dir=structure_dir,
        max_cases=max_cases,
        require_full_labels=require_full_labels,
        required_targets=required_targets,
        summary_csv=summary_csv,
        esm_dim=esm_dim,
        esm_embeddings_dir=esm_embeddings_dir,
        require_esm_embeddings=require_esm_embeddings,
        ring_features_dir=ring_features_dir,
        external_features_root_dir=external_features_root_dir,
        external_feature_source=external_feature_source,
        require_external_features=require_external_features,
        unsupported_metal_policy=unsupported_metal_policy,
        invalid_structure_policy=invalid_structure_policy,
        ec_label_depth=ec_label_depth,
        ec_label_to_index=ec_label_to_index,
        metal_example_unit=metal_example_unit,
    ).pockets
