from __future__ import annotations

from typing import Any

from data_structures import PocketRecord
from graph.construction import pocket_to_pyg_data
from label_schemes import METAL_TARGET_LABELS
from training.config import TrainConfig
from training.feature_sources import build_pocket_feature_coverage
from training.splits import PocketSplit, pocket_split_key


def validate_graphs(
    pockets: list[PocketRecord],
    config: TrainConfig,
    precomputed_graphs: list[Any] | None = None,
) -> None:
    if precomputed_graphs is not None:
        if len(precomputed_graphs) != len(pockets):
            raise ValueError("Graph preflight received mismatched precomputed data.")
        return

    for pocket in pockets:
        try:
            pocket_to_pyg_data(
                pocket,
                esm_dim=config.esm_dim,
                edge_radius=config.edge_radius,
                require_ring_edges=config.require_ring_edges,
                node_feature_set=config.node_feature_set,
            )
        except Exception as exc:
            raise ValueError(f"Graph preflight failed for pocket {pocket.pocket_id!r}: {exc}") from exc


def missing_label_names(
    label_ids: set[int],
    label_map: dict[int, str],
) -> list[str]:
    return [label_name for label_idx, label_name in label_map.items() if label_idx not in label_ids]


def run_preflight_checks(
    split: PocketSplit,
    config: TrainConfig,
    *,
    ec_label_map: dict[int, str],
    train_graphs: list[Any] | None = None,
    val_graphs: list[Any] | None = None,
) -> dict[str, object]:
    has_validation = config.val_fraction > 0.0 or config.n_folds is not None
    if not split.train_pockets:
        raise ValueError("Preflight failed: training split is empty.")
    if has_validation and not split.val_pockets:
        raise ValueError("Preflight failed: validation split is empty, but --val-fraction > 0.")
    if not has_validation and split.val_pockets:
        raise ValueError("Preflight failed: validation pockets exist, but --val-fraction is 0.")

    empty_train = [pocket.pocket_id for pocket in split.train_pockets if not pocket.residues]
    if empty_train:
        raise ValueError(f"Preflight failed: training pockets without residues: {empty_train[:5]}")

    empty_val = [pocket.pocket_id for pocket in split.val_pockets if not pocket.residues]
    if empty_val:
        raise ValueError(f"Preflight failed: validation pockets without residues: {empty_val[:5]}")

    train_metal_ids = {
        int(pocket.y_metal) for pocket in split.train_pockets if pocket.y_metal is not None and int(pocket.y_metal) in METAL_TARGET_LABELS
    }
    train_ec_ids = {int(pocket.y_ec) for pocket in split.train_pockets if pocket.y_ec is not None and int(pocket.y_ec) in ec_label_map}
    val_metal_ids = {int(pocket.y_metal) for pocket in split.val_pockets if pocket.y_metal is not None and int(pocket.y_metal) in METAL_TARGET_LABELS}
    val_ec_ids = {int(pocket.y_ec) for pocket in split.val_pockets if pocket.y_ec is not None and int(pocket.y_ec) in ec_label_map}

    if config.task in ("joint", "metal") and len(train_metal_ids) < 2:
        raise ValueError("Preflight failed: training split contains fewer than 2 metal classes.")
    if config.task in ("joint", "ec") and len(train_ec_ids) < 2:
        raise ValueError("Preflight failed: training split contains fewer than 2 EC classes.")
    if config.require_all_task_classes:
        missing_train_metal = missing_label_names(train_metal_ids, METAL_TARGET_LABELS)
        missing_train_ec = missing_label_names(train_ec_ids, ec_label_map)
        if config.task in ("joint", "metal") and missing_train_metal:
            raise ValueError(
                "Preflight failed: training split is missing required metal classes: "
                f"{', '.join(missing_train_metal)}."
            )
        if config.task in ("joint", "ec") and missing_train_ec:
            raise ValueError(
                "Preflight failed: training split is missing required EC classes: "
                f"{', '.join(missing_train_ec)}."
            )

    overlap = sorted(
        {pocket_split_key(pocket, config.split_by) for pocket in split.train_pockets}.intersection(
            pocket_split_key(pocket, config.split_by) for pocket in split.val_pockets
        )
    )
    if overlap:
        raise ValueError(
            "Preflight failed: train/validation leakage detected under "
            f"--split-by {config.split_by!r}: {overlap[:5]}"
        )

    validate_graphs(split.train_pockets, config, precomputed_graphs=train_graphs)
    validate_graphs(split.val_pockets, config, precomputed_graphs=val_graphs)

    train_feature_coverage = build_pocket_feature_coverage(split.train_pockets)
    val_feature_coverage = build_pocket_feature_coverage(split.val_pockets)
    warnings: list[str] = []
    if config.require_all_task_classes and has_validation and split.val_pockets:
        missing_val_metal = missing_label_names(val_metal_ids, METAL_TARGET_LABELS)
        missing_val_ec = missing_label_names(val_ec_ids, ec_label_map)
        if config.task in ("joint", "metal") and missing_val_metal:
            warnings.append(
                "Validation split is missing metal classes: "
                f"{', '.join(missing_val_metal)}."
            )
        if config.task in ("joint", "ec") and missing_val_ec:
            warnings.append(
                "Validation split is missing EC classes: "
                f"{', '.join(missing_val_ec)}."
            )

    if config.task in ("joint", "metal") and has_validation and len(val_metal_ids) < 2:
        warnings.append("Validation split contains fewer than 2 metal classes.")
    if config.task in ("joint", "ec") and has_validation and len(val_ec_ids) < 2:
        warnings.append("Validation split contains fewer than 2 EC classes.")
    if train_feature_coverage["esm_residue_coverage"] == 0.0:
        warnings.append("Training split has no ESM residue coverage.")
    if train_feature_coverage["external_feature_residue_coverage"] == 0.0:
        warnings.append("Training split has no external feature residue coverage.")
    if has_validation and split.val_pockets and val_feature_coverage["esm_residue_coverage"] == 0.0:
        warnings.append("Validation split has no ESM residue coverage.")
    if (
        has_validation
        and split.val_pockets
        and val_feature_coverage["external_feature_residue_coverage"] == 0.0
    ):
        warnings.append("Validation split has no external feature residue coverage.")

    return {"warnings": warnings}
