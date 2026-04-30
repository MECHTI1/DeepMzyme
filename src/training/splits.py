from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import WeightedRandomSampler

from data_structures import PocketRecord
from label_schemes import METAL_SYMBOL_TO_TARGET, METAL_TARGET_LABELS
from training.config import TrainConfig, VALID_EC_GROUP_WEIGHTING_CHOICES, VALID_SPLIT_BY_CHOICES
from training.feature_sources import build_pocket_feature_coverage
from training.labels import parse_structure_identity

EC_SAMPLE_WEIGHT_METADATA_KEY = "ec_sample_weight"
EC_GROUP_KEY_METADATA_KEY = "ec_group_key"
EC_GROUP_ID_METADATA_KEY = "ec_group_id"


@dataclass(frozen=True)
class PocketSplit:
    train_pockets: list[PocketRecord]
    val_pockets: list[PocketRecord]


def validate_split_by(split_by: str) -> str:
    if split_by not in VALID_SPLIT_BY_CHOICES:
        raise ValueError(
            f"Unsupported --split-by value: {split_by!r}. "
            f"Expected one of: {', '.join(repr(choice) for choice in VALID_SPLIT_BY_CHOICES)}."
        )
    return split_by


def pocket_split_key(pocket: PocketRecord, split_by: str) -> str:
    pdbid, chain, _ec = parse_structure_identity(pocket.structure_id)
    if split_by == "structure_id":
        return pocket.structure_id
    if split_by == "pdbid":
        return pdbid
    if split_by == "pdbid_chain":
        return f"{pdbid}__chain_{chain}"
    if split_by == "pocket_id":
        return pocket.pocket_id
    raise AssertionError(f"Unhandled split_by value: {split_by!r}")


def validate_ec_group_weighting(mode: str) -> str:
    if mode not in VALID_EC_GROUP_WEIGHTING_CHOICES:
        raise ValueError(
            f"Unsupported --ec-group-weighting value: {mode!r}. "
            f"Expected one of: {', '.join(repr(choice) for choice in VALID_EC_GROUP_WEIGHTING_CHOICES)}."
        )
    return mode


def ec_grouping_mode_for_metrics(weighting_mode: str) -> str:
    weighting_mode = validate_ec_group_weighting(weighting_mode)
    return "structure_id" if weighting_mode == "none" else weighting_mode


def ec_group_key_for_pocket(pocket: PocketRecord, mode: str) -> str:
    mode = validate_ec_group_weighting(mode)
    if mode == "none":
        raise ValueError("'none' is not a concrete EC grouping mode.")
    pdbid, chain, _ec = parse_structure_identity(pocket.structure_id)
    if mode == "structure_id":
        return str(pocket.structure_id)
    if mode == "pdbid_chain":
        return f"{pdbid}__chain_{chain}"
    if mode == "pdbid":
        return str(pdbid)
    raise AssertionError(f"Unhandled EC grouping mode: {mode!r}")


def ec_group_size_distribution(pockets: list[PocketRecord], mode: str) -> dict[str, int]:
    if not pockets:
        return {}
    group_counts: Counter[str] = Counter()
    for pocket in pockets:
        if pocket.y_ec is None:
            continue
        group_counts[ec_group_key_for_pocket(pocket, mode)] += 1
    size_counts: Counter[int] = Counter(group_counts.values())
    return {str(size): int(count) for size, count in sorted(size_counts.items())}


def count_ec_groups(pockets: list[PocketRecord], mode: str) -> int:
    return len(
        {
            ec_group_key_for_pocket(pocket, mode)
            for pocket in pockets
            if pocket.y_ec is not None
        }
    )


def assign_ec_group_metadata(
    pockets: list[PocketRecord],
    *,
    weighting_mode: str,
) -> None:
    weighting_mode = validate_ec_group_weighting(weighting_mode)
    metric_mode = ec_grouping_mode_for_metrics(weighting_mode)

    metric_keys = sorted(
        {
            ec_group_key_for_pocket(pocket, metric_mode)
            for pocket in pockets
            if pocket.y_ec is not None
        }
    )
    metric_key_to_index = {key: idx for idx, key in enumerate(metric_keys)}

    weighting_counts: Counter[str] = Counter()
    if weighting_mode != "none":
        for pocket in pockets:
            if pocket.y_ec is None:
                continue
            weighting_counts[ec_group_key_for_pocket(pocket, weighting_mode)] += 1

    for pocket in pockets:
        pocket.metadata[EC_SAMPLE_WEIGHT_METADATA_KEY] = 1.0
        if pocket.y_ec is None:
            pocket.metadata[EC_GROUP_KEY_METADATA_KEY] = None
            pocket.metadata[EC_GROUP_ID_METADATA_KEY] = -1
            continue

        metric_key = ec_group_key_for_pocket(pocket, metric_mode)
        pocket.metadata[EC_GROUP_KEY_METADATA_KEY] = metric_key
        pocket.metadata[EC_GROUP_ID_METADATA_KEY] = int(metric_key_to_index[metric_key])
        if weighting_mode != "none":
            weight_key = ec_group_key_for_pocket(pocket, weighting_mode)
            pocket.metadata[EC_SAMPLE_WEIGHT_METADATA_KEY] = 1.0 / float(weighting_counts[weight_key])


def split_pockets(
    pockets: list[PocketRecord],
    val_fraction: float,
    split_by: str,
    seed: int,
    task: str = "joint",
) -> PocketSplit:
    split_by = validate_split_by(split_by)
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"--val-fraction must be in [0, 1), got {val_fraction}")
    if val_fraction == 0.0:
        return PocketSplit(train_pockets=pockets, val_pockets=[])

    grouped: dict[str, list[PocketRecord]] = {}
    for pocket in pockets:
        grouped.setdefault(pocket_split_key(pocket, split_by), []).append(pocket)

    group_items = list(grouped.items())
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(group_items), generator=generator).tolist()
    shuffled = [group_items[idx] for idx in order]
    shuffled.sort(key=lambda item: len(item[1]), reverse=True)

    target_val_size = max(1, int(round(len(pockets) * val_fraction)))
    val_pockets: list[PocketRecord] = []
    train_pockets: list[PocketRecord] = []
    val_count = 0
    remaining_pocket_count = len(pockets)
    desired_val_label_counts = desired_label_counts_for_split(pockets, task=task, val_fraction=val_fraction)
    current_val_label_counts: dict[str, int] = {}

    for _group_key, group_pockets in shuffled:
        group_size = len(group_pockets)
        remaining_pocket_count -= group_size
        must_assign_to_val = val_count < target_val_size and (val_count + remaining_pocket_count) < target_val_size
        proposed_val_label_counts = merge_label_counts(
            current_val_label_counts,
            label_counts_for_pockets(group_pockets, task=task),
        )
        current_penalty = val_assignment_penalty(
            val_count=val_count,
            target_val_size=target_val_size,
            current_label_counts=current_val_label_counts,
            desired_label_counts=desired_val_label_counts,
        )
        proposed_penalty = val_assignment_penalty(
            val_count=val_count + group_size,
            target_val_size=target_val_size,
            current_label_counts=proposed_val_label_counts,
            desired_label_counts=desired_val_label_counts,
        )

        if must_assign_to_val or (val_count < target_val_size and proposed_penalty <= current_penalty):
            val_pockets.extend(group_pockets)
            val_count += group_size
            current_val_label_counts = proposed_val_label_counts
        else:
            train_pockets.extend(group_pockets)

    if not train_pockets or not val_pockets:
        raise ValueError(
            "Validation split produced an empty train or validation set. "
            "Adjust --val-fraction or --split-by."
        )

    return PocketSplit(train_pockets=train_pockets, val_pockets=val_pockets)


def split_pockets_k_fold(
    pockets: list[PocketRecord],
    *,
    n_folds: int,
    fold_index: int,
    split_by: str,
    seed: int,
    task: str = "joint",
) -> PocketSplit:
    split_by = validate_split_by(split_by)
    if n_folds < 2:
        raise ValueError(f"--n-folds must be at least 2, got {n_folds}")
    if not 0 <= fold_index < n_folds:
        raise ValueError(f"--fold-index must be in [0, {n_folds - 1}], got {fold_index}")
    if len(pockets) < n_folds:
        raise ValueError(f"Cannot split {len(pockets)} pockets into {n_folds} folds.")

    grouped: dict[str, list[PocketRecord]] = {}
    for pocket in pockets:
        grouped.setdefault(pocket_split_key(pocket, split_by), []).append(pocket)
    if len(grouped) < n_folds:
        raise ValueError(f"Cannot split {len(grouped)} {split_by} groups into {n_folds} folds.")

    group_items = list(grouped.items())
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(group_items), generator=generator).tolist()
    shuffled = [group_items[idx] for idx in order]
    shuffled.sort(key=lambda item: len(item[1]), reverse=True)

    target_fold_size = max(1, int(round(len(pockets) / n_folds)))
    desired_label_counts = {
        key: count / n_folds
        for key, count in label_counts_for_pockets(pockets, task).items()
    }
    fold_pockets: list[list[PocketRecord]] = [[] for _ in range(n_folds)]
    fold_label_counts: list[dict[str, int]] = [{} for _ in range(n_folds)]

    for _group_key, group_pockets in shuffled:
        group_label_counts = label_counts_for_pockets(group_pockets, task=task)
        best_fold = min(
            range(n_folds),
            key=lambda idx: (
                val_assignment_penalty(
                    val_count=len(fold_pockets[idx]) + len(group_pockets),
                    target_val_size=target_fold_size,
                    current_label_counts=merge_label_counts(fold_label_counts[idx], group_label_counts),
                    desired_label_counts=desired_label_counts,
                ),
                len(fold_pockets[idx]),
                idx,
            ),
        )
        fold_pockets[best_fold].extend(group_pockets)
        fold_label_counts[best_fold] = merge_label_counts(fold_label_counts[best_fold], group_label_counts)

    val_pockets = fold_pockets[fold_index]
    train_pockets = [
        pocket
        for idx, fold in enumerate(fold_pockets)
        if idx != fold_index
        for pocket in fold
    ]
    if not train_pockets or not val_pockets:
        raise ValueError(
            "K-fold split produced an empty train or validation set. "
            "Adjust --n-folds or --split-by."
        )
    return PocketSplit(train_pockets=train_pockets, val_pockets=val_pockets)


def task_label_keys_for_pocket(pocket: PocketRecord, task: str) -> list[str]:
    keys: list[str] = []
    if task in ("joint", "metal") and pocket.y_metal is not None:
        keys.append(f"metal:{int(pocket.y_metal)}")
        site_symbol_key = metal_site_symbol_key_for_pocket(pocket)
        if site_symbol_key is not None:
            keys.append(f"metal_site:{site_symbol_key}")
    if task in ("joint", "ec") and pocket.y_ec is not None:
        keys.append(f"ec:{int(pocket.y_ec)}")
    return keys


def metal_site_symbol_key_for_pocket(pocket: PocketRecord) -> str | None:
    raw_symbols = (
        pocket.metadata.get("matched_summary_site_metal_types")
        or pocket.metadata.get("metal_symbols_observed")
        or [pocket.metal_element]
    )
    if isinstance(raw_symbols, str):
        raw_symbols = [raw_symbols]

    symbols = sorted(
        {
            str(symbol).strip().upper()
            for symbol in raw_symbols
            if str(symbol).strip().upper() in METAL_SYMBOL_TO_TARGET
        }
    )
    if not symbols:
        return None
    return "+".join(symbols)


def metal_sampler_key_for_pocket(pocket: PocketRecord) -> tuple[int, str]:
    if pocket.y_metal is None:
        raise ValueError("Cannot build metal sampler for a pocket without y_metal.")
    label_id = int(pocket.y_metal)
    if label_id == METAL_SYMBOL_TO_TARGET["CO"]:
        site_symbol_key = metal_site_symbol_key_for_pocket(pocket) or "ClassVIII_unknown"
        return label_id, site_symbol_key
    return label_id, METAL_TARGET_LABELS[label_id]


def build_balanced_metal_site_sampler(pockets: list[PocketRecord]) -> WeightedRandomSampler:
    if not pockets:
        raise ValueError("Cannot build a sampler for an empty training set.")

    keys = [metal_sampler_key_for_pocket(pocket) for pocket in pockets]
    key_counts = Counter(keys)
    subkeys_by_label: dict[int, set[str]] = {}
    for label_id, site_key in key_counts:
        subkeys_by_label.setdefault(label_id, set()).add(site_key)

    n_labels = len(subkeys_by_label)
    if n_labels == 0:
        raise ValueError("Cannot build a metal sampler without metal labels.")

    weights = []
    for key in keys:
        label_id, site_key = key
        n_subkeys = len(subkeys_by_label[label_id])
        weights.append(1.0 / (n_labels * n_subkeys * key_counts[(label_id, site_key)]))

    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def label_counts_for_pockets(pockets: list[PocketRecord], task: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for pocket in pockets:
        for key in task_label_keys_for_pocket(pocket, task):
            counts[key] = counts.get(key, 0) + 1
    return counts


def merge_label_counts(base: dict[str, int], extra: dict[str, int]) -> dict[str, int]:
    merged = dict(base)
    for key, value in extra.items():
        merged[key] = merged.get(key, 0) + value
    return merged


def desired_label_counts_for_split(
    pockets: list[PocketRecord],
    *,
    task: str,
    val_fraction: float,
) -> dict[str, float]:
    return {
        key: count * val_fraction
        for key, count in label_counts_for_pockets(pockets, task).items()
    }


def val_assignment_penalty(
    *,
    val_count: int,
    target_val_size: int,
    current_label_counts: dict[str, int],
    desired_label_counts: dict[str, float],
) -> float:
    size_penalty = abs(val_count - target_val_size)
    label_penalty = sum(
        abs(current_label_counts.get(key, 0) - desired_count)
        for key, desired_count in desired_label_counts.items()
    )
    overshoot_penalty = max(0, val_count - target_val_size)
    return (1000.0 * overshoot_penalty) + (2.0 * size_penalty) + (10.0 * label_penalty)


def count_labels(
    pockets: list[PocketRecord],
    attr_name: str,
    label_map: dict[int, str],
) -> dict[str, int]:
    counts = {label_name: 0 for label_name in label_map.values()}
    for pocket in pockets:
        label_idx = getattr(pocket, attr_name)
        if label_idx is None:
            continue
        counts[label_map[int(label_idx)]] += 1
    return counts


def build_dataset_summary(
    split: PocketSplit,
    config: TrainConfig,
    feature_load_report: dict[str, Any],
    ec_label_map: dict[int, str],
) -> dict[str, Any]:
    ec_group_mode = ec_grouping_mode_for_metrics(config.ec_group_weighting)
    return {
        "structure_dir": str(config.structure_dir),
        "summary_csv": str(config.summary_csv),
        "esm_embeddings_dir": config.esm_embeddings_dir,
        "external_features_root_dir": config.external_features_root_dir,
        "external_feature_source": config.external_feature_source,
        "n_train_pockets": len(split.train_pockets),
        "n_val_pockets": len(split.val_pockets),
        "task": config.task,
        "node_feature_set": config.node_feature_set,
        "val_fraction": config.val_fraction,
        "split_by": config.split_by,
        "n_folds": config.n_folds,
        "fold_index": config.fold_index,
        "selection_metric": config.selection_metric,
        "unsupported_metal_policy": config.unsupported_metal_policy,
        "invalid_structure_policy": config.invalid_structure_policy,
        "balance_metal_site_symbols": config.balance_metal_site_symbols,
        "ec_label_depth": config.ec_label_depth,
        "ec_group_weighting": config.ec_group_weighting,
        "ec_group_metric_mode": ec_group_mode,
        "ec_group_weighting_applies_to": "ec_cross_entropy_only",
        "ec_labels": ec_label_map,
        "feature_load_report": feature_load_report,
        "train_feature_coverage": build_pocket_feature_coverage(split.train_pockets),
        "val_feature_coverage": build_pocket_feature_coverage(split.val_pockets),
        "train_metal_distribution": count_labels(split.train_pockets, "y_metal", METAL_TARGET_LABELS),
        "train_ec_distribution": count_labels(split.train_pockets, "y_ec", ec_label_map),
        "val_metal_distribution": count_labels(split.val_pockets, "y_metal", METAL_TARGET_LABELS),
        "val_ec_distribution": count_labels(split.val_pockets, "y_ec", ec_label_map),
        "n_train_ec_groups": count_ec_groups(split.train_pockets, ec_group_mode),
        "n_val_ec_groups": count_ec_groups(split.val_pockets, ec_group_mode),
        "train_ec_group_size_distribution": ec_group_size_distribution(split.train_pockets, ec_group_mode),
        "val_ec_group_size_distribution": ec_group_size_distribution(split.val_pockets, ec_group_mode),
        "train_metal_site_distribution": count_metal_site_symbols(split.train_pockets),
        "val_metal_site_distribution": count_metal_site_symbols(split.val_pockets),
    }


def count_metal_site_symbols(pockets: list[PocketRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for pocket in pockets:
        site_key = metal_site_symbol_key_for_pocket(pocket)
        if site_key is not None:
            counts[site_key] = counts.get(site_key, 0) + 1
    return dict(sorted(counts.items()))
