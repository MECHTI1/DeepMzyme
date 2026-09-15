from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from label_schemes import METAL_TARGET_LABELS

COLLAPSED4_CLASS_ORDER = ("Mn", "Cu", "Zn", "Class VIII")
REQUIRED_SIX_CLASS_METAL_LABELS = ("Mn", "Cu", "Zn", "Fe", "Co", "Ni")


def _label_to_index(label_map: Mapping[int, str]) -> dict[str, int]:
    return {str(label_name): int(label_idx) for label_idx, label_name in label_map.items()}


def validate_required_six_class_metal_labels(
    label_map: Mapping[int, str] = METAL_TARGET_LABELS,
) -> None:
    observed = tuple(str(label_name) for _label_idx, label_name in sorted(label_map.items()))
    missing = [label_name for label_name in REQUIRED_SIX_CLASS_METAL_LABELS if label_name not in observed]
    if missing:
        raise ValueError(
            "Collapsed-4 metal loss requires six-class metal labels "
            f"{list(REQUIRED_SIX_CLASS_METAL_LABELS)}. "
            f"Missing labels: {missing}. Observed labels: {list(observed)}."
        )


def collapsed4_source_indices(
    label_map: Mapping[int, str] = METAL_TARGET_LABELS,
    *,
    require_six_class: bool = False,
) -> tuple[tuple[int, ...], ...]:
    if require_six_class:
        validate_required_six_class_metal_labels(label_map)

    by_name = _label_to_index(label_map)
    source_names_by_class: tuple[tuple[str, ...], ...] = (
        ("Mn",),
        ("Cu",),
        ("Zn",),
        ("Fe", "Co", "Ni") if require_six_class else ("Fe", "Co", "Ni", "Class VIII"),
    )
    sources: list[tuple[int, ...]] = []
    for collapsed_name, source_names in zip(COLLAPSED4_CLASS_ORDER, source_names_by_class):
        indices = tuple(by_name[name] for name in source_names if name in by_name)
        if not indices:
            observed = [str(label_name) for _idx, label_name in sorted(label_map.items())]
            raise ValueError(
                f"Cannot build collapsed-4 metal class {collapsed_name!r}. "
                f"Expected at least one of {list(source_names)}; observed labels: {observed}."
            )
        sources.append(indices)
    return tuple(sources)


def collapse_metal_logits_to_4(
    logits: Tensor,
    *,
    label_map: Mapping[int, str] = METAL_TARGET_LABELS,
    require_six_class: bool = False,
) -> Tensor:
    if logits.ndim < 2:
        raise ValueError(f"Expected metal logits with class dimension, got shape {tuple(logits.shape)}.")
    source_indices = collapsed4_source_indices(label_map, require_six_class=require_six_class)
    collapsed_logits = [
        torch.logsumexp(logits.index_select(-1, torch.as_tensor(indices, device=logits.device)), dim=-1)
        for indices in source_indices
    ]
    return torch.stack(collapsed_logits, dim=-1)


def collapse_metal_targets_to_4(
    targets: Tensor,
    *,
    label_map: Mapping[int, str] = METAL_TARGET_LABELS,
    require_six_class: bool = False,
) -> Tensor:
    collapsed_sources = collapsed4_source_indices(label_map, require_six_class=require_six_class)
    index_to_collapsed = {
        source_idx: collapsed_idx
        for collapsed_idx, source_indices in enumerate(collapsed_sources)
        for source_idx in source_indices
    }
    try:
        collapsed = [index_to_collapsed[int(target_idx)] for target_idx in targets.detach().cpu().tolist()]
    except KeyError as exc:
        observed = [str(label_name) for _idx, label_name in sorted(label_map.items())]
        raise ValueError(
            f"Cannot collapse metal target id {exc.args[0]!r}; observed labels: {observed}."
        ) from exc
    return torch.as_tensor(collapsed, dtype=torch.long, device=targets.device)


def collapse_metal_label_ids_to_4(
    label_ids: Sequence[int],
    *,
    label_map: Mapping[int, str] = METAL_TARGET_LABELS,
    require_six_class: bool = False,
) -> list[int]:
    if not label_ids:
        return []
    targets = torch.as_tensor(list(label_ids), dtype=torch.long)
    return collapse_metal_targets_to_4(
        targets,
        label_map=label_map,
        require_six_class=require_six_class,
    ).tolist()


def collapsed4_cross_entropy_from_logits(
    logits: Tensor,
    targets: Tensor,
    *,
    weight: Tensor | None = None,
    label_smoothing: float = 0.0,
    label_map: Mapping[int, str] = METAL_TARGET_LABELS,
    require_six_class: bool = True,
) -> Tensor:
    collapsed_logits = collapse_metal_logits_to_4(
        logits,
        label_map=label_map,
        require_six_class=require_six_class,
    )
    collapsed_targets = collapse_metal_targets_to_4(
        targets,
        label_map=label_map,
        require_six_class=require_six_class,
    )
    return F.cross_entropy(
        collapsed_logits,
        collapsed_targets,
        weight=weight,
        label_smoothing=float(label_smoothing),
    )


def metal_loss_with_optional_collapsed4(
    six_class_loss: Tensor,
    logits: Tensor,
    targets: Tensor,
    *,
    alpha: float,
    collapsed4_weight: Tensor | None = None,
    label_smoothing: float = 0.0,
    label_map: Mapping[int, str] = METAL_TARGET_LABELS,
) -> tuple[Tensor, Tensor | None]:
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"metal collapsed loss weight must be in [0, 1], got {alpha}.")
    if alpha == 0.0:
        return six_class_loss, None
    collapsed4_loss = collapsed4_cross_entropy_from_logits(
        logits,
        targets,
        weight=collapsed4_weight,
        label_smoothing=label_smoothing,
        label_map=label_map,
        require_six_class=True,
    )
    return ((1.0 - alpha) * six_class_loss) + (alpha * collapsed4_loss), collapsed4_loss
