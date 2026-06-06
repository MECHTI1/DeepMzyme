from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


TASK_DEFAULT_RANK_METRICS = {
    "metal": "mean_val_metal_balanced_acc",
    "joint": "mean_validation_metric",
    "ec": "mean_validation_metric",
}

TASK_DEFAULT_MIN_METRICS = {
    "metal": "min_val_metal_min_recall",
    "joint": "min_validation_metric",
    "ec": "min_validation_metric",
}

TASK_DEFAULT_STD_METRICS = {
    "metal": "std_val_metal_balanced_acc",
    "joint": "std_validation_metric",
    "ec": "std_validation_metric",
}

TASK_DEFAULT_TIE_BREAKERS = {
    "metal": (
        "mean_val_metal_min_recall_desc",
        "min_validation_metric_desc",
        "std_val_metal_balanced_acc_asc",
        "model_complexity_proxy_asc",
    ),
    "joint": (
        "min_validation_metric_desc",
        "std_validation_metric_asc",
        "model_complexity_proxy_asc",
    ),
    "ec": (
        "min_validation_metric_desc",
        "std_validation_metric_asc",
        "model_complexity_proxy_asc",
    ),
}

METAL_RARE_RECALL_METRICS = (
    "mean_val_metal_min_recall",
    "min_val_metal_min_recall",
    "mean_val_metal_per_class_recall",
)

EC_RARE_RECALL_METRICS = (
    "mean_val_ec_group_min_recall",
    "min_val_ec_group_min_recall",
    "mean_val_ec_min_recall",
    "min_val_ec_min_recall",
)

METRIC_ALIASES = {
    "mean_validation_balanced_acc": "mean_validation_metric",
}


@dataclass(frozen=True)
class PipelineMetricPolicy:
    task: str
    requested_metric: str
    rank_metric: str | None
    min_metric: str | None
    std_metric: str | None
    rare_recall_metrics: tuple[str, ...]
    rare_recall_policy: str
    tie_breakers: tuple[str, ...]
    task_specific_notes: tuple[str, ...]
    available_numeric_metrics: tuple[str, ...]
    suggested_metric: str | None


def normalize_pipeline_task(task: object) -> str:
    text = str(task or "").strip().lower()
    if text in {"metal", "ec", "joint"}:
        return text
    return "metal"


def metric_is_numeric_in_rows(rows: Iterable[dict], metric: str) -> bool:
    for row in rows:
        value = row.get(metric)
        try:
            number = float(value)
        except Exception:
            continue
        if number == number and number not in (float("inf"), float("-inf")):
            return True
    return False


def numeric_metric_columns(rows: Iterable[dict]) -> tuple[str, ...]:
    rows = list(rows)
    columns = sorted({str(key) for row in rows for key in row})
    return tuple(column for column in columns if metric_is_numeric_in_rows(rows, column))


def _first_available(candidates: Iterable[str], available_columns: set[str]) -> str | None:
    for candidate in candidates:
        if candidate in available_columns:
            return candidate
    return None


def _suggested_rank_metric(task: str, available_columns: set[str]) -> str | None:
    if task == "metal":
        return _first_available(
            (
                "mean_val_metal_balanced_acc",
                "mean_validation_metric",
                "mean_validation_balanced_acc",
            ),
            available_columns,
        )
    if task == "ec":
        return _first_available(
            (
                "mean_val_ec_group_balanced_acc",
                "mean_val_ec_balanced_acc",
                "mean_validation_metric",
                "val_ec_group_balanced_acc",
            ),
            available_columns,
        )
    return _first_available(
        (
            "mean_validation_metric",
            "mean_val_joint_balanced_acc",
            "mean_val_metal_balanced_acc",
            "mean_val_ec_group_balanced_acc",
        ),
        available_columns,
    )


def resolve_pipeline_metric_policy(
    task: object,
    available_columns: Iterable[str],
    requested_metric: object | None = None,
    *,
    rare_recall_policy: object = "auto",
    block_on_missing_rare_recall: bool = True,
    requested_tie_breakers: object = "auto",
) -> PipelineMetricPolicy:
    task_name = normalize_pipeline_task(task)
    available = {str(column) for column in available_columns if str(column)}
    requested = str(requested_metric or "auto").strip() or "auto"
    requested_rare_policy = str(rare_recall_policy or "auto").strip().lower() or "auto"
    if requested_rare_policy not in {"auto", "required", "off"}:
        requested_rare_policy = "auto"

    notes: list[str] = []
    suggested = _suggested_rank_metric(task_name, available)

    if requested == "auto":
        rank_metric = suggested or TASK_DEFAULT_RANK_METRICS[task_name]
        notes.append(f"Rank metric auto-resolved for task={task_name}: {rank_metric}")
    elif requested in available:
        rank_metric = requested
    elif requested in METRIC_ALIASES and METRIC_ALIASES[requested] in available:
        rank_metric = METRIC_ALIASES[requested]
        notes.append(f"Metric alias used: {requested} -> {rank_metric}")
    else:
        rank_metric = None
        notes.append(
            f"Requested metric {requested!r} is not available as a numeric Stage 6 column."
        )

    min_metric = _first_available((TASK_DEFAULT_MIN_METRICS[task_name], "min_validation_metric"), available)
    std_metric = _first_available((TASK_DEFAULT_STD_METRICS[task_name], "std_validation_metric"), available)

    if requested_rare_policy == "off":
        effective_rare_policy = "off"
    elif task_name == "ec" and requested_rare_policy == "auto":
        effective_rare_policy = "off"
        notes.append(
            "EC task detected: metal rare-recall metrics are not required by the auto policy."
        )
    elif task_name in {"metal", "joint"} and requested_rare_policy == "auto":
        effective_rare_policy = "required" if block_on_missing_rare_recall else "optional"
    else:
        effective_rare_policy = "required"

    if task_name == "ec":
        rare_metrics = tuple(metric for metric in EC_RARE_RECALL_METRICS if metric in available)
    else:
        rare_metrics = tuple(metric for metric in METAL_RARE_RECALL_METRICS if metric in available)

    raw_tie_breakers = str(requested_tie_breakers or "auto").strip()
    if raw_tie_breakers in {"", "auto"}:
        tie_breakers = TASK_DEFAULT_TIE_BREAKERS[task_name]
    else:
        tie_breakers = tuple(part.strip() for part in raw_tie_breakers.split(",") if part.strip())

    return PipelineMetricPolicy(
        task=task_name,
        requested_metric=requested,
        rank_metric=rank_metric,
        min_metric=min_metric,
        std_metric=std_metric,
        rare_recall_metrics=rare_metrics,
        rare_recall_policy=effective_rare_policy,
        tie_breakers=tie_breakers,
        task_specific_notes=tuple(notes),
        available_numeric_metrics=tuple(sorted(available)),
        suggested_metric=suggested,
    )
