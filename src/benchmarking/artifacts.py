from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

REALISTIC_SUBSET_ARTIFACT_TYPE = "deepmzyme.realistic_subset"
REALISTIC_SUBSET_SCHEMA_VERSION = 2
BENCHMARK_RESULT_ARTIFACT_TYPE = "deepmzyme.benchmark_result"
BENCHMARK_RESULT_SCHEMA_VERSION = 2


def _portable_value(value: Any, *, location: str) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"Portable artifact key at {location} must be a string, got {type(key).__name__}.")
            converted[key] = _portable_value(item, location=f"{location}.{key}")
        return converted
    if isinstance(value, (list, tuple)):
        return [_portable_value(item, location=f"{location}[{index}]") for index, item in enumerate(value)]
    raise TypeError(
        f"Portable artifact value at {location} has unsupported type {type(value).__module__}."
        f"{type(value).__name__}; expected a mapping, list, scalar, or tensor."
    )


def graph_to_tensor_mapping(graph: Any) -> dict[str, Any]:
    if not hasattr(graph, "to_dict"):
        raise TypeError(f"Graph object {type(graph).__name__} does not provide to_dict().")
    mapping = _portable_value(graph.to_dict(), location="graph")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Graph mapping must be a non-empty dictionary.")
    return mapping


def validate_portable_subset(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("Portable subset payload must be a dictionary.")
    if payload.get("artifact_type") != REALISTIC_SUBSET_ARTIFACT_TYPE:
        raise ValueError(f"Unexpected artifact_type: {payload.get('artifact_type')!r}.")
    if payload.get("schema_version") != REALISTIC_SUBSET_SCHEMA_VERSION:
        raise ValueError(f"Unsupported realistic-subset schema_version: {payload.get('schema_version')!r}.")
    graphs = payload.get("graphs")
    if not isinstance(graphs, list) or not graphs:
        raise ValueError("Portable subset requires a non-empty graphs list.")
    for index, graph in enumerate(graphs):
        if not isinstance(graph, dict) or not graph:
            raise TypeError(f"graphs[{index}] must be a non-empty tensor mapping.")
        _portable_value(graph, location=f"graphs[{index}]")
    normalization = payload.get("normalization")
    if not isinstance(normalization, dict):
        raise TypeError("Portable subset requires a normalization mapping.")
    for field_name in ("means", "stds"):
        if not isinstance(normalization.get(field_name), dict):
            raise TypeError(f"normalization.{field_name} must be a tensor mapping.")
    if not isinstance(normalization.get("clamp"), (int, float)):
        raise TypeError("normalization.clamp must be numeric.")
    if not isinstance(payload.get("metadata"), dict):
        raise TypeError("Portable subset requires a metadata mapping.")
    if not isinstance(payload.get("provenance"), dict):
        raise TypeError("Portable subset requires a provenance mapping.")
    _portable_value(payload, location="payload")
    return payload


def load_portable_subset(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return validate_portable_subset(payload)


def reconstruct_pocket_graphs(payload: dict[str, Any]) -> list[Any]:
    from graph.construction import PocketData

    validate_portable_subset(payload)
    return [PocketData(**graph_mapping) for graph_mapping in payload["graphs"]]


def validate_benchmark_result(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("Benchmark result must be a dictionary.")
    if payload.get("artifact_type") != BENCHMARK_RESULT_ARTIFACT_TYPE:
        raise ValueError(f"Unexpected benchmark result artifact_type: {payload.get('artifact_type')!r}.")
    if payload.get("schema_version") != BENCHMARK_RESULT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported benchmark result schema_version: {payload.get('schema_version')!r}.")
    if payload.get("status") not in {"ok", "error"}:
        raise ValueError("Benchmark result status must be 'ok' or 'error'.")
    required_mappings = (
        "invocation",
        "source_control",
        "runner",
        "runtime_environment",
        "optimizer",
        "seeds",
        "timing",
    )
    for field_name in required_mappings:
        if not isinstance(payload.get(field_name), dict):
            raise TypeError(f"Benchmark result field {field_name!r} must be a mapping.")
    if not isinstance(payload.get("benchmark_id"), str):
        raise TypeError("Benchmark result benchmark_id must be a string.")
    if payload["status"] == "ok":
        median = payload["timing"].get("median_step_time_seconds")
        throughput = payload["timing"].get("samples_per_second")
        if not isinstance(median, (int, float)) or median <= 0:
            raise ValueError("Successful benchmark result requires a positive median step time.")
        if not isinstance(throughput, (int, float)) or throughput <= 0:
            raise ValueError("Successful benchmark result requires positive throughput.")
        workload = payload.get("workload")
        batch_size = workload.get("batch_size") if isinstance(workload, dict) else payload.get("current_batch_size")
        if isinstance(batch_size, int) and not math.isclose(
            throughput,
            batch_size / median,
            rel_tol=1e-9,
            abs_tol=0.0,
        ):
            raise ValueError("Benchmark throughput does not equal batch_size / median_step_time_seconds.")
    elif not isinstance(payload.get("error"), dict):
        raise TypeError("Failed benchmark result requires an error mapping.")
    return payload
