from __future__ import annotations

import gc
import json
import os
import statistics
import sys
import tarfile
import time
import traceback
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
LOCAL_SRC = REPO_ROOT / "src"
if LOCAL_SRC.is_dir():
    sys.path.insert(0, str(LOCAL_SRC))
REMOTE_SOURCE_ARCHIVE = Path("/content/deepmzyme_benchmark_src.tar.gz")
REMOTE_SRC = Path("/content/deepmzyme_benchmark_src")
if REMOTE_SOURCE_ARCHIVE.is_file():
    REMOTE_SRC.mkdir(parents=True, exist_ok=True)
    if not (REMOTE_SRC / "model.py").is_file():
        with tarfile.open(REMOTE_SOURCE_ARCHIVE, "r:gz") as archive:
            archive.extractall(REMOTE_SRC, filter="data")
    sys.path.insert(0, str(REMOTE_SRC))

import torch
from torch_geometric.data import Data

from data_structures import AA_ORDER, EDGE_SOURCE_TYPES, INTERACTION_SUMMARIES_OPTIONAL_WITH_RING
from model_variants import build_pocket_classifier


CURRENT_BATCH_SIZE = 12
MEASURED_STEPS = 20
WARMUP_STEPS = 3
RESIDUE_NODES_PER_GRAPH = 64
METAL_NODES_PER_GRAPH = 1
DIRECTED_NEIGHBORS_PER_NODE = 24
MAX_CAPACITY_PROBE_BATCH = 16_384
ESM_DIM = 960
RESULT_PATH = Path(os.environ.get("DEEPMZYME_BENCHMARK_RESULT", "/content/benchmark_result.json"))


MODEL_CONFIG: dict[str, Any] = {
    "model_architecture": "gvp",
    "esm_dim": ESM_DIM,
    "hidden_s": 128,
    "hidden_v": 8,
    "edge_hidden": 64,
    "n_layers": 2,
    "n_metal": 5,
    "n_ec": 7,
    "esm_fusion_dim": 64,
    "head_mlp_layers": 1,
    "head_mlp_dropout": 0.2,
    "esm_graph_encoder_dropout": 0.1,
    "node_rbf_sigma": 0.75,
    "edge_rbf_sigma": 0.75,
    "node_rbf_use_raw_distances": False,
    "classifier_pool_distance_cutoff": 0.0,
    "structural_readout_scope": "residue_and_metal",
    "use_node_type_embedding": True,
    "use_site_angle_features": True,
    "normalize_message_aggregation": False,
    "use_esm_branch": True,
    "fusion_mode": "hybrid",
    "use_early_esm": True,
    "early_esm_dim": 32,
    "early_esm_dropout": 0.05,
    "early_esm_raw": False,
    "early_esm_scope": "all",
    "joint_loss_weighting": "uncertainty",
    "metal_loss_weight": 2.0,
    "ec_loss_weight": 0.25,
    "metal_loss_function": "cross_entropy",
    "metal_focal_gamma": 2.0,
    "metal_label_smoothing": 0.0,
    "metal_collapsed_loss_weight": 0.0,
    "ec_contrastive_weight": 0.0,
    "ec_contrastive_temperature": 0.1,
    "predict_metal": True,
    "predict_ec": True,
}


def is_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in message


def make_representative_batch(batch_size: int, device: torch.device) -> Data:
    nodes_per_graph = RESIDUE_NODES_PER_GRAPH + METAL_NODES_PER_GRAPH
    node_count = batch_size * nodes_per_graph
    local_node = torch.arange(nodes_per_graph, device=device).repeat(batch_size)
    graph_batch = torch.arange(batch_size, device=device).repeat_interleave(nodes_per_graph)
    residue_mask = local_node < RESIDUE_NODES_PER_GRAPH
    metal_mask = ~residue_mask

    x_role = torch.zeros((node_count, 2), dtype=torch.float32, device=device)
    x_role[:, 0] = (local_node < 8).float()
    x_role[:, 1] = ((local_node >= 8) & (local_node < 24)).float()
    x_esm = torch.randn((node_count, ESM_DIM), dtype=torch.float32, device=device)
    x_esm[metal_mask] = 0.0

    local_src = torch.arange(nodes_per_graph, device=device).repeat_interleave(
        DIRECTED_NEIGHBORS_PER_NODE
    )
    neighbor_offsets = torch.arange(
        1,
        DIRECTED_NEIGHBORS_PER_NODE + 1,
        device=device,
    ).repeat(nodes_per_graph)
    local_dst = (local_src + neighbor_offsets) % nodes_per_graph
    graph_offsets = (
        torch.arange(batch_size, device=device).repeat_interleave(local_src.numel())
        * nodes_per_graph
    )
    edge_src = local_src.repeat(batch_size) + graph_offsets
    edge_dst = local_dst.repeat(batch_size) + graph_offsets
    edge_index = torch.stack([edge_src, edge_dst], dim=0)
    edge_count = edge_index.size(1)

    edge_vector_raw = torch.randn((edge_count, 3), dtype=torch.float32, device=device)
    edge_dist = edge_vector_raw.norm(dim=-1, keepdim=True).clamp_min_(1e-3)
    edge_interaction_type = torch.zeros(
        (edge_count, len(INTERACTION_SUMMARIES_OPTIONAL_WITH_RING)),
        dtype=torch.float32,
        device=device,
    )
    edge_interaction_type[::4, 0] = 1.0
    edge_source_type = torch.zeros(
        (edge_count, len(EDGE_SOURCE_TYPES)),
        dtype=torch.float32,
        device=device,
    )
    edge_source_type[:, 0] = 1.0
    edge_source_type[::4, 0] = 0.0
    edge_source_type[::4, 1] = 1.0

    data = Data(
        x_esm=x_esm,
        hydrophobicity_kd=torch.randn((node_count, 1), device=device),
        x_reschem=torch.randn((node_count, len(AA_ORDER) + 5), device=device),
        x_role=x_role,
        x_dist_raw=torch.rand((node_count, 3), device=device) * 10.0,
        x_misc=torch.randn((node_count, 1), device=device),
        x_env_burial=torch.randn((node_count, 1), device=device),
        x_env_electrostatics=torch.randn((node_count, 2), device=device),
        x_vec=torch.randn((node_count, 2, 3), device=device),
        edge_index=edge_index,
        edge_dist_raw=torch.cat([edge_dist, edge_dist], dim=-1),
        edge_seqsep=torch.rand((edge_count, 1), device=device),
        edge_same_chain=torch.ones((edge_count, 1), device=device),
        edge_vector_raw=edge_vector_raw,
        edge_interaction_type=edge_interaction_type,
        edge_source_type=edge_source_type,
        residue_node_mask=residue_mask,
        metal_node_mask=metal_mask,
        node_type_id=metal_mask.long(),
        site_metal_stats=torch.randn((batch_size, 4), device=device),
        site_ligand_angle_stats=torch.randn((batch_size, 8), device=device),
        y_metal=torch.arange(batch_size, device=device) % 5,
        y_ec=torch.arange(batch_size, device=device) % 7,
        batch=graph_batch,
    )
    return data


def training_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer, batch: Data) -> float:
    optimizer.zero_grad(set_to_none=True)
    outputs = model(batch)
    loss = outputs["loss"]
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach().item())


def clear_cuda_state(optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def probe_batch_size(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    device: torch.device,
) -> tuple[bool, int | None, str | None]:
    batch = None
    try:
        clear_cuda_state(optimizer)
        torch.cuda.reset_peak_memory_stats()
        batch = make_representative_batch(batch_size, device)
        training_step(model, optimizer, batch)
        torch.cuda.synchronize()
        peak = int(torch.cuda.max_memory_allocated())
        del batch
        clear_cuda_state(optimizer)
        return True, peak, None
    except BaseException as exc:
        oom = is_oom_error(exc)
        error = f"{type(exc).__name__}: {exc}"
        if batch is not None:
            del batch
        clear_cuda_state(optimizer)
        if oom:
            return False, None, error
        raise


def find_largest_batch_size(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, bool, list[dict[str, Any]]]:
    probes: list[dict[str, Any]] = []
    low = CURRENT_BATCH_SIZE
    high = CURRENT_BATCH_SIZE * 2

    while high <= MAX_CAPACITY_PROBE_BATCH:
        fits, peak, error = probe_batch_size(model, optimizer, high, device)
        probes.append({"batch_size": high, "fits": fits, "peak_memory_bytes": peak, "error": error})
        if not fits:
            break
        low = high
        high *= 2

    if high > MAX_CAPACITY_PROBE_BATCH:
        if low < MAX_CAPACITY_PROBE_BATCH:
            fits, peak, error = probe_batch_size(model, optimizer, MAX_CAPACITY_PROBE_BATCH, device)
            probes.append(
                {
                    "batch_size": MAX_CAPACITY_PROBE_BATCH,
                    "fits": fits,
                    "peak_memory_bytes": peak,
                    "error": error,
                }
            )
            if fits:
                return MAX_CAPACITY_PROBE_BATCH, True, probes
            high = MAX_CAPACITY_PROBE_BATCH
        else:
            return low, True, probes

    while low + 1 < high:
        mid = (low + high) // 2
        fits, peak, error = probe_batch_size(model, optimizer, mid, device)
        probes.append({"batch_size": mid, "fits": fits, "peak_memory_bytes": peak, "error": error})
        if fits:
            low = mid
        else:
            high = mid
    return low, False, probes


def write_result(payload: dict[str, Any]) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def main() -> None:
    started_at = time.time()
    payload: dict[str, Any] = {
        "benchmark": "DeepMzyme real GVP+ESM hybrid training step",
        "current_batch_size": CURRENT_BATCH_SIZE,
        "measured_steps": MEASURED_STEPS,
        "warmup_steps": WARMUP_STEPS,
        "precision": "fp32",
        "amp": False,
        "graph_shape": {
            "residue_nodes_per_graph": RESIDUE_NODES_PER_GRAPH,
            "generic_metal_nodes_per_graph": METAL_NODES_PER_GRAPH,
            "directed_neighbors_per_node": DIRECTED_NEIGHBORS_PER_NODE,
            "esm_dim": ESM_DIM,
            "synthetic_inputs": True,
        },
        "model_config": MODEL_CONFIG,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "oom": False,
    }

    if not torch.cuda.is_available():
        payload["runtime_error"] = "CUDA is not available."
        payload["elapsed_seconds"] = time.time() - started_at
        write_result(payload)
        return

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    payload.update(
        {
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_total_memory_bytes": int(properties.total_memory),
            "gpu_compute_capability": f"{properties.major}.{properties.minor}",
        }
    )

    try:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        model = build_pocket_classifier(**MODEL_CONFIG).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3.705631497756492e-5, weight_decay=1e-5)
        model.train()
        payload["model_class"] = f"{type(model).__module__}.{type(model).__name__}"
        payload["model_parameter_count"] = sum(parameter.numel() for parameter in model.parameters())

        current_batch = make_representative_batch(CURRENT_BATCH_SIZE, device)
        for _ in range(WARMUP_STEPS):
            training_step(model, optimizer, current_batch)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        step_times: list[float] = []
        losses: list[float] = []
        for _ in range(MEASURED_STEPS):
            torch.cuda.synchronize()
            step_started = time.perf_counter()
            losses.append(training_step(model, optimizer, current_batch))
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - step_started)

        payload.update(
            {
                "median_step_time_seconds": statistics.median(step_times),
                "step_times_seconds": step_times,
                "last_loss": losses[-1],
                "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            }
        )
        del current_batch
        clear_cuda_state(optimizer)

        largest_batch, capacity_censored, probes = find_largest_batch_size(model, optimizer, device)
        payload.update(
            {
                "largest_batch_size_fit": largest_batch,
                "largest_batch_size_capacity_censored": capacity_censored,
                "capacity_probe_limit": MAX_CAPACITY_PROBE_BATCH,
                "capacity_probes": probes,
            }
        )
    except BaseException as exc:
        payload["oom"] = is_oom_error(exc)
        payload["runtime_error"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc()
        try:
            torch.cuda.empty_cache()
        except BaseException:
            pass

    payload["elapsed_seconds"] = time.time() - started_at
    write_result(payload)


main()
