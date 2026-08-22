from __future__ import annotations

import gc
import hashlib
import json
import os
import statistics
import sys
import tarfile
import time
import traceback
import urllib.request
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
    with tarfile.open(REMOTE_SOURCE_ARCHIVE, "r:gz") as archive:
        archive.extractall(REMOTE_SRC, filter="data")
    sys.path.insert(0, str(REMOTE_SRC))

import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from model_variants import build_pocket_classifier


CURRENT_BATCH_SIZE = 12
MEASURED_STEPS = 20
WARMUP_STEPS = 3
DATALOADER_NUM_WORKERS = 4
DATALOADER_PIN_MEMORY = True
ESM_DIM = 960
SUBSET_PATH = Path(os.environ.get("DEEPMZYME_BENCHMARK_SUBSET", "/content/realistic_subset.pt"))
SUBSET_URL = os.environ.get(
    "DEEPMZYME_BENCHMARK_SUBSET_URL",
    "https://huggingface.co/datasets/GMBioinformatics/DeepMzyme/resolve/main/"
    "benchmarks/gvp_esm_hybrid_realistic_subset_v1/realistic_subset.pt",
)
SUBSET_SHA256 = os.environ.get(
    "DEEPMZYME_BENCHMARK_SUBSET_SHA256",
    "84e7e039f1df5b3a7b32dc3d4ac1b8fa21bba2827679b4d3f1650d394e2754bf",
)
RESULT_PATH = Path(
    os.environ.get("DEEPMZYME_BENCHMARK_RESULT", "/content/benchmark_result_realistic.json")
)


# Kept byte-for-byte equivalent in meaning to the preceding synthetic comparison.
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


NORMALIZABLE_NODE_FIELDS = {
    "hydrophobicity_kd",
    "x_dist_raw",
    "x_misc",
    "x_env_burial",
    "x_env_electrostatics",
}


class NormalizedGraphDataset(Dataset):
    """Mirror PocketGraphDataset's precomputed-data path on the compact payload."""

    def __init__(
        self,
        graphs: list[Any],
        means: dict[str, torch.Tensor],
        stds: dict[str, torch.Tensor],
        clamp_value: float,
    ) -> None:
        self.graphs = graphs
        self.means = means
        self.stds = stds
        self.clamp_value = float(clamp_value)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> Any:
        data = self.graphs[index].clone()
        for feature_name, mean in self.means.items():
            if not hasattr(data, feature_name):
                continue
            value = getattr(data, feature_name).float()
            if feature_name == "x_dist_raw" and not hasattr(data, "x_dist_raw_raw"):
                setattr(data, "x_dist_raw_raw", value.clone())
            normalized = (value - mean) / self.stds[feature_name]
            if (
                feature_name in NORMALIZABLE_NODE_FIELDS
                and hasattr(data, "metal_node_mask")
                and normalized.ndim > 0
                and normalized.size(0) == int(data.metal_node_mask.numel())
            ):
                normalized = normalized.clone()
                normalized[data.metal_node_mask.to(dtype=torch.bool)] = 0.0
            setattr(data, feature_name, normalized.clamp(-self.clamp_value, self.clamp_value))
        return data


def is_oom_error(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def training_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer, batch: Any) -> float:
    optimizer.zero_grad(set_to_none=True)
    outputs = model(batch)
    loss = outputs["loss"]
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach().item())


def write_result(payload: dict[str, Any]) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_subset_available() -> None:
    if SUBSET_PATH.is_file():
        actual_sha256 = sha256_file(SUBSET_PATH)
        if actual_sha256 != SUBSET_SHA256:
            raise ValueError(
                f"Benchmark subset checksum mismatch at {SUBSET_PATH}: "
                f"expected {SUBSET_SHA256}, got {actual_sha256}"
            )
        return

    SUBSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    partial_path = SUBSET_PATH.with_name(f"{SUBSET_PATH.name}.part")
    request = urllib.request.Request(SUBSET_URL, headers={"User-Agent": "DeepMzyme-benchmark/1"})
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(request) as response, partial_path.open("wb") as handle:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                handle.write(chunk)
                digest.update(chunk)
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != SUBSET_SHA256:
            raise ValueError(
                f"Downloaded benchmark subset checksum mismatch: "
                f"expected {SUBSET_SHA256}, got {actual_sha256}"
            )
        os.replace(partial_path, SUBSET_PATH)
    finally:
        partial_path.unlink(missing_ok=True)


def main() -> None:
    started_at = time.time()
    payload: dict[str, Any] = {
        "benchmark": "DeepMzyme realistic end-to-end GVP+ESM hybrid training step",
        "current_batch_size": CURRENT_BATCH_SIZE,
        "measured_steps": MEASURED_STEPS,
        "warmup_steps": WARMUP_STEPS,
        "precision": "fp32",
        "amp": False,
        "dataloading_in_timed_region": True,
        "dataloader_num_workers": DATALOADER_NUM_WORKERS,
        "dataloader_pin_memory": DATALOADER_PIN_MEMORY,
        "model_config": MODEL_CONFIG,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_arch_list": torch.cuda.get_arch_list(),
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
        ensure_subset_available()
        payload["input_subset_artifact"] = {
            "path": str(SUBSET_PATH),
            "source_url": SUBSET_URL,
            "sha256": SUBSET_SHA256,
        }
        compact = torch.load(SUBSET_PATH, map_location="cpu", weights_only=False)
        payload["input_subset"] = compact["metadata"]
        dataset = NormalizedGraphDataset(
            compact["graphs"],
            compact["normalization_means"],
            compact["normalization_stds"],
            compact["normalization_clamp"],
        )
        if len(dataset) != CURRENT_BATCH_SIZE * MEASURED_STEPS:
            raise ValueError(
                f"Expected {CURRENT_BATCH_SIZE * MEASURED_STEPS} graphs, got {len(dataset)}"
            )
        loader = DataLoader(
            dataset,
            batch_size=CURRENT_BATCH_SIZE,
            shuffle=False,
            num_workers=DATALOADER_NUM_WORKERS,
            pin_memory=DATALOADER_PIN_MEMORY,
            persistent_workers=True,
        )

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        model = build_pocket_classifier(**MODEL_CONFIG).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=3.705631497756492e-5, weight_decay=1e-5
        )
        model.train()
        payload["model_class"] = f"{type(model).__module__}.{type(model).__name__}"
        payload["model_parameter_count"] = sum(p.numel() for p in model.parameters())

        warmup_iterator = iter(loader)
        for _ in range(WARMUP_STEPS):
            batch = next(warmup_iterator).to(device)
            training_step(model, optimizer, batch)
            del batch
        torch.cuda.synchronize()

        # A fresh iterator measures one complete, steady-state 20-batch epoch.
        measured_iterator = iter(loader)
        torch.cuda.reset_peak_memory_stats()
        step_times: list[float] = []
        losses: list[float] = []
        batch_residue_counts: list[int] = []
        batch_node_counts: list[int] = []
        batch_edge_counts: list[int] = []
        for _ in range(MEASURED_STEPS):
            torch.cuda.synchronize()
            step_started = time.perf_counter()
            batch = next(measured_iterator)
            batch_residue_counts.append(int(batch.residue_node_mask.sum().item()))
            batch_node_counts.append(int(batch.num_nodes))
            batch_edge_counts.append(int(batch.edge_index.size(1)))
            batch = batch.to(device)
            losses.append(training_step(model, optimizer, batch))
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - step_started)
            del batch

        payload.update(
            {
                "median_step_time_seconds": statistics.median(step_times),
                "step_times_seconds": step_times,
                "last_loss": losses[-1],
                "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "batch_residue_counts": batch_residue_counts,
                "batch_node_counts": batch_node_counts,
                "batch_edge_counts": batch_edge_counts,
            }
        )
        optimizer.zero_grad(set_to_none=True)
        del measured_iterator, warmup_iterator, loader, dataset, compact, optimizer, model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
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


if __name__ == "__main__":
    main()
