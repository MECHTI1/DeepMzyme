from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import tarfile
import time
import traceback
import urllib.request
from pathlib import Path
from typing import Any, Sequence

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
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from benchmarking.artifacts import (
    BENCHMARK_RESULT_ARTIFACT_TYPE,
    BENCHMARK_RESULT_SCHEMA_VERSION,
    load_portable_subset,
    reconstruct_pocket_graphs,
    validate_benchmark_result,
)
from model_variants import build_pocket_classifier
from training.reproducibility import (
    runtime_environment_payload,
    sha256_file,
    source_control_payload,
    utc_timestamp,
)

DEFAULT_SUBSET_PATH = Path("/content/realistic_subset_v2.pt")
DEFAULT_RESULT_PATH = Path("/content/benchmark_result_realistic_v2.json")
ESM_DIM = 960

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the DeepMzyme realistic benchmark schema v2 workload.")
    parser.add_argument("--subset", type=Path, default=Path(os.environ.get("DEEPMZYME_BENCHMARK_SUBSET", DEFAULT_SUBSET_PATH)))
    parser.add_argument("--subset-url", default=os.environ.get("DEEPMZYME_BENCHMARK_SUBSET_URL"))
    parser.add_argument("--subset-sha256", default=os.environ.get("DEEPMZYME_BENCHMARK_SUBSET_SHA256"))
    parser.add_argument("--result", type=Path, default=Path(os.environ.get("DEEPMZYME_BENCHMARK_RESULT", DEFAULT_RESULT_PATH)))
    parser.add_argument("--source-commit", default=os.environ.get("DEEPMZYME_SOURCE_COMMIT"))
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--measured-steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3.705631497756492e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    return parser


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


def ensure_subset_available(path: Path, url: str | None, expected_sha256: str) -> None:
    if path.is_file():
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Benchmark subset checksum mismatch at {path}: expected {expected_sha256}, got {actual_sha256}"
            )
        return
    if not url:
        raise FileNotFoundError(f"Benchmark subset is absent at {path} and no --subset-url was provided.")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_name(f"{path.name}.part")
    request = urllib.request.Request(url, headers={"User-Agent": "DeepMzyme-benchmark/2"})
    try:
        with urllib.request.urlopen(request) as response, partial_path.open("wb") as handle:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                handle.write(chunk)
        actual_sha256 = sha256_file(partial_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Downloaded benchmark subset checksum mismatch: expected {expected_sha256}, got {actual_sha256}"
            )
        os.replace(partial_path, path)
    finally:
        partial_path.unlink(missing_ok=True)


def write_result(path: Path, payload: dict[str, Any]) -> None:
    validate_benchmark_result(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started_at = time.time()
    local_source_control = source_control_payload(REPO_ROOT)
    source_commit = args.source_commit or local_source_control.get("commit")
    runner_path = Path(__file__).resolve()
    payload: dict[str, Any] = {
        "artifact_type": BENCHMARK_RESULT_ARTIFACT_TYPE,
        "schema_version": BENCHMARK_RESULT_SCHEMA_VERSION,
        "benchmark_id": "gvp_esm_hybrid_realistic_training_step_v2",
        "status": "error",
        "created_at_utc": utc_timestamp(),
        "invocation": {"argv": [sys.executable, *sys.argv], "working_directory": str(Path.cwd())},
        "source_control": {**local_source_control, "commit": source_commit},
        "runner": {"path": str(runner_path), "sha256": sha256_file(runner_path)},
        "runtime_environment": runtime_environment_payload(),
        "input_artifact": {
            "path": str(args.subset),
            "source_url": args.subset_url,
            "sha256": args.subset_sha256,
        },
        "workload": {
            "batch_size": args.batch_size,
            "measured_steps": args.measured_steps,
            "warmup_steps": args.warmup_steps,
            "dataloader_num_workers": args.num_workers,
            "dataloader_pin_memory": True,
            "dataloading_in_timed_region": True,
            "precision": "fp32",
            "amp": False,
            "model_config": MODEL_CONFIG,
        },
        "optimizer": {
            "class": "torch.optim.AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip_norm": 1.0,
        },
        "seeds": {
            "python": None,
            "torch": args.seed,
            "torch_cuda_all": args.seed,
            "dataloader_shuffle": False,
            "deterministic_algorithms": False,
        },
        "timing": {
            "step_times_seconds": [],
            "median_step_time_seconds": None,
            "samples_per_second": None,
            "elapsed_seconds": None,
        },
        "oom": False,
    }

    try:
        if not source_commit:
            raise ValueError("A source commit is required; run in a Git checkout or pass --source-commit.")
        if not args.subset_sha256 or len(args.subset_sha256) != 64:
            raise ValueError("--subset-sha256 must provide the exact 64-character v2 artifact checksum.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        ensure_subset_available(args.subset, args.subset_url, args.subset_sha256)
        compact = load_portable_subset(args.subset)
        graphs = reconstruct_pocket_graphs(compact)
        normalization = compact["normalization"]
        dataset = NormalizedGraphDataset(
            graphs,
            normalization["means"],
            normalization["stds"],
            normalization["clamp"],
        )
        expected_graphs = args.batch_size * args.measured_steps
        if len(dataset) != expected_graphs:
            raise ValueError(f"Expected {expected_graphs} graphs, got {len(dataset)}")
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )

        device = torch.device("cuda:0")
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        model = build_pocket_classifier(**MODEL_CONFIG).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        model.train()
        payload["model"] = {
            "class": f"{type(model).__module__}.{type(model).__name__}",
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        }
        payload["input_artifact"]["metadata"] = compact["metadata"]
        payload["input_artifact"]["provenance"] = compact["provenance"]

        warmup_iterator = iter(loader)
        for _ in range(args.warmup_steps):
            batch = next(warmup_iterator).to(device)
            training_step(model, optimizer, batch)
            del batch
        torch.cuda.synchronize()

        measured_iterator = iter(loader)
        torch.cuda.reset_peak_memory_stats()
        step_times: list[float] = []
        losses: list[float] = []
        batch_residue_counts: list[int] = []
        batch_node_counts: list[int] = []
        batch_edge_counts: list[int] = []
        for _ in range(args.measured_steps):
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

        median_step = statistics.median(step_times)
        payload["timing"].update(
            {
                "step_times_seconds": step_times,
                "median_step_time_seconds": median_step,
                "samples_per_second": args.batch_size / median_step,
            }
        )
        payload["measurements"] = {
            "last_loss": losses[-1],
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "batch_residue_counts": batch_residue_counts,
            "batch_node_counts": batch_node_counts,
            "batch_edge_counts": batch_edge_counts,
        }
        payload["status"] = "ok"
        optimizer.zero_grad(set_to_none=True)
        del measured_iterator, warmup_iterator, loader, dataset, graphs, compact, optimizer, model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception as exc:
        payload["oom"] = is_oom_error(exc)
        payload["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    payload["timing"]["elapsed_seconds"] = time.time() - started_at
    write_result(args.result, payload)
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
