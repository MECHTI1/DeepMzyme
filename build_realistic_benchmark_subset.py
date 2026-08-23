from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Match the historical workload without encoding the target metal identity in graph features.
os.environ["DEEPGM_METAL_LABEL_SCHEME"] = "five_class"

from benchmarking.artifacts import (  # noqa: E402
    REALISTIC_SUBSET_ARTIFACT_TYPE,
    REALISTIC_SUBSET_SCHEMA_VERSION,
    graph_to_tensor_mapping,
    validate_portable_subset,
)
from training.data import load_training_pockets_with_report_from_dir  # noqa: E402
from training.graph_dataset import build_graph_data_list, compute_feature_normalization_stats  # noqa: E402
from training.reproducibility import sha256_file, source_control_payload, utc_timestamp  # noqa: E402

DEFAULT_DATASET_ROOT = REPO_ROOT / "DeepMzyme_Data" / "CARE_task1_30_clusterRes30_train_test_metallo"
DEFAULT_ESM_DIR = REPO_ROOT / "DeepMzyme_Data" / "esm_embeddings"
DEFAULT_RING_DIR = REPO_ROOT / "DeepMzyme_Data" / "RING_features"
DEFAULT_EXTERNAL_DIR = REPO_ROOT / "DeepMzyme_Data" / "updated_feature_extraction"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "bench" / "gvp_esm_hybrid_realistic_subset_v2"
SAMPLE_SIZE = 240
SAMPLE_SEED = 42
ESM_DIM = 960


def distribution(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": min(ordered),
        "median": statistics.median(ordered),
        "max": max(ordered),
    }


def empirical_quantile_indices(population_size: int, sample_size: int) -> list[int]:
    if sample_size > population_size:
        raise ValueError(f"sample_size={sample_size} exceeds population_size={population_size}")
    if sample_size == 1:
        return [population_size // 2]
    return [round(index * (population_size - 1) / (sample_size - 1)) for index in range(sample_size)]


def csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the portable DeepMzyme realistic benchmark subset v2.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--esm-embeddings-dir", type=Path, default=DEFAULT_ESM_DIR)
    parser.add_argument("--ring-features-dir", type=Path, default=DEFAULT_RING_DIR)
    parser.add_argument("--external-features-dir", type=Path, default=DEFAULT_EXTERNAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--sample-seed", type=int, default=SAMPLE_SEED)
    parser.add_argument("--source-bundle-id", required=True)
    parser.add_argument("--source-bundle-sha256", required=True)
    parser.add_argument("--artifact-url", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train_dir = args.dataset_root / "train"
    summary_csv = train_dir / "final_data_summarazing_table_transition_metals_only_catalytic.csv"
    for required_path in (
        train_dir,
        summary_csv,
        args.esm_embeddings_dir,
        args.ring_features_dir,
        args.external_features_dir,
    ):
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    result = load_training_pockets_with_report_from_dir(
        structure_dir=train_dir,
        require_full_labels=True,
        required_targets=("metal", "ec"),
        summary_csv=summary_csv,
        esm_dim=ESM_DIM,
        esm_embeddings_dir=args.esm_embeddings_dir,
        require_esm_embeddings=True,
        ring_features_dir=args.ring_features_dir,
        external_features_root_dir=args.external_features_dir,
        external_feature_source="updated",
        require_external_features=True,
        unsupported_metal_policy="error",
        invalid_structure_policy="skip",
        ec_label_depth=1,
    )
    pockets = result.pockets
    ordered = sorted(pockets, key=lambda pocket: (len(pocket.residues), pocket.pocket_id))
    selected = [ordered[index] for index in empirical_quantile_indices(len(ordered), args.sample_size)]
    random.Random(args.sample_seed).shuffle(selected)

    graphs = build_graph_data_list(
        selected,
        esm_dim=ESM_DIM,
        edge_radius=6.0,
        use_ring_edges=True,
        require_ring_edges=True,
        node_feature_set="conservative",
        omit_node_features=(),
        metal_node_mode="per_metal",
    )
    stats = compute_feature_normalization_stats(graphs, clamp_value=5.0)
    source_control = source_control_payload(REPO_ROOT)
    source_site_count = csv_row_count(summary_csv)
    eligible_sizes = [len(pocket.residues) for pocket in pockets]
    sampled_sizes = [int(graph.residue_node_mask.sum().item()) for graph in graphs]

    metadata: dict[str, Any] = {
        "source_dataset": "CARE_task1_30_clusterRes30 train",
        "source_site_count": source_site_count,
        "eligible_feature_complete_population": distribution(eligible_sizes),
        "sampled_distribution": distribution(sampled_sizes),
        "sample_size": len(graphs),
        "sample_seed": args.sample_seed,
        "batch_size": 12,
        "full_batches": len(graphs) // 12,
        "held_out_test_used": False,
        "label_scheme": "five_class",
        "ec_class_count": len(result.ec_index_to_label),
        "sampling_algorithm": {
            "name": "evenly_spaced_empirical_residue_count_quantiles_then_seeded_shuffle",
            "ordering": "(residue_count, pocket_id)",
            "quantile_index_formula": "round(i * (population_size - 1) / (sample_size - 1))",
            "shuffle": "python random.Random(sample_seed).shuffle",
        },
        "graph_config": {
            "esm_dim": ESM_DIM,
            "edge_radius": 6.0,
            "use_ring_edges": True,
            "require_ring_edges": True,
            "node_feature_set": "conservative",
            "metal_node_mode": "per_metal",
        },
    }
    provenance = {
        "created_at_utc": utc_timestamp(),
        "source_bundle": {"identifier": args.source_bundle_id, "sha256": args.source_bundle_sha256},
        "source_summary_csv": {
            "path": str(summary_csv),
            "sha256": sha256_file(summary_csv),
            "row_count": source_site_count,
        },
        "eligibility_rules": {
            "required_targets": ["metal", "ec"],
            "require_full_labels": True,
            "require_esm_embeddings": True,
            "require_ring_edges": True,
            "external_feature_source": "updated",
            "require_external_features": True,
            "unsupported_metal_policy": "error",
            "invalid_structure_policy": "skip",
            "ec_label_depth": 1,
        },
        "load_report": result.feature_report,
        "graph_builder_source_control": source_control,
        "builder": {
            "path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    payload = {
        "artifact_type": REALISTIC_SUBSET_ARTIFACT_TYPE,
        "schema_version": REALISTIC_SUBSET_SCHEMA_VERSION,
        "graphs": [graph_to_tensor_mapping(graph) for graph in graphs],
        "normalization": {
            "means": stats.means,
            "stds": stats.stds,
            "clamp": float(stats.clamp_value),
        },
        "metadata": metadata,
        "provenance": provenance,
    }
    validate_portable_subset(payload)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "realistic_subset_v2.pt"
    manifest_path = args.output_dir / "realistic_subset_v2.json"
    torch.save(payload, output_path)
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys, torch; torch.load(sys.argv[1], map_location='cpu', weights_only=True)",
            str(output_path),
        ],
        cwd=args.output_dir,
        check=True,
    )
    safely_loaded = torch.load(output_path, map_location="cpu", weights_only=True)
    validate_portable_subset(safely_loaded)

    manifest = {
        "artifact_type": REALISTIC_SUBSET_ARTIFACT_TYPE,
        "schema_version": REALISTIC_SUBSET_SCHEMA_VERSION,
        "artifact": {
            "filename": output_path.name,
            "url": args.artifact_url,
            "size_bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
            "weights_only_load_verified": True,
        },
        "metadata": metadata,
        "provenance": provenance,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
