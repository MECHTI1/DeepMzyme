from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Match the live notebook before importing label-dependent project modules.
os.environ["DEEPGM_METAL_LABEL_SCHEME"] = "five_class"

from training.data import load_training_pockets_with_report_from_dir
from training.graph_dataset import build_graph_data_list, compute_feature_normalization_stats


DATASET_ROOT = REPO_ROOT / "DeepMzyme_Data" / "CARE_task1_30_clusterRes30_train_test_metallo"
TRAIN_DIR = DATASET_ROOT / "train"
SUMMARY_CSV = TRAIN_DIR / "final_data_summarazing_table_transition_metals_only_catalytic.csv"
ESM_DIR = REPO_ROOT / "DeepMzyme_Data" / "esm_embeddings"
RING_DIR = REPO_ROOT / "DeepMzyme_Data" / "RING_features"
EXTERNAL_DIR = REPO_ROOT / "DeepMzyme_Data" / "updated_feature_extraction"
OUTPUT_PATH = REPO_ROOT / "bench" / "realistic_subset.pt"
METADATA_PATH = REPO_ROOT / "bench" / "realistic_subset.json"

SAMPLE_SIZE = 240
SAMPLE_SEED = 42
ESM_DIM = 960
ARTIFACT_VERSION = "v1"
ARTIFACT_URL = (
    "https://huggingface.co/datasets/GMBioinformatics/DeepMzyme/resolve/main/"
    "benchmarks/gvp_esm_hybrid_realistic_subset_v1/realistic_subset.pt"
)


def distribution(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": min(ordered),
        "median": statistics.median(ordered),
        "max": max(ordered),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def empirical_quantile_indices(population_size: int, sample_size: int) -> list[int]:
    if sample_size > population_size:
        raise ValueError(f"sample_size={sample_size} exceeds population_size={population_size}")
    if sample_size == 1:
        return [population_size // 2]
    return [round(i * (population_size - 1) / (sample_size - 1)) for i in range(sample_size)]


def main() -> None:
    for required_path in (TRAIN_DIR, SUMMARY_CSV, ESM_DIR, RING_DIR, EXTERNAL_DIR):
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    result = load_training_pockets_with_report_from_dir(
        structure_dir=TRAIN_DIR,
        require_full_labels=True,
        required_targets=("metal", "ec"),
        summary_csv=SUMMARY_CSV,
        esm_dim=ESM_DIM,
        esm_embeddings_dir=ESM_DIR,
        require_esm_embeddings=True,
        ring_features_dir=RING_DIR,
        external_features_root_dir=EXTERNAL_DIR,
        external_feature_source="updated",
        require_external_features=True,
        unsupported_metal_policy="error",
        invalid_structure_policy="skip",
        ec_label_depth=1,
    )
    pockets = result.pockets
    ordered = sorted(pockets, key=lambda pocket: (len(pocket.residues), pocket.pocket_id))
    selected = [ordered[index] for index in empirical_quantile_indices(len(ordered), SAMPLE_SIZE)]

    # Deterministically mix sizes so each batch resembles a shuffled training batch.
    random.Random(SAMPLE_SEED).shuffle(selected)
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

    full_sizes = [len(pocket.residues) for pocket in pockets]
    sampled_sizes = [int(graph.residue_node_mask.sum().item()) for graph in graphs]
    metadata = {
        "source": "actual CARE_task1_30_clusterRes30_train_test_metallo training pockets",
        "selection": "240 evenly spaced empirical residue-count quantiles, then seed-42 shuffle",
        "held_out_test_used": False,
        "full_training_distribution": distribution(full_sizes),
        "sampled_distribution": distribution(sampled_sizes),
        "sample_seed": SAMPLE_SEED,
        "sample_size": len(graphs),
        "batch_size": 12,
        "full_batches": len(graphs) // 12,
        "graph_config": {
            "esm_dim": ESM_DIM,
            "edge_radius": 6.0,
            "use_ring_edges": True,
            "require_ring_edges": True,
            "node_feature_set": "conservative",
            "metal_node_mode": "per_metal",
        },
        "ec_class_count": len(result.ec_index_to_label),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "graphs": graphs,
            "normalization_means": stats.means,
            "normalization_stds": stats.stds,
            "normalization_clamp": stats.clamp_value,
            "metadata": metadata,
        },
        OUTPUT_PATH,
    )
    metadata["serialized_bytes"] = OUTPUT_PATH.stat().st_size
    metadata["artifact"] = {
        "version": ARTIFACT_VERSION,
        "filename": OUTPUT_PATH.name,
        "huggingface_url": ARTIFACT_URL,
        "sha256": sha256_file(OUTPUT_PATH),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
