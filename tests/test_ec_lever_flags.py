"""Existing options behind campaign levers L2, L3, L6 and L7 behave as the EC campaign assumes (synthetic)."""
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from ec_fixtures import default_ec_proteins, tiny_ec_argv, write_ec_dataset


@pytest.fixture()
def dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPMZYME_PARSE_CACHE_DIR", str(tmp_path / "cache"))
    return write_ec_dataset(tmp_path / "data", default_ec_proteins())


def _fit(paths, tmp_path, name, *extra):
    import json

    from training.config import parse_args
    from training.run import run_training

    argv = tiny_ec_argv(paths, tmp_path / "runs", "--val-fraction", "0.3", "--ec-class-vocabulary", "fixed_seven",
                        "--run-name", name, *extra)
    run_dir = run_training(parse_args(argv))
    return run_dir, json.loads((run_dir / "run_config.json").read_text())


def test_l6_dropout_flags_reach_the_modules():
    from model_variants.factory import build_pocket_classifier  # noqa: F401  (import check)
    from training.config import parse_args

    config = parse_args(["--task", "ec", "--head-mlp-dropout", "0.3", "--esm-graph-encoder-dropout", "0.2"])
    assert (config.head_mlp_dropout, config.esm_graph_encoder_dropout) == (0.3, 0.2)


def test_l7_short_cosine_terminal_fit_uses_closed_form_lrs(dataset, tmp_path):
    import math

    run_dir, run_config = _fit(dataset, tmp_path, "l7", "--epochs", "3", "--lr-schedule", "cosine",
                               "--checkpoint-rule", "terminal")
    history = run_config["history"]
    assert [epoch["epoch"] for epoch in history] == [1, 2, 3]
    expected = [1e-3 * (1 + math.cos(math.pi * e / 3)) / 2 for e in range(3)]
    assert [epoch["lr"] for epoch in history] == pytest.approx(expected)
    assert run_config["selected_checkpoint_epoch"] == 3
    groups = (run_dir / "optimizer_groups.json").read_text()
    assert '"realised_steps_match": true' in groups


def test_l2_raw_distance_attributes_and_edge_radius(dataset, tmp_path):
    _run_dir, run_config = _fit(dataset, tmp_path, "l2", "--epochs", "1", "--rbf-use-raw-distances",
                                "--edge-radius", "8.0")
    config = run_config["config"]
    assert config["edge_radius"] == 8.0
    assert config["node_rbf_use_raw_distances"] and config["edge_rbf_use_raw_distances"]


def test_l2_graphs_keep_raw_distances_after_normalization(dataset):
    from training.data import load_training_pockets_with_report_from_dir
    from training.graph_dataset import PocketGraphDataset

    pockets = load_training_pockets_with_report_from_dir(
        dataset["structures"], required_targets=("ec",), summary_csv=dataset["summary"],
        esm_embeddings_dir=dataset["esm"], esm_dim=8, require_external_features=False, load_workers=1,
    ).pockets[:3]
    stats = PocketGraphDataset.fit_normalization_stats(pockets, esm_dim=8, edge_radius=8.0)
    graph = PocketGraphDataset(pockets, esm_dim=8, edge_radius=8.0, normalization_stats=stats)[0]
    assert hasattr(graph, "x_dist_raw_raw") and hasattr(graph, "edge_dist_raw_raw")


def test_l3_per_fit_generator_is_independent_of_the_global_rng(dataset):
    from training.data import load_training_pockets_with_report_from_dir
    from training.graph_dataset import augment_pocket_for_training

    pocket = load_training_pockets_with_report_from_dir(
        dataset["structures"], required_targets=("ec",), summary_csv=dataset["summary"],
        esm_embeddings_dir=dataset["esm"], esm_dim=8, require_external_features=False, load_workers=1,
    ).pockets[0]

    def draw(global_seed):
        torch.manual_seed(global_seed)
        generator = torch.Generator().manual_seed(123)
        augmented = augment_pocket_for_training(pocket, position_noise_std=0.5, outer_residue_dropout=0.5,
                                                generator=generator)
        return [coord.clone() for coord in augmented.metal_coords], len(augmented.residues)

    first, second = draw(1), draw(999)
    assert first[1] == second[1] and all(torch.equal(a, b) for a, b in zip(first[0], second[0]))
    torch.manual_seed(1)
    global_a = augment_pocket_for_training(pocket, position_noise_std=0.5).metal_coords[0]
    torch.manual_seed(1)
    global_b = augment_pocket_for_training(pocket, position_noise_std=0.5).metal_coords[0]
    assert torch.equal(global_a, global_b)


def test_l3_per_fit_fits_replay_and_refuse_workers(dataset, tmp_path):
    from training.config import parse_args

    _a, first = _fit(dataset, tmp_path, "l3a", "--epochs", "1", "--position-noise-std", "0.2",
                     "--augmentation-rng", "per_fit")
    _b, second = _fit(dataset, tmp_path, "l3b", "--epochs", "1", "--position-noise-std", "0.2",
                      "--augmentation-rng", "per_fit")
    assert first["history"][0]["train_loss"] == second["history"][0]["train_loss"]
    assert first["config"]["augmentation_rng"] == "per_fit"
    with pytest.raises(SystemExit):
        parse_args(["--task", "ec", "--augmentation-rng", "per_fit", "--num-workers", "2"])
