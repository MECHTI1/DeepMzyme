"""Raw graph caching preserves geometry, labels, order and train-only statistics."""
from __future__ import annotations

import copy
import hashlib
import os
import random
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_structures import PocketRecord, ResidueRecord
from graph.construction import pocket_to_pyg_data
from label_schemes import configure_active_metal_label_scheme
from training import graph_dataset, raw_graph_cache as cache
from training.graph_dataset import PocketGraphDataset, build_graph_data_list, compute_feature_normalization_stats

OPTIONS = dict(esm_dim=8, edge_radius=8.0, shell_role_source="geometry")


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    root = tmp_path / "graphs"
    monkeypatch.setenv(cache.GRAPH_CACHE_ENV, str(root))
    # A tiny source tree exercises the real content hashing without repeatedly
    # scanning unrelated modules in every invalidation case.
    source = tmp_path / "src"
    source.mkdir()
    (source / "builder.py").write_text("# original implementation\n")
    monkeypatch.setattr(cache, "_SRC_ROOT", source)
    ring = tmp_path / "ring.tsv"
    ring.write_text("NodeId1\tNodeId2\tInteraction\tAtom1\tAtom2\n"
                    "A:1:_:ASP\tA:3:_:ASP\tHBOND:SC_SC\tOD1\tOD1\n")
    pockets = []
    for index in range(3):
        residues = []
        for number, distance in enumerate((2.0, 5.0, 8.0), start=1):
            residues.append(ResidueRecord(
                chain_id="A", resseq=number, icode="", resname="ASP",
                atoms={name: torch.tensor([distance + 2 * index, y, z]) for name, y, z in
                       (("N", 0.3, 0.), ("CA", 0., 0.), ("C", -0.3, 0.), ("O", -0.5, 0.),
                        ("CB", 0., 0.8), ("CG", 0., 0.3), ("OD1", 0., 0.), ("OD2", 0.4, 0.))},
                esm_embedding=torch.full((8,), float(number + index)), has_esm_embedding=True))
        pockets.append(PocketRecord(structure_id=f"fixture{index}", pocket_id=f"ion{index}",
                                    metal_element="ZN", metal_coords=[torch.zeros(3)], residues=residues,
                                    y_metal=index, y_ec=index,
                                    metadata={"source_uid": f"synthetic:{index}", "ring_edges_path": str(ring)}))
    yield pockets, root, source
    configure_active_metal_label_scheme("split_all_metals")


def assert_graph_equal(left, right):
    assert set(left.keys()) == set(right.keys())
    for key in left.keys():
        if isinstance(left[key], torch.Tensor):
            assert left[key].dtype == right[key].dtype
            assert torch.equal(left[key], right[key]), key
        else:
            assert left[key] == right[key], key


def count_builds(monkeypatch):
    calls = []

    def build(pocket, **options):
        calls.append(pocket.pocket_id)
        return pocket_to_pyg_data(pocket, **options)

    monkeypatch.setattr(graph_dataset, "pocket_to_pyg_data", build)
    return calls


def test_cache_hits_preserve_every_tensor_order_rng_and_train_only_normalization(fixture, monkeypatch):
    pockets, root, _ = fixture
    expected = [pocket_to_pyg_data(p, **OPTIONS) for p in pockets]
    calls = count_builds(monkeypatch)
    torch_rng, python_rng, numpy_rng = torch.get_rng_state(), random.getstate(), np.random.get_state()
    cold = build_graph_data_list(pockets, **OPTIONS)
    warm = build_graph_data_list(list(reversed(pockets)), **OPTIONS)
    assert calls == [p.pocket_id for p in pockets]
    assert len(list(root.rglob("*.pkl"))) == len(pockets)
    assert torch.equal(torch_rng, torch.get_rng_state()) and python_rng == random.getstate()
    current_numpy = np.random.get_state()
    assert numpy_rng[0] == current_numpy[0] and np.array_equal(numpy_rng[1], current_numpy[1])
    assert numpy_rng[2:] == current_numpy[2:]
    for a, b, c in zip(expected, cold, reversed(warm)):
        assert_graph_equal(a, b)
        assert_graph_equal(a, c)
    original_stats = compute_feature_normalization_stats(expected[:2])
    cached_stats = compute_feature_normalization_stats(cold[:2])
    all_stats = compute_feature_normalization_stats(cold)
    assert any(not torch.equal(cached_stats.means[k], all_stats.means[k]) for k in cached_stats.means)
    for key in original_stats.means:
        assert torch.equal(original_stats.means[key], cached_stats.means[key])
        assert torch.equal(original_stats.stds[key], cached_stats.stds[key])
    dataset = PocketGraphDataset(pockets, precomputed_data=cold, normalization_stats=cached_stats, **OPTIONS)
    reference = PocketGraphDataset(pockets, precomputed_data=expected, normalization_stats=original_stats, **OPTIONS)
    for index in range(len(pockets)):
        assert_graph_equal(dataset[index], reference[index])
    dataset[0].x_dist_raw.add_(100)
    cold[0].x_dist_raw.add_(200)
    assert_graph_equal(build_graph_data_list(pockets[:1], **OPTIONS)[0], expected[0])


@pytest.mark.parametrize("mutation", ["coordinates", "embedding", "embedding_dtype", "metal_target", "ec_target",
                                      "metadata", "external_feature"])
def test_pocket_content_changes_invalidate_cache(fixture, monkeypatch, mutation):
    pockets, root, _ = fixture
    pocket = pockets[0]
    calls = count_builds(monkeypatch)
    build_graph_data_list([pocket], **OPTIONS)
    changed = copy.deepcopy(pocket)
    if mutation == "coordinates":
        changed.residues[0].atoms["CA"][0] += 0.25
    elif mutation == "embedding":
        changed.residues[0].esm_embedding[0] += 0.25
    elif mutation == "embedding_dtype":
        changed.residues[0].esm_embedding = changed.residues[0].esm_embedding.to(torch.bfloat16)
    elif mutation == "metal_target":
        changed.y_metal = 3
    elif mutation == "ec_target":
        changed.y_ec = 3
    elif mutation == "metadata":
        changed.metadata.update(source_uid="different ion", ec_sample_weight=0.25)
    else:
        changed.residues[0].external_features["residue_sasa"] = 2.0
    actual = build_graph_data_list([changed], **OPTIONS)[0]
    assert len(calls) == 2 and len(list(root.rglob("*.pkl"))) == 2
    assert_graph_equal(actual, pocket_to_pyg_data(changed, **OPTIONS))


@pytest.mark.parametrize("option,value", [("edge_radius", 6.0), ("shell_role_source", "edge_mode"),
                                         ("metal_node_mode", "per_metal")])
def test_graph_options_invalidate_cache(fixture, monkeypatch, option, value):
    pockets, root, _ = fixture
    calls = count_builds(monkeypatch)
    build_graph_data_list(pockets[:1], **OPTIONS)
    changed = {**OPTIONS, option: value}
    actual = build_graph_data_list(pockets[:1], **changed)[0]
    assert len(calls) == 2 and len(list(root.iterdir())) == 2
    assert_graph_equal(actual, pocket_to_pyg_data(pockets[0], **changed))


@pytest.mark.parametrize("mutation", ["source", "torch_version", "pyg_version", "label_scheme"])
def test_source_libraries_and_target_scheme_invalidate_namespace(fixture, monkeypatch, mutation):
    pockets, root, source = fixture
    configure_active_metal_label_scheme("six_class")
    calls = count_builds(monkeypatch)
    build_graph_data_list(pockets[:1], **OPTIONS)
    if mutation == "source":
        (source / "builder.py").write_text("# changed implementation\n")
    elif mutation == "torch_version":
        monkeypatch.setattr(cache.torch, "__version__", "changed-version")
    elif mutation == "pyg_version":
        monkeypatch.setattr(cache.torch_geometric, "__version__", "changed-version")
    else:
        configure_active_metal_label_scheme("four_class")
    build_graph_data_list(pockets[:1], **OPTIONS)
    assert len(calls) == 2 and len(list(root.iterdir())) == 2


@pytest.mark.parametrize("mode", ["disabled", "use_ring_edges", "require_ring_edges"])
def test_disabled_cache_and_ring_never_use_namespace(fixture, monkeypatch, mode):
    pockets, root, _ = fixture
    options = dict(OPTIONS)
    if mode == "disabled":
        monkeypatch.delenv(cache.GRAPH_CACHE_ENV)
    else:
        options[mode] = True
    monkeypatch.setattr(cache, "_source_identity", lambda: pytest.fail("Cache should be bypassed"))
    actual = build_graph_data_list(pockets[:1], **options)[0]
    assert_graph_equal(actual, pocket_to_pyg_data(pockets[0], **options))
    assert not root.exists()


@pytest.mark.parametrize("mutation", ["truncated", "wrong_input_identity"])
def test_corrupt_or_misplaced_cache_entries_fail_without_rebuilding(fixture, monkeypatch, mutation):
    pockets, root, _ = fixture
    build_graph_data_list(pockets[:1], **OPTIONS)
    path = next(root.rglob("*.pkl"))
    if mutation == "truncated":
        path.write_bytes(path.read_bytes()[:-1])
        error = "integrity check failed"
    else:
        changed = copy.deepcopy(pockets[0])
        changed.y_metal = 3
        wrong_key = hashlib.sha256(cache._compact_bytes(changed)).hexdigest()
        path.rename(path.with_name(f"{wrong_key}.pkl"))
        pockets[0] = changed
        error = "input identity differs"
    monkeypatch.setattr(graph_dataset, "pocket_to_pyg_data", lambda *a, **k: pytest.fail("Corruption silently rebuilt"))
    with pytest.raises(ValueError, match=error):
        build_graph_data_list(pockets[:1], **OPTIONS)


def test_failed_atomic_write_leaves_no_partial_entry(fixture, monkeypatch):
    pockets, root, _ = fixture

    def fail_replace(*args):
        raise OSError("synthetic interrupted write")

    with monkeypatch.context() as patch:
        patch.setattr(cache.os, "replace", fail_replace)
        with pytest.raises(OSError, match="interrupted write"):
            build_graph_data_list(pockets[:1], **OPTIONS)
    assert not list(root.rglob("*.pkl")) and not list(root.rglob("*.tmp"))
    actual = build_graph_data_list(pockets[:1], **OPTIONS)[0]
    assert_graph_equal(actual, pocket_to_pyg_data(pockets[0], **OPTIONS))


def test_cache_content_key_is_stable_in_a_fresh_process(fixture):
    pockets, _, source = fixture
    configure_active_metal_label_scheme("split_all_metals")
    build_graph_data_list(pockets[:1], **OPTIONS)
    script = """
import pickle, sys
from pathlib import Path
from training import raw_graph_cache as cache, graph_dataset
pocket, options, source = pickle.loads(sys.stdin.buffer.read())
cache._SRC_ROOT = Path(source)
def unexpected_build(*args, **kwargs):
    raise AssertionError('Fresh process failed to reuse the same raw pocket content')
graph_dataset.pocket_to_pyg_data = unexpected_build
assert len(graph_dataset.build_graph_data_list([pocket], **options)) == 1
"""
    result = subprocess.run([sys.executable, "-c", script],
                            input=cache._compact_bytes((pockets[0], OPTIONS, str(source))),
                            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
                            capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr.decode()
    assert "hits=1 misses=0" in result.stdout.decode()


def test_cache_reports_bounded_progress_and_final_totals(fixture, capsys):
    pockets, _, _ = fixture
    build_graph_data_list([pockets[0]] * 501, **OPTIONS)
    output = capsys.readouterr().out
    assert output.count("[GRAPH-CACHE]") == 2
    assert "processed=500/501 hits=499 misses=1" in output
    assert "hits=500 misses=1" in output
