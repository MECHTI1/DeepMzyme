"""Input-level scientific and fail-closed checks for the bounded RING audit."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import audit_metal_ring_inputs as ring_audit
from graph.construction import pocket_to_pyg_data
from graph.ring_edges import ring_edges_output_path
from training.config import TrainConfig
from training.esm_feature_loading import embedding_path_candidates
from training.graph_dataset import compute_feature_normalization_stats
from training.runtime_preparation import updated_external_feature_path_candidates
from test_shell_role_source import pocket


def graph_pair(pocket):
    return [pocket_to_pyg_data(pocket, esm_dim=4, edge_radius=6, use_ring_edges=ring,
                               shell_role_source="geometry") for ring in (False, True)]


def test_graph_audit_counts_added_and_existing_annotated_pairs(pocket):
    path = Path(pocket.metadata["ring_edges_path"])
    with path.open("a") as stream:
        stream.write("A:1:_:ASP\tA:2:_:ASP\tHBOND:SC_SC\tOD1\tOD1\n")
    result = ring_audit.compare_graphs(*graph_pair(pocket), "fixture")
    assert result["radius_undirected_pairs"] == 2
    assert result["ring_added_undirected_pairs"] == 1
    assert result["ring_annotated_existing_undirected_pairs"] == 1
    assert result["ring_undirected_pairs"] == 2
    assert not result["no_ring_effect"]


@pytest.mark.parametrize("change,match", [
    ("node", "Node/site input changed"), ("site", "Node/site input changed"),
    ("nonfinite", "Nonfinite"), ("endpoint", "Out-of-range"),
    ("radius_geometry", "replaced existing radius geometry"),
])
def test_graph_audit_rejects_invalid_or_confounding_inputs(pocket, change, match):
    off, on = graph_pair(pocket)
    if change == "node":
        on.x_role[1, 1] = 0
    elif change == "site":
        on.site_metal_stats[0, 0] += 1
    elif change == "nonfinite":
        on.edge_vector_raw[0, 0] = float("nan")
    elif change == "endpoint":
        on.edge_index[0, 0] = on.num_nodes
    else:
        on.edge_dist_raw[0, 0] += .25
    with pytest.raises(ValueError, match=match):
        ring_audit.compare_graphs(off, on, "fixture")


def test_no_ring_effect_pocket_is_counted_without_removal(pocket):
    off, _ = graph_pair(pocket)
    assert ring_audit.compare_graphs(off, off.clone(), "fixture")["no_ring_effect"]


def test_normalizer_audit_allows_only_edge_distribution_changes(pocket):
    off, on = [compute_feature_normalization_stats([graph]) for graph in graph_pair(pocket)]
    result = ring_audit.compare_normalizers(off, on)
    assert result["non_edge_normalization_identical"]
    assert result["changed_edge_statistics"]
    on.means["x_misc"] += 1
    with pytest.raises(ValueError, match="Non-edge normalization changed"):
        ring_audit.compare_normalizers(off, on)


def configs(tmp_path):
    train = tmp_path / "data" / "dataset" / "train"
    train.mkdir(parents=True)
    summary = train / "sites.csv"
    summary.write_text("fixture\n")
    off = TrainConfig(
        structure_dir=train, summary_csv=summary, task="metal", model_architecture="only_gvp",
        metal_label_scheme="four_class", use_esm_branch=False, require_esm_embeddings=False,
        esm_embeddings_dir=None, esm_dim=4, use_ring_edges=False, require_ring_edges=False,
        shell_role_source="geometry", prepare_missing_ring_edges=False,
        prepare_missing_esm_embeddings=False, external_features_root_dir=tmp_path / "external",
        require_external_features=True, device="cpu", edge_radius=6,
    )
    return off, replace(off, use_ring_edges=True, require_ring_edges=True, ring_features_dir=tmp_path / "ring")


def test_cache_audit_hashes_without_loading_esm_and_requires_pinned_integrity(tmp_path, pocket, monkeypatch):
    _, config = configs(tmp_path)
    structure = config.structure_dir / "1abc__chain_A__EC_1.1.1.1.pdb"
    structure.write_text("structure fixture\n")
    monkeypatch.setattr(ring_audit, "resolve_structure_files", lambda *args, **kwargs: [structure])
    external = updated_external_feature_path_candidates(
        structure, structure_root=config.structure_dir,
        external_features_root_dir=config.external_features_root_dir,
    )[0]
    embedding = embedding_path_candidates(config.structure_dir.parent.parent / "esm_embeddings", structure)[0]
    ring = ring_edges_output_path(config.ring_features_dir, structure)
    for path, content in [(external, b'{"fixture": true}'), (embedding, b"not a tensor; hash only"),
                          (Path(str(embedding) + ".json"), b'{"fixture": true}'),
                          (ring, Path(pocket.metadata["ring_edges_path"]).read_bytes())]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    result = ring_audit.collect_training_cache(config)
    assert result["structures"] == 1 and len(result["files"]) == 4
    assert not result["esm_tensor_loading"]
    pin = {item["path"]: item for item in result["files"]}
    checked = ring_audit.collect_training_cache(config, expected_ring_files=pin, expected_feature_files=pin)
    assert checked["ring_provenance_pinned"] and checked["external_and_esm_provenance_pinned"]
    embedding.write_bytes(b"replacement")
    with pytest.raises(ValueError, match="checksum mismatch"):
        ring_audit.collect_training_cache(config, expected_ring_files=pin, expected_feature_files=pin)
    ring.write_text("NodeId1\tNodeId2\tInteraction\tAtom1\tAtom2\nBAD\tA:3:_:ASP\tHBOND:SC_SC\tOD1\tOD1\n")
    with pytest.raises(ValueError, match="Malformed RING node"):
        ring_audit.collect_training_cache(config)


def mock_preparation(tmp_path, pocket, monkeypatch):
    off, on = configs(tmp_path)
    ring_path = Path(pocket.metadata["ring_edges_path"]).resolve()
    info = ring_path.stat()
    cache = dict(structures=8, files=[dict(path=str(ring_path), bytes=info.st_size,
                 mtime_ns=info.st_mtime_ns, sha256=ring_audit.digest(ring_path), kind="ring")],
                 ring_files={}, missing_esm=0, missing_external=0, missing_ring=0,
                 ring_provenance_pinned=True, external_and_esm_provenance_pinned=True,
                 esm_tensor_loading=False)
    monkeypatch.setattr(ring_audit, "collect_training_cache", lambda *args, **kwargs: deepcopy(cache))
    expected = {"metal_label_scheme": "split_all_metals", "retained_split_identity": {}}
    samples = {}
    for part in ("train", "validation"):
        samples[part], examples = [], []
        for label in range(4):
            sample = deepcopy(pocket)
            sample.structure_id = f"{part}_{label}"
            sample.pocket_id = f"{part}_{label}_metal0"
            sample.y_metal = label
            samples[part].append(sample)
            examples.append(dict(structure_id=sample.structure_id, pocket_id=sample.pocket_id,
                                 group=sample.structure_id, y_metal=label))
        expected["retained_split_identity"][part] = {"examples": examples}

    def prepare(config):
        graphs = {part: [pocket_to_pyg_data(sample, esm_dim=4, edge_radius=6,
                     use_ring_edges=config.use_ring_edges, shell_role_source=config.shell_role_source)
                     for sample in subset] for part, subset in samples.items()}
        return SimpleNamespace(
            dataset_summary=deepcopy(expected),
            split=SimpleNamespace(train_pockets=samples["train"], val_pockets=samples["validation"]),
            train_loader=SimpleNamespace(dataset=SimpleNamespace(precomputed_data=graphs["train"])),
            val_loader=SimpleNamespace(dataset=SimpleNamespace(precomputed_data=graphs["validation"])),
            normalization_stats=compute_feature_normalization_stats(graphs["train"]),
        )

    monkeypatch.setattr(ring_audit, "prepare_run", prepare)
    return off, on, expected


def test_full_audit_binds_cohort_and_writes_verifiable_cache_receipt(tmp_path, pocket, monkeypatch):
    off, on, expected = mock_preparation(tmp_path, pocket, monkeypatch)
    output = tmp_path / "audit"
    result = ring_audit.audit(off, on, expected, output)
    assert result["status"] == "passed"
    assert result["splits"]["train"]["pockets"] == result["splits"]["validation"]["pockets"] == 4
    assert len(result["pockets"]) == 8
    assert result["all_node_and_site_tensors_identical"] and result["edge_endpoints_valid"]
    assert not result["training_performed"] and not result["held_out_evaluation"]
    assert result["cache_audit_sha256"] == ring_audit.digest(output / "training_cache_audit.json")
    cache = json.loads((output / "training_cache_audit.json").read_text())
    assert cache["status"] == "passed" and cache["failures"] == []
    # Failure must invalidate a previous success rather than leave it reusable.
    wrong = deepcopy(expected)
    wrong["retained_split_identity"]["validation"]["examples"][0]["group"] = "changed"
    with pytest.raises(ValueError, match="differs from parent"):
        ring_audit.audit(off, on, wrong, output)
    assert json.loads((output / "ring_input_audit.json").read_text())["status"] == "failed"
    assert json.loads((output / "training_cache_audit.json").read_text())["failures"]


def test_heldout_or_unmatched_configs_fail_before_any_input_access(tmp_path, monkeypatch):
    off, on = configs(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("Data preparation/cache access must not happen")

    monkeypatch.setattr(ring_audit, "prepare_run", forbidden)
    monkeypatch.setattr(ring_audit, "collect_training_cache", forbidden)
    with pytest.raises(ValueError, match="held-out"):
        ring_audit.audit(off, replace(on, test_structure_dir=tmp_path / "test"), {}, tmp_path / "audit")
    with pytest.raises(ValueError, match="Non-RING configuration mismatch"):
        ring_audit.audit(off, replace(on, learning_rate=.25), {}, tmp_path / "audit")
    with pytest.raises(ValueError, match="native six-class"):
        ring_audit.audit(off, on, {"metal_label_scheme": "four_class"}, tmp_path / "audit")
