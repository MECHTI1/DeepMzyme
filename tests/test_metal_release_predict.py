"""Synthetic explicit-site release regressions; no real training or held-out data."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import copy
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from metal_release import predict
from metal_release.manifest import ReleaseError, sha256_file
from pmm_campaign_fixtures import write_pdb


def _artifact(path):
    return {"path": str(path), "sha256": sha256_file(path)}


@pytest.fixture
def release(tmp_path_factory, monkeypatch):
    import label_schemes
    from data_structures import EXTERNAL_FEATURE_NAMES
    from embed_helpers.esmc import extract_chain_sequences
    from export_validation_predictions import factory_kwargs
    from graph.structure_parsing import parse_structure_file
    from model_variants import build_pocket_classifier
    from training.config import TrainConfig

    tmp_path = tmp_path_factory.mktemp("metal_release_predict")
    old_threads = torch.get_num_threads()
    old_scheme = label_schemes.ACTIVE_METAL_LABEL_SCHEME
    torch.set_num_threads(1)
    monkeypatch.delenv("DEEPMZYME_GRAPH_CACHE_DIR", raising=False)
    structure = tmp_path / "fixture.pdb"
    write_pdb(structure, ions=[("A", 201, "", "ZN", (0, 0, 0))], chains=("A", "B"))
    cache = tmp_path / "cache"
    parsed = parse_structure_file(str(structure))
    for sequence in extract_chain_sequences(parsed).values():
        embedding = torch.arange(len(sequence) * 8, dtype=torch.float32).reshape(len(sequence), 8) / 100
        path = predict._sequence_cache_path(cache, "synthetic_esmc", sequence, predict._embedding_profile("cpu"))
        predict._save_sequence_cache(path, embedding, sequence=sequence, model_name="synthetic_esmc")
    members, groups = {}, []
    for target, architecture in (("four_class", "gvp"), ("five_class", "only_esm")):
        ids = []
        for fold in range(2):
            member_id = f"{architecture}_{target}_{fold}"
            ids.append(member_id)
            directory = tmp_path / member_id
            directory.mkdir()
            config = json.loads(json.dumps(asdict(TrainConfig(
                task="metal", metal_example_unit="ion", metal_label_scheme=target, model_architecture=architecture,
                checkpoint_rule="terminal",
                n_folds=2, fold_index=fold, seed=42, esm_dim=8, hidden_s=16, hidden_v=4, edge_hidden=8,
                gvp_layers=1, esm_fusion_dim=8, head_mlp_layers=1, shell_role_source="geometry",
                omit_node_features=tuple(EXTERNAL_FEATURE_NAMES), require_esm_embeddings=True,
                use_ring_edges=False, require_ring_edges=False, require_external_features=False,
            )), default=str))
            labels = label_schemes.metal_labels_for_scheme(target)
            checkpoint = {"config": config, "metal_labels": labels, "ec_labels": {}, "model_state_dict": {}}
            torch.manual_seed(11 + fold)
            kwargs = factory_kwargs(ROOT / "src/training/run.py", config, checkpoint, labels, {})
            model = build_pocket_classifier(**kwargs)
            checkpoint["model_state_dict"] = model.state_dict()
            checkpoint["normalization_stats"] = {
                "means": {"x_dist_raw": torch.full((1, 3), float(fold + 1))},
                "stds": {"x_dist_raw": torch.full((1, 3), float(fold + 2))}, "clamp_value": 5.0,
            }
            checkpoint_path = directory / "checkpoint.pt"
            torch.save(checkpoint, checkpoint_path)
            config_path = directory / "run_config.json"
            config_path.write_text(json.dumps({"config": config}))
            receipt_path = directory / "selected_checkpoint.json"
            receipt_path.write_text(json.dumps({"selected_checkpoint_sha256": sha256_file(checkpoint_path)}))
            predictions_path = directory / "val_predictions.csv"
            predictions_path.write_text("source_uid,p_common4_Mn\nsynthetic,0.25\n")
            members[member_id] = {"target": target, "fold": fold, "seed": 42,
                "checkpoint": _artifact(checkpoint_path), "run_config": _artifact(config_path),
                "selected_checkpoint": _artifact(receipt_path), "validation_predictions": _artifact(predictions_path)}
        groups.append({"id": architecture, "weight": 1.0 if target == "four_class" else 0.5, "members": ids})
    manifest = {"schema": "deepmzyme-metal-release-v1", "reference": "mixed", "release_status": "provisional",
        "n_folds": 2, "embedding_model": "synthetic_esmc", "members": members, "candidates": {"mixed": {"groups": groups}}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    yield path, structure, cache, manifest
    label_schemes.configure_active_metal_label_scheme(old_scheme)
    torch.set_num_threads(old_threads)


def _scores(report):
    return np.asarray([list(site["class_scores"].values()) for site in report["sites"]])


def test_observed_atom_equals_center_and_mixed_targets_restore_state(release):
    import label_schemes

    path, structure, cache, _ = release
    before = label_schemes.ACTIVE_METAL_LABEL_SCHEME
    report = predict.predict_sites(path, structure, [
        {"id": "located", "atom": "A:201:_:ZN"}, {"id": "center", "center": [0, 0, 0]},
    ], embedding_cache=cache, include_member_predictions=True)
    np.testing.assert_array_equal(_scores(report)[0], _scores(report)[1])
    assert label_schemes.ACTIVE_METAL_LABEL_SCHEME == before
    assert report["member_count"] == 4 and report["device"] == "cpu"
    assert report["calibration"] == "none"
    assert report["runtime"]["compatibility_status"].startswith("unverified")
    assert report["memory"]["peak_process_rss_bytes"] > 0
    assert report["embeddings"]["generated_sequences"] == 0
    assert all(site["predicted_class"] in report["class_order"] for site in report["sites"])
    assert len(report["member_predictions"]) == 4
    assert all(len(item["native_predictions"]) == 2 for item in report["member_predictions"].values())


def test_parsed_symbol_and_labels_do_not_change_inputs_or_predictions(release, tmp_path):
    from training.graph_dataset import build_graph_data_list

    path, structure, cache, _ = release
    altered = tmp_path / "changed.pdb"
    write_pdb(altered, ions=[("A", 201, "", "FE", (0, 0, 0))], chains=("A", "B"))
    _, a, _ = predict.build_site_pocket(structure, center=[0, 0, 0])
    _, b, _ = predict.build_site_pocket(altered, center=[0, 0, 0])
    assert a.y_metal is b.y_metal is None and a.y_ec is b.y_ec is None
    ga, gb = build_graph_data_list([a, b], esm_dim=8, shell_role_source="geometry")
    for key in ga.keys():
        if isinstance(ga[key], torch.Tensor):
            assert torch.equal(ga[key], gb[key]), key
    ra = predict.predict_structure(path, structure, center=[0, 0, 0], embedding_cache=cache)
    rb = predict.predict_structure(path, altered, center=[0, 0, 0], embedding_cache=cache)
    np.testing.assert_array_equal(_scores(ra), _scores(rb))


def test_cached_uncached_batch_and_member_normalization_are_equivalent(release, monkeypatch, tmp_path):
    path, structure, cache, _ = release
    sites = [{"id": "a", "center": [0, 0, 0]}, {"id": "b", "center": [0.25, 0, 0]}]
    uncached = predict.predict_sites(path, structure, sites, embedding_cache=cache, batch_size=1)
    monkeypatch.setenv("DEEPMZYME_GRAPH_CACHE_DIR", str(tmp_path / "graphs"))
    cold = predict.predict_sites(path, structure, sites, embedding_cache=cache, batch_size=2)
    warm = predict.predict_sites(path, structure, sites, embedding_cache=cache, batch_size=2)
    np.testing.assert_allclose(_scores(uncached), _scores(cold), atol=2e-6, rtol=0)
    np.testing.assert_array_equal(_scores(cold), _scores(warm))
    single = predict.predict_structure(path, structure, center=sites[1]["center"], embedding_cache=cache)
    np.testing.assert_allclose(_scores(single)[0], _scores(uncached)[1], atol=2e-6, rtol=0)


@pytest.mark.parametrize("fault", ["wrong_sequence", "wrong_model", "nonfinite", "wrong_shape"])
def test_invalid_embeddings_fail_before_model_execution(release, fault):
    path, structure, cache, _ = release
    for embedding_path in (cache / "synthetic_esmc").rglob("*.pt"):
        payload = torch.load(embedding_path, weights_only=True)
        if fault == "wrong_sequence":
            payload["metadata"]["source_sequence_sha256"] = "0" * 64
        elif fault == "wrong_model":
            payload["metadata"]["esm_model_name"] = "different_model"
        elif fault == "nonfinite":
            payload["embeddings"][0, 0] = float("nan")
        else:
            payload["embeddings"] = payload["embeddings"][:, :3]
        torch.save(payload, embedding_path)
    with pytest.raises(ReleaseError, match="embedding|Embedding"):
        predict.predict_structure(path, structure, center=[0, 0, 0], embedding_cache=cache)


def test_missing_embeddings_do_not_implicitly_generate(release, monkeypatch, tmp_path):
    from embed_helpers import esmc

    path, structure, _, _ = release
    monkeypatch.setattr(esmc, "_esmc_sdk", lambda *a, **k: pytest.fail("Unexpected ESM generation"))
    with pytest.raises(ReleaseError, match="Missing verified ESM"):
        predict.predict_structure(path, structure, center=[0, 0, 0], embedding_cache=tmp_path / "empty")


def test_explicit_generation_reuses_sequence_cache_without_backbone_reload(release, monkeypatch, tmp_path):
    from embed_helpers import esmc

    path, structure, _, _ = release
    loads, sequences = [], []

    class FakeESMC(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.ones(1))

        @classmethod
        def from_pretrained(cls, model_name, *, device):
            assert device == torch.device("cpu")
            loads.append((str(device), model_name))
            return cls()

        def encode(self, protein):
            return protein

        def logits(self, protein, config):
            assert self.parameter.dtype == torch.bfloat16
            assert self.parameter.device.type == "cpu" and not self.training
            sequences.append(protein.sequence)
            values = torch.arange(len(protein.sequence) * 8, dtype=torch.float32).reshape(-1, 8) / 100
            return SimpleNamespace(embeddings=values)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a: pytest.fail("CPU requested CUDA allocation"))
    monkeypatch.setattr(esmc, "_esmc_sdk", lambda: (FakeESMC, SimpleNamespace, lambda **kwargs: kwargs))
    cache = tmp_path / "generated"
    generated = predict.predict_structure(path, structure, center=[0, 0, 0], embedding_cache=cache,
                                          generate_embeddings=True)
    warm = predict.predict_structure(path, structure, center=[0, 0, 0], embedding_cache=cache)
    assert loads == [("cpu", "synthetic_esmc")]
    assert generated["embeddings"]["generated_sequences"] == len(set(sequences)) == 2
    assert warm["embeddings"]["generated_sequences"] == 0
    assert generated["embeddings"]["model_dtype"] == "bfloat16"
    assert generated["runtime"]["torch_num_threads"] == 1
    np.testing.assert_array_equal(_scores(generated), _scores(warm))


@pytest.mark.parametrize("field", ["use_ring_edges", "require_external_features"])
def test_unimplemented_required_features_fail_instead_of_using_defaults(release, field):
    path, structure, cache, manifest = release
    member = next(iter(manifest["members"].values()))
    config_path = Path(member["run_config"]["path"])
    config = json.loads(config_path.read_text())
    config["config"][field] = True
    config_path.write_text(json.dumps(config))
    member["run_config"] = _artifact(config_path)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ReleaseError, match="adapter"):
        predict.predict_structure(path, structure, center=[0, 0, 0], embedding_cache=cache)


def test_mmCIF_matches_pdb_and_altloc_is_explicit(release, tmp_path):
    from Bio.PDB import MMCIFIO, PDBParser

    path, structure, cache, _ = release
    cif = tmp_path / "fixture.cif"
    writer = MMCIFIO()
    writer.set_structure(PDBParser(QUIET=True).get_structure("fixture", structure))
    writer.save(str(cif))
    a = predict.predict_structure(path, structure, atom="A:201:_:ZN", embedding_cache=cache)
    b = predict.predict_structure(path, cif, atom="A:201:_:ZN", embedding_cache=cache)
    np.testing.assert_array_equal(_scores(a), _scores(b))
    alternate = tmp_path / "altloc.pdb"
    write_pdb(alternate, ions=[("A", 201, "", "ZN", (0, 0, 0))], altloc_ion=True)
    with pytest.raises(ReleaseError, match="unambiguous altloc"):
        predict.build_site_pocket(alternate, atom="A:201:_:ZN")
    _, _, report = predict.build_site_pocket(alternate, atom="A:201:_:ZN:B")
    np.testing.assert_allclose(report["center_angstrom"], [.8, 0, 0])


def test_checkpoint_hash_and_config_mismatch_fail_closed(release):
    path, structure, cache, manifest = release
    member = next(iter(manifest["members"].values()))
    checkpoint_path = Path(member["checkpoint"]["path"])
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    checkpoint["config"]["seed"] = 13
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ReleaseError, match="SHA-256 mismatch"):
        predict.predict_structure(path, structure, center=[0, 0, 0], embedding_cache=cache)
    member["checkpoint"]["sha256"] = sha256_file(checkpoint_path)
    receipt_path = Path(member["selected_checkpoint"]["path"])
    receipt_path.write_text(json.dumps({"selected_checkpoint_sha256": member["checkpoint"]["sha256"]}))
    member["selected_checkpoint"] = _artifact(receipt_path)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ReleaseError, match="configurations differ"):
        predict.predict_structure(path, structure, center=[0, 0, 0], embedding_cache=cache)


def test_tuple_checkpoint_config_matches_json_without_ignoring_fields(release):
    path, structure, cache, manifest = release
    for member in manifest["members"].values():
        checkpoint_path = Path(member["checkpoint"]["path"])
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        for field in ("final_test_checkpoint_paths", "final_test_seed_values", "final_test_source_run_dirs", "omit_node_features"):
            checkpoint["config"][field] = tuple(checkpoint["config"][field])
        torch.save(checkpoint, checkpoint_path)
        member["checkpoint"] = _artifact(checkpoint_path)
        receipt_path = Path(member["selected_checkpoint"]["path"])
        receipt_path.write_text(json.dumps({"selected_checkpoint_sha256": member["checkpoint"]["sha256"]}))
        member["selected_checkpoint"] = _artifact(receipt_path)
    path.write_text(json.dumps(manifest))
    report = predict.predict_structure(path, structure, center=[0, 0, 0], embedding_cache=cache)
    assert report["member_count"] == 4


def test_blocked_manifest_invalid_sites_and_unsupported_features(release):
    path, structure, cache, manifest = release
    manifest["release_status"] = "blocked"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ReleaseError, match="blocked"):
        predict.predict_structure(path, structure, center=[0, 0, 0], embedding_cache=cache)
    with pytest.raises(ReleaseError, match="exactly one"):
        predict.build_site_pocket(structure, center=[0, 0, 0], atom="A:201:_:ZN")
    with pytest.raises(ReleaseError, match="finite"):
        predict.build_site_pocket(structure, center=[float("nan"), 0, 0])
    with pytest.raises(ReleaseError, match="no usable"):
        predict.build_site_pocket(structure, center=[1000, 0, 0])


def test_parity_gate_requires_every_eligible_member_and_bound_fixture_identity(release, tmp_path):
    from metal_release.manifest import COMMON4, NATIVE
    from metal_release.parity import compare_development_parity

    path, _, _, manifest = release
    fixtures = [{"source_uid": f"dev{fold}", "physical_ion_id": f"pdb{fold}|m0|A|201|-|ZN|-",
                 "fold": fold, "site_id": "site_1", "structure_sha256": str(fold) * 64,
                 "center_angstrom": [0.0, 0.0, 0.0]} for fold in range(2)]
    membership = path.parent / "folds.csv"
    with membership.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_uid", "physical_ion_id", "fold"])
        writer.writeheader()
        writer.writerows({key: fixture[key] for key in writer.fieldnames} for fixture in fixtures)
    manifest["fold_membership"] = _artifact(membership)
    replayed = {}
    for member_id, member in manifest["members"].items():
        native = [.1, .2, .6, .1] if member["target"] == "four_class" else [.1, .2, .6, .04, .06]
        common = [.1, .2, .6, .1]
        fixture = fixtures[member["fold"]]
        row = {"source_uid": fixture["source_uid"], "physical_ion_id": fixture["physical_ion_id"],
               "fold": member["fold"], "model_seed": 42, "checkpoint_sha256": member["checkpoint"]["sha256"],
               "pred_native": 2, "pred_common4": 2,
               **{f"p_native_{label.replace(' ', '_')}": value for label, value in zip(NATIVE[member["target"]], native)},
               **{f"p_common4_{label.replace(' ', '_')}": value for label, value in zip(COMMON4, common)}}
        predictions_path = Path(member["validation_predictions"]["path"])
        with predictions_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        member["validation_predictions"] = _artifact(predictions_path)
        replayed[member_id] = {"native_probabilities": [native], "common4_probabilities": [common],
            "native_predictions": [2], "common4_predictions": [2], "checkpoint_sha256": member["checkpoint"]["sha256"]}
    path.write_text(json.dumps(manifest))
    reports = [{"structure_sha256": fixture["structure_sha256"], "candidate": "mixed",
                "manifest_sha256": sha256_file(path), "sites": [{"id": "site_1", "center_angstrom": [0, 0, 0]}],
                "runtime": {"source_sha256": {"model.py": "a" * 64}, "packages": {"torch": "fixture"}},
                "member_predictions": copy.deepcopy(replayed)} for fixture in fixtures]
    result = compare_development_parity(path, reports, fixtures)
    assert result["status"] == "passed" and len(result["covered_members"]) == 4
    assert result["max_abs_difference"] == 0
    incomplete = compare_development_parity(path, reports[:1], fixtures[:1])
    assert incomplete["status"] == "blocked" and len(incomplete["uncovered_members"]) == 2
    altered = copy.deepcopy(reports)
    member_id = next(key for key, member in manifest["members"].items() if member["fold"] == 1)
    # A fold-1 model's dev0 output is ineligible and cannot influence this check.
    altered[0]["member_predictions"][member_id]["common4_probabilities"] = [[.9, .05, .04, .01]]
    assert compare_development_parity(path, altered, fixtures)["status"] == "passed"
    altered[1]["member_predictions"][member_id]["common4_probabilities"] = [[.9, .05, .04, .01]]
    assert compare_development_parity(path, altered, fixtures)["status"] == "blocked"
    wrong_identity = copy.deepcopy(fixtures)
    wrong_identity[0]["physical_ion_id"] = "different physical ion"
    assert compare_development_parity(path, reports, wrong_identity)["status"] == "blocked"
    missing_provenance = copy.deepcopy(fixtures)
    del missing_provenance[0]["structure_sha256"]
    assert compare_development_parity(path, reports, missing_provenance)["status"] == "blocked"


def test_parity_runner_preflight_failure_persistence_and_no_overwrite(release, monkeypatch):
    from embed_helpers.esmc import extract_chain_sequences
    from graph.structure_parsing import parse_structure_file
    from metal_release.parity import _load_development_fixture, run_development_parity
    from metal_release.manifest import load_manifest
    from training.esm_feature_loading import build_embedding_payload, residue_keys_for_structure_chain

    path, structure, _, manifest = release
    root = path.parent
    source = {"source_uid": "development_fixture", "physical_ion_id": "fixture|m0|A|201|-|ZN|-",
              "structure_name": structure.name, "structure_sha256": sha256_file(structure),
              "coord_x": 0, "coord_y": 0, "coord_z": 0, "context_chains": "A;B"}

    def write_rows(filename, rows):
        output = root / filename
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return _artifact(output)

    cohort = write_rows("train_cohort.csv", [source])
    folds = write_rows("fold_membership.csv", [{"source_uid": source["source_uid"],
        "physical_ion_id": source["physical_ion_id"], "fold": 0}])
    structures = write_rows("structure_manifest.csv", [{"structure_name": structure.name,
        "relative_path": structure.name, "sha256": sha256_file(structure)}])
    manifest["fold_membership"] = folds
    for member in manifest["members"].values():
        config_path = Path(member["run_config"]["path"])
        config = json.loads(config_path.read_text())
        config["config"]["source_cohort_sha256"] = cohort["sha256"]
        config_path.write_text(json.dumps(config))
        member["run_config"] = _artifact(config_path)
    path.write_text(json.dumps(manifest))
    parsed = parse_structure_file(str(structure))
    legacy = root / "historical_embeddings"
    legacy.mkdir()
    embeddings = {}
    for chain, sequence in extract_chain_sequences(parsed).items():
        keys, _ = residue_keys_for_structure_chain(parsed, chain)
        output = legacy / f"{structure.stem}_chain_{chain}_esmc.pt"
        payload = build_embedding_payload(torch.ones(len(sequence), 8), keys, chain_id=chain, metadata={
            "esm_model_name": "synthetic_esmc", "model_dtype": "bfloat16",
            "source_sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest()})
        torch.save(payload, output)
        sidecar = output.with_name(output.name + ".json")
        sidecar.write_text(json.dumps(payload["metadata"]))
        embeddings[chain] = {"payload": _artifact(output), "sidecar": _artifact(sidecar)}
    fixture = {"source_uid": source["source_uid"], "physical_ion_id": source["physical_ion_id"],
        "fold": 0, "site_id": "site_1", "center_angstrom": [0, 0, 0], "structure": _artifact(structure),
        "embedding_cache": str(legacy), "embeddings": embeddings, "context_chains": ["A", "B"]}
    fixture_path = root / "development_fixture.json"
    fixture_payload = {"split": "development", "fixtures": [fixture], "source_cohort": cohort,
                       "fold_membership": folds, "structure_manifest": structures}
    fixture_path.write_text(json.dumps(fixture_payload))
    loaded, _ = _load_development_fixture(load_manifest(path), fixture_path)
    assert loaded[0]["structure_sha256"] == sha256_file(structure)
    bad = copy.deepcopy(fixture_payload)
    bad["fixtures"][0]["center_angstrom"] = [1, 0, 0]
    fixture_path.write_text(json.dumps(bad))
    with pytest.raises(ReleaseError, match="center differs"):
        run_development_parity(path, fixture_path, root / "bad_output")
    assert not (root / "bad_output").exists()
    fixture_path.write_text(json.dumps(fixture_payload))
    monkeypatch.setattr(predict, "predict_sites", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("simulated inference failure")))
    result = run_development_parity(path, fixture_path, root / "parity_output")
    assert result["status"] == "blocked" and "simulated inference failure" in result["errors"][0]
    assert json.loads((root / "parity_output/parity.json").read_text())["status"] == "blocked"
    with pytest.raises(ReleaseError, match="already exists"):
        run_development_parity(path, fixture_path, root / "parity_output")
