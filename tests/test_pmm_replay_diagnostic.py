"""Synthetic tests for the append-only replay diagnostic; no campaign data/GPU."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Batch, Data

spec = importlib.util.spec_from_file_location("audit_pmm_replay", Path(__file__).parents[1] / "audit_pmm_replay.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.mark.parametrize("change", ["dtype", "shape", "order", "metadata", "edge_order", "signed_zero"])
def test_fingerprints_detect_input_changes(change):
    graph = Data(x=torch.tensor([[0., 1.], [2., 3.]]),
                 edge_index=torch.tensor([[0, 1], [1, 0]]), metadata={"uid": "one"})
    other = graph.clone()
    if change == "dtype":
        other.x = other.x.double()
    elif change == "shape":
        other.x = other.x.reshape(1, 4)
    elif change == "order":
        other.x = other.x.flip(0)
    elif change == "metadata":
        other.metadata["uid"] = "two"
    elif change == "edge_order":
        other.edge_index = other.edge_index.flip(1)
    else:
        other.x[0, 0] = -0.
    assert audit.graph_hash(graph) != audit.graph_hash(other)


def test_ec_metadata_exclusion_is_explicit_and_does_not_hide_targets():
    graph = Data(x=torch.ones(2, 3), y_metal=torch.tensor([1]), ec_group_id=torch.tensor([12]))
    other = graph.clone()
    other.ec_group_id.fill_(-1)
    assert audit.graph_hash(graph) != audit.graph_hash(other)
    assert audit.graph_hash(graph, audit.EC_METADATA) == audit.graph_hash(other, audit.EC_METADATA)
    assert set(audit.field_differences(audit.graph_fields(graph), audit.graph_fields(other))) == {"ec_group_id"}
    other.y_metal.fill_(2)
    assert audit.graph_hash(graph, audit.EC_METADATA) != audit.graph_hash(other, audit.EC_METADATA)


def cache_file(tmp_path, *, key="a" * 64, stored_key=None):
    path = tmp_path / f"{key}.pkl"
    payload = pickle.dumps((stored_key or key, Data(x=torch.ones(2, 3))))
    path.write_bytes(hashlib.sha256(payload).hexdigest().encode() + b"\n" + payload)
    return path


def test_cache_read_detects_corruption_and_wrong_identity(tmp_path):
    path = cache_file(tmp_path)
    graph, digest = audit.read_cache(path)
    assert torch.equal(graph.x, torch.ones(2, 3)) and digest == audit.file_sha(path)
    path.write_bytes(path.read_bytes()[:-3])
    with pytest.raises(ValueError, match="Corrupt"):
        audit.read_cache(path)
    path = cache_file(tmp_path, stored_key="b" * 64)
    with pytest.raises(ValueError, match="Misidentified"):
        audit.read_cache(path)


def test_output_is_new_and_cannot_overlap_input(tmp_path):
    run = tmp_path / "original_run"
    run.mkdir()
    (run / "receipt.json").write_text("unchanged")
    for unsafe in (run, run / "new", tmp_path):
        with pytest.raises(ValueError, match="overlaps"):
            audit.new_output_dir(unsafe, (run,))
    output = audit.new_output_dir(tmp_path / "diagnostic", (run,))
    with pytest.raises(FileExistsError):
        audit.new_output_dir(output, (run,))
    assert (run / "receipt.json").read_text() == "unchanged"


class TinyModel(torch.nn.Module):
    def __init__(self, mutation=None):
        super().__init__()
        self.register_buffer("offset", torch.zeros(()))
        self.mutation = mutation
        self.predict_ec = False
        self.predict_metal = True

    def forward(self, batch):
        if self.mutation == "model":
            self.offset.add_(1)
        if self.mutation == "input":
            batch.x.add_(1)
        logits = batch.x[batch.ptr[:-1]] + self.offset
        return {"logits_metal": logits, "loss": logits.sum() * 0}


def tiny_batches():
    return [Batch.from_data_list([Data(x=torch.tensor([[1., 2., 0., -1., -2., -3.]]),
                                      y_metal=torch.tensor([1]))])]


def test_fixed_input_repeats_preserve_state_and_original_snapshot():
    from label_schemes import configure_active_metal_label_scheme

    configure_active_metal_label_scheme("six_class")
    batches = tiny_batches()
    before = audit.graph_hash(batches[0])
    passes, hashes = audit.repeated_forward(TinyModel(), batches, device="cpu", repeats=3)
    assert len(passes) == 3
    assert all(torch.equal(passes[0]["native"], item["native"]) for item in passes)
    assert audit.graph_hash(batches[0]) == before == hashes["input_hashes_before_and_after"][0]


@pytest.mark.parametrize("mutation,expected", [("model", "Model state"), ("input", "mutated an input")])
def test_repeats_fail_on_mutable_model_or_input(mutation, expected):
    batches = tiny_batches()
    before = audit.graph_hash(batches[0])
    with pytest.raises(ValueError, match=expected):
        audit.repeated_forward(TinyModel(mutation), batches, device="cpu", repeats=2)
    assert audit.graph_hash(batches[0]) == before


def test_collapse_sums_probabilities_before_argmax_and_reports_changed_predictions():
    from label_schemes import configure_active_metal_label_scheme

    configure_active_metal_label_scheme("six_class")
    native, common = audit.probability_views(torch.tensor([[.3, .1, .1, .2, .15, .15]]).log())
    assert native.argmax(1).tolist() == [0] and common.argmax(1).tolist() == [3]
    assert torch.allclose(common, torch.tensor([[.3, .1, .1, .5]]))
    comparison = audit.compare_probabilities(common, torch.tensor([[.6, .1, .1, .2]]))
    assert comparison["changed_predictions"] == 1
    assert comparison["probability_fields_above_original_1e_6"] == 2
    with pytest.raises(ValueError, match="Nonfinite"):
        audit.probability_views(torch.full((1, 6), float("nan")))


def test_saved_predictions_reject_duplicate_or_missing_uids(tmp_path):
    path = tmp_path / "predictions.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_uid"])
        writer.writeheader()
        writer.writerows([{"source_uid": "one"}, {"source_uid": "one"}])
    with pytest.raises(ValueError, match="duplicated"):
        audit.saved_rows(path, ["one", "two"])


def test_time_window_requires_explicit_timezone():
    assert audit.utc_timestamp("2026-09-27T18:14:00Z") < audit.utc_timestamp("2026-09-27T18:54:00Z")
    with pytest.raises(ValueError, match="timezone"):
        audit.utc_timestamp("2026-09-27T18:14:00")


def test_prepare_capture_and_cpu_evaluate_preserve_original_files(tmp_path, monkeypatch):
    """Exercise both diagnostic phases; stub only expensive frozen reconstruction."""
    from torch_geometric.loader import DataLoader
    from training import campaign_runtime
    from training.graph_dataset import FeatureNormalizationStats, PocketGraphDataset
    from label_schemes import configure_active_metal_label_scheme

    configure_active_metal_label_scheme("six_class")
    run, namespace = tmp_path / "run", tmp_path / "cache"
    run.mkdir()
    namespace.mkdir()
    train = tmp_path / "dataset" / "train"
    train.mkdir(parents=True)
    cohort, fold = tmp_path / "cohort.csv", tmp_path / "fold.csv"
    cohort.write_text("synthetic cohort")
    fold.write_text("synthetic fold")
    config = {"task": "metal", "batch_size": 16, "seed": 42, "deterministic": False,
              "campaign_run_identity": json.dumps({"source_tree_sha256": "synthetic"}),
              "structure_dir": str(train), "source_cohort_csv": str(cohort),
              "source_cohort_sha256": audit.file_sha(cohort), "fold_membership_csv": str(fold),
              "fold_membership_sha256": audit.file_sha(fold)}
    (run / "run_config.json").write_text(json.dumps({"config": config}))
    (run / "selected_checkpoint.json").write_text("original receipt")
    graphs = [Data(x=torch.tensor([[float(i), 2., 0., -1., -2., -3.]]),
                   pos=torch.tensor([[float(i), 0., 0.]]), y_metal=torch.tensor([1]),
                   y_ec=torch.tensor([-1]),
                   ec_group_id=torch.tensor([-1]), ec_sample_weight=torch.tensor([1.])) for i in range(2)]
    pockets = [SimpleNamespace(metadata={"source_uid": f"uid-{i}"}) for i in range(2)]
    normalization = FeatureNormalizationStats(means={"x": torch.zeros(1, 6)}, stds={"x": torch.ones(1, 6)})
    norm_payload = {"means": normalization.means, "stds": normalization.stds}
    torch.save({"config": config, "normalization_stats": norm_payload}, run / "best_model_checkpoint.pt")
    dataset = PocketGraphDataset(pockets, esm_dim=6, precomputed_data=graphs, normalization_stats=normalization)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    model = TinyModel()
    model.predict_ec = False
    native, common = audit.probability_views(torch.cat([graph.x for graph in graphs]))
    rows = []
    for index in range(2):
        row = {"source_uid": f"uid-{index}", "y_native": "1"}
        for prefix, values, labels in (("p_native_", native, ("Mn", "Cu", "Zn", "Fe", "Co", "Ni")),
                                      ("p_common4_", common, ("Mn", "Cu", "Zn", "Class_VIII"))):
            row.update({prefix + label: f"{float(value):.8f}" for label, value in zip(labels, values[index])})
        rows.append(row)
    with (run / "val_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    originals = {path: audit.file_sha(path) for path in run.iterdir()}
    for index, graph in enumerate(graphs):
        old = graph.clone()
        old.ec_group_id.fill_(10 + index)
        key = str(index) * 64
        payload = pickle.dumps((key, old))
        path = namespace / f"{key}.pkl"
        path.write_bytes(hashlib.sha256(payload).hexdigest().encode() + b"\n" + payload)
        when = audit.utc_timestamp("2026-09-27T18:30:00Z")
        os.utime(path, (when, when))

    def frozen_replay(_run, *, device, output_dir):
        assert not os.environ["DEEPMZYME_GRAPH_CACHE_DIR"] and device == "cpu"
        output_dir.mkdir()
        campaign_runtime.export_selected_validation_predictions(
            model=model, val_loader=loader, val_pockets=pockets,
            best_checkpoint={"normalization_stats": {"means": normalization.means, "stds": normalization.stds}})
        pytest.fail("Capture should interrupt before any replay export")

    monkeypatch.setattr(campaign_runtime, "replay_campaign_run", frozen_replay)
    monkeypatch.setattr(audit, "verify_source", lambda expected: expected)
    prep = tmp_path / "prepared"
    manifest = audit.prepare(SimpleNamespace(run_dir=run, cache_namespace=namespace,
                            cache_after="2026-09-27T18:14:00Z", cache_before="2026-09-27T18:54:00Z",
                            allow_ec_metadata_differences=True, output_dir=prep))
    assert manifest["status"] == "predictive_inputs_matched"
    assert len(manifest["examples"]) == 2 and len(manifest["batches"]) == 1
    assert set(manifest["batches"][0]["field_differences"]) == {"ec_group_id"}
    assert not (prep / "capture" / "val_predictions.csv").exists()
    assert not (prep / "capture" / "replay_receipt.json").exists()
    monkeypatch.setattr(campaign_runtime, "load_campaign_prediction_components", lambda *a, **k: (TinyModel(), None, None))
    evaluation = tmp_path / "evaluation"
    result = audit.evaluate(SimpleNamespace(prepared_dir=prep, output_dir=evaluation,
                                           condition="original", device="cpu", repeats=2))
    assert result["status"] == "diagnosis_complete" and result["completed_repeats"] == 2
    assert not result["certifies_fit"] and not result["training_performed"]
    assert all(row["native"]["changed_predictions"] == 0 for row in result["comparisons_to_saved"])
    assert all(audit.file_sha(path) == expected for path, expected in originals.items())
    # Recover a failed strict comparison where only the inactive EC target differs.
    failed_dir = tmp_path / "failed_prepare"
    failed_dir.mkdir()
    failed_examples = []
    for index, graph in enumerate(graphs):
        old = graph.clone()
        old.y_ec.fill_(0)
        key = str(index) * 64
        payload = pickle.dumps((key, old))
        path = namespace / f"{key}.pkl"
        path.write_bytes(hashlib.sha256(payload).hexdigest().encode() + b"\n" + payload)
        os.utime(path, (when, when))
        failed_examples.append({"source_uid": f"uid-{index}", "fresh_fields": audit.graph_fields(graph),
                                "candidate_field_mismatches": [{"cache_path": str(path), "differences":
                                    audit.field_differences(audit.graph_fields(graph), audit.graph_fields(old))}]})
    failed_manifest = failed_dir / "input_manifest.json"
    failed_manifest.write_text(json.dumps({**manifest, "status": "input_mismatch", "examples": failed_examples}))
    monkeypatch.setattr(campaign_runtime, "load_campaign_prediction_components",
                        lambda *a, **k: (TinyModel(), normalization, {"esm_dim": 6}))
    recovered_dir = tmp_path / "recovered"
    recovered = audit.recover(SimpleNamespace(failed_manifest=failed_manifest, output_dir=recovered_dir))
    assert recovered["schema"] == audit.RECOVERY_SCHEMA
    assert recovered["excluded_from_predictive_equality"] == ["y_ec"]
    assert recovered["disabled_ec_cpu_check"]["logits_metal_bitwise_equal"]
    assert recovered["disabled_ec_cpu_check"]["loss_bitwise_equal"]
    result = audit.evaluate(SimpleNamespace(prepared_dir=recovered_dir, output_dir=tmp_path / "recovered_eval",
                                           condition="original", device="cpu", repeats=2))
    assert result["schema"] == audit.RECOVERY_SCHEMA and result["status"] == "diagnosis_complete"
    assert all(audit.file_sha(path) == expected for path, expected in originals.items())
    (prep / "input_snapshot.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Input snapshot changed"):
        audit.evaluate(SimpleNamespace(prepared_dir=prep, output_dir=tmp_path / "tampered_evaluation",
                                       condition="original", device="cpu", repeats=2))


def test_ec_recovery_refuses_active_head_changed_metal_or_other_tensor(tmp_path):
    old = Data(x=torch.ones(1, 6), y_ec=torch.tensor([0]), y_metal=torch.tensor([1]))
    fresh = old.clone()
    fresh.y_ec.fill_(-1)
    key = "c" * 64
    path = tmp_path / f"{key}.pkl"
    payload = pickle.dumps((key, old))
    path.write_bytes(hashlib.sha256(payload).hexdigest().encode() + b"\n" + payload)
    example = {"fresh_fields": audit.graph_fields(fresh), "candidate_field_mismatches": [
        {"cache_path": str(path), "differences": audit.field_differences(audit.graph_fields(fresh), audit.graph_fields(old))}]}
    restored, _, _ = audit.restore_captured_graph(example)
    assert audit.graph_hash(restored) == audit.graph_hash(fresh)
    model = TinyModel()
    model.predict_ec = True
    with pytest.raises(ValueError, match="standalone metal"):
        audit.verify_disabled_ec_effect(model, fresh, old)
    example["fresh_fields"]["y_metal"] = audit.descriptor(torch.tensor([2]))
    with pytest.raises(ValueError, match="Cached tensors differ"):
        audit.restore_captured_graph(example)
