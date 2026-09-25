"""Campaign input/replay/RNG regressions using synthetic training-side examples."""
from __future__ import annotations

import csv
import json
import multiprocessing
import os
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from pmm_campaign_fixtures import build_train_dir, standard_entries, write_fake_embeddings
from benchmarking import pmm_ion_campaign as campaign
from benchmarking import pmm_ion_cohort as cohort
from benchmarking import pmm_ion_features as features
from training.campaign_runtime import CampaignContractError, replay_campaign_run
from training.source_cohort import CohortBindingError, sha256_file


@pytest.fixture(autouse=True)
def restore_scheme():
    from label_schemes import configure_active_metal_label_scheme

    yield
    configure_active_metal_label_scheme("split_all_metals")


def make_campaign(tmp_path, monkeypatch, n_pdb=3):
    train, source, source_sha = build_train_dir(tmp_path / "dataset", standard_entries(n_pdb))
    monkeypatch.setattr(cohort, "PMM_SOURCE_SHA256", source_sha)
    paths = campaign.CampaignPaths(tmp_path / "campaign")
    cohort.run_audit(train, source, paths.root, workers=1)
    return train, paths


def make_features(tmp_path, monkeypatch):
    train, paths = make_campaign(tmp_path, monkeypatch)
    monkeypatch.setattr(features, "ESM_DIM", 8)
    features.plan_embeddings(paths, train)
    esm = tmp_path / "embeddings"
    write_fake_embeddings(train, esm, dim=8)
    return train, paths, esm


@pytest.mark.parametrize("arguments", [
    ["--task", "ec", "--val-fraction", "0.2"],
    ["--task", "metal", "--val-fraction", "0.2"],
    ["--task", "metal", "--metal-example-unit", "ion", "--val-fraction", "0.2"],
    ["--task", "metal", "--metal-example-unit", "ion", "--source-cohort-csv", "frozen.csv",
     "--source-cohort-sha256", "0" * 64],
])
def test_export_configuration_rejected_before_any_loading(arguments):
    from training.config import parse_args

    with pytest.raises(SystemExit):
        parse_args(arguments + ["--export-validation-predictions"])


def test_programmatic_export_configuration_is_guarded():
    from dataclasses import replace
    from training.config import parse_args
    from training.run import validate_training_configuration

    base = parse_args(["--task", "metal", "--model-architecture", "only_gvp"])
    with pytest.raises(ValueError, match="requires --source-cohort-csv"):
        validate_training_configuration(replace(base, export_validation_predictions=True))
    with pytest.raises(ValueError, match="requires a validation split"):
        validate_training_configuration(replace(base, export_validation_predictions=True, metal_example_unit="ion",
                                               source_cohort_csv="frozen.csv", source_cohort_sha256="0" * 64))


@pytest.mark.parametrize("mutation", ["wrong_sequence", "wrong_order", "nonfinite", "missing_sidecar", "foreign_sidecar"])
def test_certification_rejects_unproven_embeddings(tmp_path, monkeypatch, mutation):
    from training.esm_feature_loading import embedding_metadata_from_payload, write_embedding_metadata_sidecar

    train, paths, esm = make_features(tmp_path, monkeypatch)
    expected = features._planned_inputs(paths, train)[0]
    path = esm / f"{Path(expected['structure_name']).stem}_chain_{expected['chain']}_esmc.pt"
    payload = torch.load(path, weights_only=True)
    if mutation == "missing_sidecar":
        path.with_name(path.name + ".json").unlink()
    elif mutation == "foreign_sidecar":
        sidecar = json.loads(path.with_name(path.name + ".json").read_text())
        sidecar["esm_model_name"] = "esmc_300m"
        path.with_name(path.name + ".json").write_text(json.dumps(sidecar))
    else:
        if mutation == "wrong_sequence":
            payload["metadata"]["source_sequence_sha256"] = "0" * 64
        elif mutation == "wrong_order":
            payload["residue_ids"] = list(reversed(payload["residue_ids"]))
        else:
            payload["embeddings"][0, 0] = float("nan")
        torch.save(payload, path)
        write_embedding_metadata_sidecar(path, embedding_metadata_from_payload(payload))
    with pytest.raises((ValueError, FileNotFoundError)):
        features.certify_inventory(paths, train, esm, load_workers=1)
    assert not paths.feature_inventory.exists()


def test_admission_detects_mutation_after_certification(tmp_path, monkeypatch):
    train, paths, esm = make_features(tmp_path, monkeypatch)
    inventory = features.certify_inventory(paths, train, esm, load_workers=1)
    receipt = features.verify_frozen_feature_inventory(paths, train)
    assert receipt["verified"] and inventory["schema_version"] == 2
    path = esm / inventory["esm"]["files"][0]["path"]
    payload = torch.load(path, weights_only=True)
    payload["embeddings"][0, 0] += 0.125
    torch.save(payload, path)
    with pytest.raises(ValueError, match="Frozen embedding content changed"):
        features.verify_frozen_feature_inventory(paths, train)


def test_admission_rehashes_without_parsing_or_loading_tensors_and_allows_relocation(tmp_path, monkeypatch):
    import embed_helpers.esmc as esmc

    train, paths, esm = make_features(tmp_path, monkeypatch)
    inventory = features.certify_inventory(paths, train, esm, load_workers=1)
    monkeypatch.setattr(esmc, "parse_structure", lambda *a, **k: pytest.fail("Admission reparsed a certified structure"))
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("Admission reloaded a certified tensor"))
    receipt = features.verify_frozen_feature_inventory(paths, train)
    assert receipt["verified"] and receipt["payloads_verified"] is False
    relocated_train = train.rename(train.with_name("relocated_train"))
    relocated_esm = esm.rename(esm.with_name("relocated_embeddings"))
    inventory["esm"]["embeddings_dir"] = str(relocated_esm)
    campaign.write_json(paths.feature_inventory, inventory)
    assert features.verify_frozen_feature_inventory(paths, relocated_train)["verified"]


@pytest.mark.parametrize("mutation", ["structure", "cohort", "plan", "plan_and_summary", "sidecar",
                                      "missing_payload", "missing_sidecar", "missing_record", "extra_record",
                                      "wrong_record_sequence"])
def test_hash_only_admission_rejects_frozen_input_drift(tmp_path, monkeypatch, mutation):
    train, paths, esm = make_features(tmp_path, monkeypatch)
    inventory = features.certify_inventory(paths, train, esm, load_workers=1)
    payload = esm / inventory["esm"]["files"][0]["path"]
    sidecar = payload.with_name(payload.name + ".json")
    if mutation == "structure":
        target = next((train / "structures").glob("*.pdb"))
        target.write_bytes(target.read_bytes() + b"\n")
    elif mutation == "cohort":
        paths.cohort.write_bytes(paths.cohort.read_bytes() + b"\n")
    elif mutation in {"plan", "plan_and_summary"}:
        plan = paths.root / "esm_generation_plan.csv"
        plan.write_bytes(plan.read_bytes() + b"\n")
        if mutation == "plan_and_summary":
            summary_path = paths.root / "esm_generation_plan.json"
            summary = campaign.read_json(summary_path)
            summary["plan_csv_sha256"] = sha256_file(plan)
            campaign.write_json(summary_path, summary)
    elif mutation == "sidecar":
        sidecar.write_bytes(sidecar.read_bytes() + b"\n")
    elif mutation.startswith("missing_") and mutation != "missing_record":
        (payload if mutation == "missing_payload" else sidecar).unlink()
    else:
        records = inventory["esm"]["files"]
        if mutation == "missing_record":
            records.pop()
        elif mutation == "extra_record":
            records.append(dict(records[0]))
        else:
            records[0]["sequence_sha256"] = "0" * 64
        inventory["esm"]["n_files"] = len(records)
        campaign.write_json(paths.feature_inventory, inventory)
    with pytest.raises((ValueError, FileNotFoundError, CohortBindingError)):
        features.verify_frozen_feature_inventory(paths, train)


def test_explicit_admission_semantic_reaudit_remains_available(tmp_path, monkeypatch):
    train, paths, esm = make_features(tmp_path, monkeypatch)
    inventory = features.certify_inventory(paths, train, esm, load_workers=1)
    validated = []
    validate = features.validate_embedding_file

    def recording_validate(path, expected):
        validated.append(path.name)
        return validate(path, expected)

    monkeypatch.setattr(features, "validate_embedding_file", recording_validate)
    receipt = features.verify_frozen_feature_inventory(paths, train, verify_payloads=True)
    assert receipt["verified"] and receipt["payloads_verified"] is True
    assert validated == [row["path"] for row in inventory["esm"]["files"]]


def test_generation_reuses_verified_sequences_across_restarts_without_model(tmp_path, monkeypatch):
    import embed_helpers.esmc as esmc

    train, paths, esm = make_features(tmp_path, monkeypatch)
    planned = features._planned_inputs(paths, train)
    missing = planned[0]
    target = esm / f"{Path(missing['structure_name']).stem}_chain_{missing['chain']}_esmc.pt"
    target.unlink()
    target.with_name(target.name + ".json").unlink()
    matching = next(row for row in planned[1:] if row["sequence_sha256"] == missing["sequence_sha256"])
    source = esm / f"{Path(matching['structure_name']).stem}_chain_{matching['chain']}_esmc.pt"
    expected = torch.load(source, weights_only=True)["embeddings"]
    monkeypatch.setattr(esmc, "load_esmc_model", lambda *a, **k: pytest.fail("Unneeded model/GPU initialization"))
    result = features.generate_embeddings(paths, train, esm, device="cpu")
    actual = features.validate_embedding_file(target, missing)
    assert result["unique_inferred"] == 0
    assert actual.dtype == expected.dtype and torch.equal(actual, expected)


def test_generation_preserves_emitted_fp32_with_bf16_parameters(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import embed_helpers.esmc as esmc

    train, paths, _ = make_features(tmp_path, monkeypatch)

    class FakeESMC:
        def parameters(self):
            return iter([torch.zeros(1, dtype=torch.bfloat16)])

        def encode(self, sequence):
            return sequence

        def logits(self, sequence, _config):
            return SimpleNamespace(embeddings=torch.full((1, len(sequence), 8), 1.003, dtype=torch.float32))

    monkeypatch.setattr(esmc, "load_esmc_model", lambda *a, **k: (FakeESMC(), "cuda"))
    monkeypatch.setattr(esmc, "_esmc_sdk", lambda: (object, lambda sequence: sequence, lambda **kwargs: kwargs))
    out_dir = tmp_path / "new_embeddings"
    result = features.generate_embeddings(paths, train, out_dir, device="cuda")
    payload = torch.load(next(out_dir.glob("*.pt")), weights_only=True)
    assert result["model_dtype"] == "torch.bfloat16"
    assert payload["embeddings"].dtype == torch.float32
    assert torch.equal(payload["embeddings"], torch.full_like(payload["embeddings"], 1.003))


def _worker_read(path):
    try:
        Path(path).read_text()
    except PermissionError:
        return "denied"
    return "unexpected_read"


def test_spawned_parse_worker_installs_held_out_guard(tmp_path, monkeypatch):
    from training.access_guard import FORBIDDEN_READ_ROOTS_ENV
    from training.parallel_loading import _init_worker

    forbidden = tmp_path / "held_out_fixture"
    forbidden.mkdir()
    (forbidden / "sentinel.txt").write_text("synthetic")
    monkeypatch.setenv(FORBIDDEN_READ_ROOTS_ENV, str(forbidden))
    with multiprocessing.get_context("spawn").Pool(1, initializer=_init_worker,
                                                    initargs=({}, "split_all_metals", None)) as pool:
        assert pool.apply(_worker_read, (str(forbidden / "sentinel.txt"),)) == "denied"


def run_tiny_campaign(tmp_path, monkeypatch):
    from training.config import parse_args
    from training.run import run_training

    train, paths = make_campaign(tmp_path, monkeypatch, n_pdb=30)
    campaign.freeze_folds(paths)
    esm = tmp_path / "esm"
    write_fake_embeddings(train, esm, dim=8)
    argv = ["--task", "metal", "--metal-example-unit", "ion", "--metal-label-scheme", "six_class",
            "--metal-eligibility-scheme", "six_class", "--structure-dir", str(train),
            "--source-cohort-csv", str(paths.cohort), "--source-cohort-sha256", sha256_file(paths.cohort),
            "--fold-membership-csv", str(paths.fold_membership),
            "--fold-membership-sha256", sha256_file(paths.fold_membership),
            "--n-folds", "5", "--fold-index", "1", "--train-val-split-by", "pdbid", "--split-seed", "42",
            "--split-stratify-by", "metal_site", "--epochs", "2", "--batch-size", "8", "--seed", "42",
            "--model-architecture", "only_esm", "--learning-rate", "3e-4", "--hidden-s", "16",
            "--hidden-v", "4", "--edge-hidden", "8", "--gvp-layers", "2", "--esm-fusion-dim", "8",
            "--esm-dim", "8", "--esm-embeddings-dir", str(esm), "--no-prepare-missing-esm-embeddings",
            "--external-features-root-dir", str(tmp_path / "empty_ext"), "--allow-missing-external-features",
            "--omit-node-features", ",".join(campaign.OMITTED_EXTERNAL_FEATURES), "--shell-role-source", "geometry",
            "--no-prepare-missing-ring-edges", "--metal-class-weight-mode", "manual",
            "--binding-residue-pooling", "first_shell_bias", "--selection-metric", "val_metal_balanced_acc",
            "--export-validation-predictions", "--campaign-run-identity", json.dumps({"fixture": True}),
            "--runs-dir", str(tmp_path / "runs"), "--load-workers", "1", "--device", "cpu"]
    runs = []
    for frequency in (1, 2):
        runs.append(run_training(parse_args(argv + ["--run-name", f"log{frequency}",
                                                     "--train-metrics-every-n-epochs", str(frequency)])))
    return runs


def test_sparse_metrics_preserve_updates_and_independent_replay(tmp_path, monkeypatch):
    first, second = run_tiny_campaign(tmp_path, monkeypatch)
    left = torch.load(first / "last_model_checkpoint.pt", weights_only=False)
    right = torch.load(second / "last_model_checkpoint.pt", weights_only=False)
    assert left["history"][0]["train_loss"] == right["history"][0]["train_loss"]
    assert left["history"][1]["train_loss"] == right["history"][1]["train_loss"]
    for name, tensor in left["model_state_dict"].items():
        assert torch.equal(tensor, right["model_state_dict"][name]), name
    profile = json.loads((second / "runtime_profile.json").read_text())
    assert set(profile["phase_seconds"]) == {"prepare", "train_epoch", "train_metric_evaluation", "validation",
                                            "checkpoint_save", "selected_export"}
    assert all(seconds > 0 for seconds in profile["phase_seconds"].values())
    assert profile["process_peak_rss_bytes"] > 0 and profile["cuda_peak_allocated_bytes"] == 0
    receipt = replay_campaign_run(second, output_dir=tmp_path / "replay")
    assert receipt["independent_replay"] and receipt["prediction_rows_verified"]
    assert receipt["normalization_refitted"] is False and receipt["training_performed"] is False
    rows = list(csv.DictReader((tmp_path / "replay" / "val_predictions.csv").open()))
    assert rows and all(row["source_uid"] and row["physical_ion_id"] and row["checkpoint_sha256"] for row in rows)
    payload = json.loads((second / "run_config.json").read_text())
    selected = next(row for row in payload["history"] if row["epoch"] == receipt["selected_epoch"])
    selected["val_metal_balanced_acc"] += 0.1
    (second / "run_config.json").write_text(json.dumps(payload))
    with pytest.raises(CampaignContractError, match="does not reconcile"):
        replay_campaign_run(second, output_dir=tmp_path / "bad_replay")
    failed = json.loads((tmp_path / "bad_replay" / "selected_checkpoint.json").read_text())
    assert failed["fit_status"] == "replay_failed"


def test_empty_resolved_validation_fails_before_graphs_or_training(tmp_path, monkeypatch):
    import training.run as training
    from training.splits import PocketSplit

    monkeypatch.setattr(training, "split_pockets_k_fold", lambda pockets, **kwargs:
                        PocketSplit(train_pockets=pockets, val_pockets=[]))
    monkeypatch.setattr(training, "build_graph_data_list", lambda *a, **k: pytest.fail("Graphs built before empty-fold gate"))
    monkeypatch.setattr(training, "train_and_select_checkpoint", lambda *a, **k: pytest.fail("Training before empty-fold gate"))
    with pytest.raises(ValueError, match="nonempty validation split"):
        run_tiny_campaign(tmp_path, monkeypatch)


def test_smoke_derives_exact_certified_inputs_for_all_nine_profiles(tmp_path, monkeypatch):
    train, paths = make_campaign(tmp_path, monkeypatch, n_pdb=30)
    features.plan_embeddings(paths, train)
    esm = tmp_path / "esm600m_dimension"
    write_fake_embeddings(train, esm, dim=1152)
    parent = features.certify_inventory(paths, train, esm, load_workers=1)
    parent_sha = sha256_file(paths.feature_inventory)
    smoke = campaign.prepare_smoke_campaign(paths, per_element_groups=5)
    campaign.campaign_manifest_guard(smoke)
    verified = features.verify_frozen_feature_inventory(smoke, train)
    inventory = campaign.read_json(smoke.feature_inventory)
    assert verified["verified"] and inventory["certified"]
    assert 0 < inventory["n_examples_loaded"] < parent["n_examples_loaded"]
    assert 0 < inventory["esm"]["n_files"] < parent["esm"]["n_files"]
    assert sha256_file(paths.feature_inventory) == parent_sha
    assert inventory["esm"]["embedding_dim"] == 1152
    assert inventory["esm"]["plan_csv_sha256"] == sha256_file(smoke.root / "esm_generation_plan.csv")
    fold = campaign.runnable_folds(smoke)[0]
    units = campaign.selected_units(None, None, None, [fold])
    assert len(units) == 9
    for config, fold, seed in units:
        argv, _, identity = campaign.build_train_command(
            smoke, python_bin=sys.executable, train_dir=train, config=config, fold=fold, seed=seed,
            device="cpu", runs_dir=smoke.runs, epochs=1, load_workers=1, smoke=True,
        )
        assert argv[argv.index("--esm-dim") + 1] == "1152"
        assert identity["epochs"] == 1 and identity["smoke"] is True
    smoke_sha = sha256_file(smoke.feature_inventory)
    campaign.prepare_smoke_campaign(paths, per_element_groups=5)
    assert sha256_file(smoke.feature_inventory) == smoke_sha
    assert features.verify_frozen_feature_inventory(smoke, train)["verified"]


def test_terminal_refit_to_authorized_synthetic_reference(tmp_path, monkeypatch):
    """Real neural terminal refit/inference; only the completed PMM receipt is a stub."""
    from benchmarking import pmm_final_report as final
    from training.config import parse_args
    from training.run import run_training

    train, paths = make_campaign(tmp_path, monkeypatch, n_pdb=30)
    campaign.freeze_folds(paths)
    features.plan_embeddings(paths, train)
    esm = tmp_path / "full_training_embeddings"
    write_fake_embeddings(train, esm, dim=1152)
    features.certify_inventory(paths, train, esm, load_workers=1)
    # Keep the real 50-epoch protocol but make the synthetic network inexpensive.
    for key, value in {"hidden_s": 8, "hidden_v": 2, "edge_hidden": 4, "gvp_layers": 1,
                       "esm_fusion_dim": 4}.items():
        monkeypatch.setitem(campaign.PROFILE, key, value)
    monkeypatch.setattr(campaign, "source_tree_sha256", lambda: "synthetic_frozen_tree")
    argv, _, identity = final.build_refit_command(paths, train, final.CONTROL,
                                                python_bin=sys.executable, device="cpu", load_workers=1)
    # Epoch snapshots add no coverage here; the terminal checkpoint is retained.
    argv.remove("--save-epoch-checkpoints")
    run_dir = run_training(parse_args(argv[3:]))
    campaign.write_json(paths.root / "validation_decision.json", {"synthetic": True, "selected": final.CONTROL})
    decision = {"identity": identity,
                "validation_decision_sha256": sha256_file(paths.root / "validation_decision.json")}
    campaign.write_json(paths.root / "stage6b_decision.json", decision)
    receipt = final.finish_refit_receipt(paths, run_dir, identity)
    assert receipt["terminal_checkpoint"] and receipt["completed_epochs"] == 50
    assert final.verify_final_refit(paths) == receipt
    terminal = torch.load(run_dir / "last_model_checkpoint.pt", weights_only=False)
    assert len(terminal["history"]) == 50
    assert terminal["dataset_summary"]["retained_split_identity"]["validation"]["n_examples"] == 0
    assert terminal["dataset_summary"]["retained_split_identity"]["train"]["n_examples"] == 60

    fake_train, reference_source, reference_sha = build_train_dir(tmp_path / "synthetic_reference", standard_entries(3))
    reference_dir = fake_train.rename(fake_train.with_name("test"))
    site_path = reference_dir / "site_manifest.csv"
    with site_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    for row in rows:
        row["source_side"] = "test"
    with site_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(final, "REFERENCE_SOURCE_SHA256", reference_sha)
    monkeypatch.setattr(final, "verify_pmm_refit", lambda _paths: {"model_sha256": "synthetic_completed_pmm"})
    reference = campaign.CampaignPaths(paths.root / "reference" / "inputs_profile")
    route_path = paths.root / "reference" / "reference_route.json"
    route = {"route": final.ROUTE, "selection_frozen": True, "campaign_dir": str(paths.root),
             "reference_dir": str(reference_dir), "reference_source_sha256": reference_sha,
             "campaign_cohort_sha256": sha256_file(paths.cohort),
             "stage6b_decision_sha256": sha256_file(paths.root / "stage6b_decision.json"),
             "deepmzyme_refit_checkpoint_sha256": receipt["checkpoint_sha256"],
             "pmm_refit_model_sha256": "synthetic_completed_pmm"}
    campaign.write_json(route_path, route)
    cohort.run_reference_audit(reference_dir, reference_source, reference.root, route_path=route_path, workers=1)
    route["reference_cohort_sha256"] = sha256_file(reference.cohort)
    campaign.write_json(route_path, route)
    features.plan_embeddings(reference, reference_dir, reference_route_json=route_path)
    features.certify_inventory(reference, reference_dir, None, load_workers=1, reference_route_json=route_path)
    result = final.predict_reference(paths, reference, reference_dir, route_path, device="cpu")
    assert result["n_rows"] == 6 and result["normalization_refitted"] is False
    assert result["verified_inputs"]["certified_scope"] == "structure_only"
    assert result["configuration"] == final.CONTROL
    monkeypatch.setattr(campaign, "source_tree_sha256", lambda: "changed_after_freeze")
    with pytest.raises(final.FinalReportError, match="Scientific source changed"):
        final.require_reference_authorization(reference, reference_dir, route_path)
