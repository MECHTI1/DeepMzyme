"""CPU-only host protocol tests with fabricated, explicit development artifacts."""

from __future__ import annotations

import copy
import csv
import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]

import metal_completion as host
import metal_completion_campaign as mc
import pmm_v3_campaign as v3
from test_metal_completion_assessment import grid


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return path


@pytest.fixture
def new_unit(tmp_path, monkeypatch):
    unit = mc.discovery_units()[0]
    root = tmp_path / "successor"
    paths = v3.V3Paths(root)
    run = paths.lane(0) / "runs" / unit.name
    run.mkdir(parents=True)
    checkpoint = run / "terminal_model_checkpoint.pt"
    checkpoint.write_bytes(b"synthetic terminal bytes; never loaded as a model")
    checkpoint_sha = host.sha256(checkpoint)
    identity = {"epochs": 50, "resolved_config_sha256": "b" * 64, "model_seed": unit.seed,
                "recipe_definition": mc.effective_loss(unit.recipe)}
    rows, membership = [], {}
    for index, element in enumerate(("MN", "CU", "ZN", "FE", "CO", "NI")):
        label = min(index, 3)
        uid = f"synthetic-{element}"
        membership[uid] = {"fold": "0", "physical_ion_id": uid, "pdbid": f"pdb{index}", "native_element": element}
        row = {"source_uid": uid, "fold": "0", "physical_ion_id": uid, "group_id": f"pdb{index}",
               "native_element": element, "metal_label_scheme": v3.TARGET_SCHEMES["four_class"],
               "model_seed": "42", "checkpoint_sha256": checkpoint_sha, "selected_epoch": "50",
               "y_common4": str(label), "y_native": str(label), "pred_common4": str(label), "pred_native": str(label)}
        for prefix in ("p_common4", "p_native"):
            row.update({f"{prefix}_{name.replace(' ', '_')}": str(float(j == label))
                        for j, name in enumerate(host.assessment.COMMON4)})
        rows.append(row)
    prediction = run / "val_predictions.csv"
    with prediction.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    receipt = {"fit_status": "completed", "reconciliation_status": "match", "campaign_run_identity": identity,
               "selected_checkpoint": checkpoint.name, "selected_checkpoint_sha256": checkpoint_sha,
               "selected_epoch": 50, "validation_predictions": {"path": prediction.name, "sha256": host.sha256(prediction)},
               "metrics": {"val_metal_collapsed4_balanced_acc": 1.0, "val_metal_balanced_acc": 1.0}}
    put(run / "selected_checkpoint.json", receipt)
    put(run / "run_metadata.json", {"fit_status": "completed", "campaign_run_identity": identity, "test_report": None})
    put(run / "run_config.json", {"history": [{}] * 50,
                                  "config": {"seed": unit.seed, "metal_class_weight_mode": "manual",
                                             **mc.recipe_config(unit.recipe)}})
    (run / "epoch_metrics.csv").write_text("epoch,val_metal_collapsed4_balanced_acc\n50,1\n")
    replay_root = run / "independent_validation_replay"
    replay_root.mkdir()
    replay_prediction = replay_root / prediction.name
    replay_prediction.write_bytes(prediction.read_bytes())
    put(replay_root / "replay_receipt.json", {**receipt, "independent_replay": True,
                                             "prediction_rows_verified": True,
                                             "replay_policy": v3.REPLAY_POLICY,
                                             "max_probability_abs_difference": 0.0,
                                             "validation_predictions": {"path": prediction.name,
                                                                        "sha256": host.sha256(replay_prediction)}})
    status = {"status": "completed", "run_name": unit.name, "unit": unit.name, "lane": 0,
              "session_id": "synthetic-session", "identity": identity}
    status_path = put(paths.lane(0) / f"run_status_{unit.name}.json", status)
    put(paths.claims / f"{unit.name}.json", status)
    argv = ["python", "-u", "train.py", "--campaign-run-identity", json.dumps(identity)]
    put(paths.lane(0) / "commands" / f"{unit.name}.json", {"identity": identity, "run_name": unit.name, "argv": argv})
    rc = root / "launch" / f"{unit.name}.rc"
    rc.parent.mkdir()
    rc.write_text("0\n")
    manifest = {"successor_root": str(root), "parent_campaign_dir": str(tmp_path / "parent"),
                "code_sha256": {}, "train_dir": str(tmp_path / "train"), "esm_dir": str(tmp_path / "esm")}
    monkeypatch.setattr(mc, "build_command", lambda *args, **kwargs: ([], {}, identity))
    monkeypatch.setattr(mc, "resolved_config_sha256", lambda argv: identity["resolved_config_sha256"])
    monkeypatch.setattr(v3, "resolve_amp", lambda *args, **kwargs: False)
    return unit, {"run_dir": str(run), "status_path": str(status_path), "worker_rc_path": str(rc)}, manifest, membership


def test_new_unit_verifies_actual_terminal_and_prediction_replay_bytes(new_unit):
    unit, source, manifest, membership = new_unit
    result = host._new_unit(unit, source, manifest, "a" * 64, membership)
    assert result["complete"] and not result["reused"]
    assert result["metrics"]["common4"]["balanced_accuracy"] == 1.0
    assert result["metrics"]["common4"]["support"]["Class VIII"] == 3
    assert len(result["evidence"]) == 12
    assert source["worker_rc_path"] in result["evidence"]


@pytest.mark.parametrize("mutation,message", [
    ("rc", "exit code"), ("rc_name", "exact worker"), ("claim", "claim differs"),
    ("checkpoint", "Checkpoint changed"), ("prediction", "Predictions changed"),
    ("replay", "replay predictions"), ("identity", "identity differs"), ("test_report", "held-out report"),
    ("relative_rc", "absolute"), ("config_seed", "Trainer-recorded"), ("config_loss", "Trainer-recorded"),
    ("config_weights", "Trainer-recorded"),
])
def test_new_unit_rejects_incomplete_or_changed_artifacts(new_unit, mutation, message, monkeypatch):
    unit, source, manifest, membership = new_unit
    run = Path(source["run_dir"])
    if mutation == "rc":
        Path(source["worker_rc_path"]).write_text("1")
    elif mutation == "relative_rc":
        monkeypatch.chdir(Path(manifest["successor_root"]))
        source["worker_rc_path"] = str(Path(source["worker_rc_path"]).relative_to(manifest["successor_root"]))
    elif mutation.startswith("config_"):
        saved = host.read_json(run / "run_config.json")
        saved["config"].update({"config_seed": {"seed": 43}, "config_loss": {"metal_label_smoothing": 0.1},
                                "config_weights": {"metal_class_weight_mode": "inverse_frequency"}}[mutation])
        put(run / "run_config.json", saved)
    elif mutation == "rc_name":
        wrong = Path(source["worker_rc_path"]).with_name("another-unit.rc")
        wrong.write_text("0")
        source["worker_rc_path"] = str(wrong)
    elif mutation == "claim":
        claim = v3.V3Paths(Path(manifest["successor_root"])).claims / f"{unit.name}.json"
        put(claim, {**host.read_json(claim), "session_id": "another-session"})
    elif mutation in ("checkpoint", "prediction", "replay"):
        path = {"checkpoint": run / "terminal_model_checkpoint.pt", "prediction": run / "val_predictions.csv",
                "replay": run / "independent_validation_replay" / "val_predictions.csv"}[mutation]
        path.write_bytes(path.read_bytes() + b"changed")
    elif mutation == "identity":
        put(Path(source["status_path"]), {**host.read_json(Path(source["status_path"])), "identity": {}})
    else:
        put(run / "run_metadata.json", {**host.read_json(run / "run_metadata.json"), "test_report": "forbidden"})
    with pytest.raises(ValueError, match=message):
        host._new_unit(unit, source, manifest, "a" * 64, membership)


def test_prediction_path_traversal_refused_before_legacy_verifier_or_outside_file_open(new_unit, monkeypatch):
    unit, source, manifest, membership = new_unit
    run = Path(source["run_dir"])
    receipt = host.read_json(run / "selected_checkpoint.json")
    receipt["validation_predictions"]["path"] = "../../heldout.csv"
    put(run / "selected_checkpoint.json", receipt)
    monkeypatch.setattr(v3, "verify_completed_unit", lambda *args: pytest.fail("Legacy verifier reached"))
    with pytest.raises(ValueError, match="Unsafe validation prediction"):
        host._new_unit(unit, source, manifest, "a" * 64, membership)


def test_historical_control_requires_actual_matching_bytes_and_recomputed_metrics(new_unit, monkeypatch):
    unit, source, manifest, membership = new_unit
    run = Path(source["run_dir"])
    status = host.read_json(Path(source["status_path"]))
    old_unit = v3.Unit(mc.FAMILY, unit.target, "baseline", unit.fold, unit.seed)
    metrics = {"n": 6, "example": "recomputed"}
    proposed = {**{key: getattr(unit, key) for key in ("name", "phase", "recipe", "target", "fold", "seed")},
                "complete": True, "parent_unit_name": old_unit.name, "parent_identity": status["identity"],
                "run_dir": str(run), "status_path": source["status_path"], "metrics": metrics,
                "evidence": host._run_evidence(run, Path(source["status_path"]))}
    calls = []
    def verified(*args, **kwargs):
        calls.append(args[1])
        return {"status": "completed", "metrics": metrics}
    monkeypatch.setattr(host.old_assessment, "load_unit", verified)
    result = host._reused_unit(unit, proposed, object(), {}, membership)
    assert result["reused"] and calls == [old_unit]
    forged = copy.deepcopy(proposed)
    forged["metrics"]["n"] = 7
    with pytest.raises(ValueError, match="metrics do not reproduce"):
        host._reused_unit(unit, forged, object(), {}, membership)
    changed = copy.deepcopy(proposed)
    changed["evidence"][str(run / "terminal_model_checkpoint.pt")] = "0" * 64
    with pytest.raises(ValueError, match="artifact bytes"):
        host._reused_unit(unit, changed, object(), {}, membership)
    # The hashed directory must be the lane's runs/<run_name>, which the frozen v3 verifier recomputes.
    moved = run.parent.parent / "bar" / run.name
    shutil.copytree(run, moved)
    elsewhere = copy.deepcopy(proposed)
    elsewhere["run_dir"] = str(moved)
    elsewhere["evidence"] = host._run_evidence(moved, Path(source["status_path"]))
    with pytest.raises(ValueError, match="directory mismatch"):
        host._reused_unit(unit, elsewhere, object(), {}, membership)


def test_confirmation_prepare_binds_the_discovery_era_specification(tmp_path, monkeypatch):
    parent = v3.V3Paths(tmp_path / "parent")
    parent.root.mkdir()
    put(parent.manifest, {"parent": True})
    spec = put(tmp_path / "spec.json", {"campaign_id": mc.CAMPAIGN_ID, "assessor_rules": host.ASSESSOR_RULES})
    monkeypatch.setattr(v3, "verify_campaign", lambda *args, **kwargs: {
        "frozen_source_tree_sha256": "d" * 64, "cohort": {"sha256": "e" * 64},
        "fold_set": {"fold_membership_sha256": "f" * 64}})
    monkeypatch.setattr(v3, "require_frozen_spec", lambda *args: {})
    monkeypatch.setattr(v3, "load_extension", lambda *args: None)
    monkeypatch.setattr(mc, "verify_training_inputs", lambda *args: {"verified": "synthetic"})
    monkeypatch.setattr(host, "code_hashes", lambda: {"host": "a" * 64})
    discovery_manifest = put(tmp_path / "discovery" / "manifest_discovery.json",
                             {"assessment_spec_sha256": host.sha256(spec)})
    collection = put(tmp_path / "discovery" / "collection.json",
                     {"manifest_path": str(discovery_manifest), "manifest_sha256": host.sha256(discovery_manifest)})
    receipt = {"result": {"phase": "discovery", "shared_recipe": "P"},
               "parent_manifest_sha256": host.sha256(parent.manifest),
               "collection_path": str(collection), "collection_sha256": host.sha256(collection)}
    monkeypatch.setattr(host, "verify_assessment_receipt", lambda *args, **kwargs: receipt)
    args = dict(parent_dir=parent.root, train_dir=tmp_path / "train", esm_dir=tmp_path / "esm",
                assessment_spec_path=spec, phase="confirmation",
                selection_receipt_path=tmp_path / "selection.json", selection_receipt_sha256="a" * 64)
    result = host.prepare(successor_root=tmp_path / "successor", **args)
    assert result["manifest"]["shared_recipe"] == "P"
    assert result["manifest"]["discovery_manifest_sha256"] == host.sha256(discovery_manifest)
    assert len(result["manifest"]["authorized_grid_units"]) == 40
    put(discovery_manifest, {"assessment_spec_sha256": "0" * 64})
    collection = put(collection, {"manifest_path": str(discovery_manifest),
                                  "manifest_sha256": host.sha256(discovery_manifest)})
    receipt["collection_sha256"] = host.sha256(collection)
    with pytest.raises(ValueError, match="discovery-era"):
        host.prepare(successor_root=tmp_path / "successor2", **args)
    assert not (tmp_path / "successor2").exists()


def test_assessor_binding_follows_the_parent_extension(monkeypatch):
    actual = host.sha256(host.ROOT / "pmm_v3_assessment.py")
    monkeypatch.setattr(v3, "load_extension", lambda parent: None)
    host._check_assessor_binding(object())
    monkeypatch.setattr(v3, "load_extension", lambda parent: {"assessor_sha256": actual})
    host._check_assessor_binding(object())
    monkeypatch.setattr(v3, "load_extension", lambda parent: {"assessor_sha256": "0" * 64})
    with pytest.raises(ValueError, match="pmm_v3_assessment.py differs"):
        host._check_assessor_binding(object())
    assert "pmm_v3_assessment.py" in host.CODE_FILES and host.code_hashes()["pmm_v3_assessment.py"] == actual


@pytest.fixture
def collection_environment(tmp_path, monkeypatch):
    records = grid()
    successor = tmp_path / "successor"
    successor.mkdir()
    manifest = {"phase": "discovery", "successor_root": str(successor), "parent_campaign_dir": str(tmp_path / "parent"),
                "parent_manifest_sha256": "b" * 64, "authorized_grid_units": list(records)}
    manifest_path = put(successor / "manifest_discovery.json", manifest)
    monkeypatch.setattr(host, "_manifest", lambda *args: manifest)
    monkeypatch.setattr(host, "code_hashes", lambda: {"host": "c" * 64})
    monkeypatch.setattr(host.old_assessment, "read_contract", lambda parent: ({}, {}))
    monkeypatch.setattr(host, "_new_unit", lambda unit, *args: records[unit.name])
    return manifest_path, manifest, records


def test_collection_requires_exact_sources_and_binds_every_record(collection_environment):
    path, manifest, records = collection_environment
    args = dict(manifest_path=path, manifest_sha256=host.sha256(path), esm_dir=Path("/synthetic/esm"))
    result = host.collect(**args, runs={name: {} for name in records})
    assert result["records_sha256"] == mc.stable_hash(result["records"])
    assert result["held_out_accessed"] is False and len(result["records"]) == 45
    with pytest.raises(ValueError, match="Incomplete grid"):
        host.collect(**args, runs={})
    with pytest.raises(ValueError, match="outside this phase"):
        host.collect(**args, runs={**{name: {} for name in records}, "unknown": {}})


def test_assessment_receipt_is_immutable_and_reproducible_and_detects_result_tampering(collection_environment):
    path, manifest, records = collection_environment
    collected = host.collect(manifest_path=path, manifest_sha256=host.sha256(path), esm_dir=Path("/synthetic/esm"),
                             runs={name: {} for name in records})
    collection_path = put(Path(manifest["successor_root"]) / "collection.json", collected)
    output = Path(manifest["successor_root"]) / "discovery_selection.json"
    args = dict(collection_path=collection_path, collection_sha256=host.sha256(collection_path),
                esm_dir=Path("/synthetic/esm"), output_path=output)
    receipt = host.assess(**args)
    assert receipt["result"]["shared_recipe"] == "B" and receipt["execution_authorized"] is False
    assert host.verify_assessment_receipt(output, host.sha256(output), esm_dir=args["esm_dir"]) == receipt
    with pytest.raises(ValueError, match="already exists"):
        host.assess(**args)
    receipt["result"]["shared_recipe"] = "P"
    put(output, receipt)
    with pytest.raises(ValueError, match="does not reproduce"):
        host.verify_assessment_receipt(output, host.sha256(output), esm_dir=args["esm_dir"])


def test_assessment_recollects_instead_of_trusting_changed_metrics(collection_environment):
    path, manifest, records = collection_environment
    collected = host.collect(manifest_path=path, manifest_sha256=host.sha256(path), esm_dir=Path("/synthetic/esm"),
                             runs={name: {} for name in records})
    saved = copy.deepcopy(collected)
    saved["records"][next(iter(records))]["metrics"]["n"] += 1
    saved["records_sha256"] = mc.stable_hash(saved["records"])
    collection_path = put(Path(manifest["successor_root"]) / "forged_collection.json", saved)
    with pytest.raises(ValueError, match="does not reproduce"):
        host.assess(collection_path=collection_path, collection_sha256=host.sha256(collection_path),
                    esm_dir=Path("/synthetic/esm"), output_path=Path(manifest["successor_root"]) / "no_receipt.json")
    assert not (Path(manifest["successor_root"]) / "no_receipt.json").exists()


def test_prepare_freezes_exact_discovery_grid_and_cannot_overwrite(tmp_path, monkeypatch):
    parent = v3.V3Paths(tmp_path / "parent")
    parent.root.mkdir()
    put(parent.manifest, {"parent": True})
    spec = put(tmp_path / "spec.json", {"campaign_id": mc.CAMPAIGN_ID, "assessor_rules": host.ASSESSOR_RULES})
    monkeypatch.setattr(v3, "verify_campaign", lambda *args, **kwargs: {
        "frozen_source_tree_sha256": "d" * 64, "cohort": {"sha256": "e" * 64},
        "fold_set": {"fold_membership_sha256": "f" * 64}})
    monkeypatch.setattr(v3, "require_frozen_spec", lambda *args: {})
    monkeypatch.setattr(mc, "verify_training_inputs", lambda *args: {"verified": "synthetic"})
    monkeypatch.setattr(host, "code_hashes", lambda: {"host": "a" * 64})
    args = dict(parent_dir=parent.root, successor_root=tmp_path / "successor", train_dir=tmp_path / "train",
                esm_dir=tmp_path / "esm", assessment_spec_path=spec)
    result = host.prepare(**args)
    assert result["manifest"]["authorized_grid_units"] == [unit.name for unit in mc.discovery_units()]
    assert result["manifest"]["execution_authorized"] is False
    assert result["manifest"]["assessment_spec_sha256"] == host.sha256(spec)
    assert result["manifest"]["training_input_verification"] == {"verified": "synthetic"}
    with pytest.raises(ValueError, match="already exists"):
        host.prepare(**args)


def test_host_rejects_changed_spec_and_output_escape_before_writing(tmp_path):
    spec = put(tmp_path / "spec.json", {"campaign_id": mc.CAMPAIGN_ID,
                                       "assessor_rules": {**host.ASSESSOR_RULES, "minimum_gain": 0.001}})
    with pytest.raises(ValueError, match="implemented rules"):
        host._verify_spec(spec)
    with pytest.raises(ValueError, match="inside successor"):
        host.write_once(tmp_path / "outside.json", {}, root=tmp_path / "successor")
    assert not (tmp_path / "outside.json").exists()
