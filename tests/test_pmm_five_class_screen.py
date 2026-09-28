"""Synthetic CPU contracts for the bounded five-class adapter; no held-out data."""
import importlib.util
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

ORIGINAL_POPEN = subprocess.Popen
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]
spec = importlib.util.spec_from_file_location("five_screen", ROOT / "run_pmm_five_class_screen.py")
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)
from benchmarking import pmm_ion_campaign as campaign
from benchmarking import pmm_ion_features
from benchmarking.pmm_execution import ExecutionBlocked
from test_pmm_ion_campaign import _campaign_with_folds, _full_inventory
from training.config import config_to_payload, parse_args


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    train, paths, weights = _campaign_with_folds(tmp_path, monkeypatch)
    _full_inventory(paths, tmp_path)
    protocol = tmp_path / "screen.json"
    protocol.write_text(json.dumps(screen.EXPECTED))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("Unexpected child process"))
    return train, paths, protocol, weights


def cli(prepared):
    train, paths, protocol, _ = prepared
    return ["--protocol-file", str(protocol), "--campaign-dir", str(paths.root), "--train-dir", str(train)]


def runtime_cli(prepared):
    now = time.time()
    return cli(prepared) + ["--action", "run", "--family", "only_gvp", "--session-id", "fixture",
                            "--execution-deadline", str(now + 6000), "--allocation-started", str(now),
                            "--execution-max-seconds", "6000", "--estimated-fit-seconds", "10",
                            "--durable-root", str(prepared[1].root.parent / "independent"),
                            "--persistence-mode", "mounted"]


def test_plan_builds_three_actual_five_class_commands_without_writes(prepared, capsys):
    train, paths, _, weights = prepared
    before = {str(p): p.read_bytes() for p in paths.root.rglob("*") if p.is_file()}
    old_schemes = dict(campaign.TARGET_SCHEMES)
    assert screen.main(cli(prepared)) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["candidate_units"] == 3 and len(preview["units"]) == 3
    assert not preview["child_invoked"] and not preview["files_written"]
    assert not preview["runtime_admission_checked"] and not preview["promotion_ready"]
    for unit in preview["units"]:
        argv, identity = unit["preview_training_argv"], unit["identity"]
        config = config_to_payload(parse_args(argv[3:]))
        assert config["metal_label_scheme"] == "five_class"
        assert config["metal_eligibility_scheme"] == "six_class"
        assert config["metal_example_unit"] == "ion" and config["task"] == "metal"
        assert config["binding_residue_pooling"] == "none"
        assert config["selection_metric"] == "val_metal_balanced_acc"
        assert (config["epochs"], config["batch_size"], config["n_folds"], config["fold_index"], config["seed"]) == (50, 16, 5, 0, 42)
        assert not config["run_test_eval"] and config["test_structure_dir"] is None
        assert config["metal_class_weight_mode"] == "manual"
        expected = weights["folds"]["0"]["four_class_multipliers"]
        for key in ("mn", "cu", "zn", "class_viii"):
            assert config[key + "_loss_multiplier"] == expected[key]
        assert config["fe_loss_multiplier"] == expected["class_viii"]
        assert identity["resolved_config_sha256"] == campaign.stable_hash({
            k: v for k, v in config.items() if k not in campaign.NON_IDENTITY_CONFIG_KEYS})
        assert json.loads(config["campaign_run_identity"]) == identity
        assert identity["five_class_screen"]["adapter_sha256"] == screen.file_sha(screen.__file__)
        assert identity["target_scheme"] == "five_class" and identity["source_tree_sha256"] == screen.EXPECTED["source_tree_sha256"]
    assert campaign.TARGET_SCHEMES == old_schemes
    assert before == {str(p): p.read_bytes() for p in paths.root.rglob("*") if p.is_file()}


def test_existing_four_six_identities_unchanged_by_five_adapter(prepared):
    train, paths, protocol, _ = prepared
    options = dict(python_bin=sys.executable, train_dir=train, fold=0, seed=42,
                   device="cpu", runs_dir=paths.runs)
    configs = [campaign.GridConfig(family, target, "none") for family in screen.EXPECTED["families"]
               for target in ("four_class", "six_class")]
    before = [campaign.build_train_command(paths, config=c, **options) for c in configs]
    args = screen.parse_args(cli(prepared))
    command, _, identity = screen.build_command(paths, args, "only_esm", screen.file_sha(protocol))
    assert [campaign.build_train_command(paths, config=c, **options) for c in configs] == before
    args.action = "run"
    assert screen.build_command(paths, args, "only_esm", screen.file_sha(protocol))[2] == identity


def predictions():
    rows, membership = [], {}
    for element, native in screen.NATIVE_TARGETS.items():
        uid = "ion_" + element
        member = {"source_uid": uid, "physical_ion_id": uid, "group_id": "pdb_" + element,
                  "native_element": element, "fold": "0"}
        membership[uid] = member
        row = {**member, "model_seed": "42", "checkpoint_sha256": "a" * 64,
               "metal_label_scheme": "five_class", "binding_residue_pooling": "none",
               "selected_epoch": "7", "y_native": str(native), "y_common4": str(min(native, 3)),
               "pred_native": "0", "pred_common4": "3"}
        # Native argmax is Mn, but probability-first collapse correctly selects VIII.
        row.update(zip(("p_native_" + label.replace(" ", "_") for label in screen.NATIVE_LABELS),
                       ("0.30", "0.10", "0.10", "0.25", "0.25")))
        row.update(zip(("p_common4_" + label.replace(" ", "_") for label in ("Mn", "Cu", "Zn", "Class VIII")),
                       ("0.30", "0.10", "0.10", "0.50")))
        rows.append(row)
    return rows, membership


def validate(rows, membership):
    screen.validate_five_rows(rows, membership, checkpoint_sha256="a" * 64, selected_epoch=7)


def test_native_five_and_probability_first_collapse_with_co_ni_merge():
    validate(*predictions())


@pytest.mark.parametrize("change", [
    {"y_native": "3"}, {"p_native_Co": "0"}, {"p_native_Class_VIII": "nan"},
    {"p_native_Fe": "0.5"}, {"p_common4_Class_VIII": "0.25"}, {"pred_common4": "0"},
    {"pred_native": "4"}, {"model_seed": "43"}, {"checkpoint_sha256": "b" * 64},
    {"metal_label_scheme": "four_class"}, {"binding_residue_pooling": "first_shell_bias"},
    {"selected_epoch": "8"}, {"physical_ion_id": "other"}, {"fold": "1"},
])
def test_malformed_or_mislabeled_five_predictions_refused(change):
    rows, membership = predictions()
    rows[-1].update(change)
    with pytest.raises(ValueError):
        validate(rows, membership)


def test_duplicate_missing_uids_and_four_column_native_vectors_refused():
    rows, membership = predictions()
    with pytest.raises(ValueError):
        validate(rows + [rows[0]], membership)
    with pytest.raises(ValueError):
        validate(rows[:-1], membership)
    del rows[0]["p_native_Fe"]
    with pytest.raises(ValueError, match="exactly five"):
        validate(rows, membership)


@pytest.mark.parametrize("arguments", [["--fold", "1"], ["--target", "six_class"], ["--readout", "first_shell_bias"],
    ["--epochs", "1"], ["--action", "assess"], ["--action", "refit"], ["--action", "test"],
    ["--family", "gvp_hybrid"], ["--device", "cuda"], ["--camp", "abbreviation"]])
def test_forbidden_overrides_and_actions_rejected(arguments):
    with pytest.raises(SystemExit):
        screen.parse_args(["--campaign-dir", "/fixture/campaign", "--train-dir", "/fixture/train"] + arguments)


@pytest.mark.parametrize("artifact", ["directory", "log", "command", "status", "dangling_symlink", "archive"])
def test_existing_artifacts_never_retrained_or_overwritten(prepared, artifact):
    paths = prepared[1]
    choices = screen.artifact_paths(paths, "only_gvp")
    path = choices[{"directory": 0, "log": 1, "command": 2, "status": 3, "dangling_symlink": 0}.get(artifact, 0)]
    if artifact == "archive":
        path = paths.runs / "_incomplete_attempts" / (screen.unit_name("only_gvp") + "__earlier")
    path.parent.mkdir(parents=True, exist_ok=True)
    if artifact == "directory":
        path.mkdir()
    elif artifact == "dangling_symlink":
        path.symlink_to(paths.root / "missing")
    else:
        path.write_text("preserve")
    with pytest.raises(ValueError, match="retry"):
        screen.main(runtime_cli(prepared))


def test_source_drift_and_changed_protocol_refused(prepared, monkeypatch):
    monkeypatch.setattr(screen, "source_tree_sha256", lambda: "drift")
    with pytest.raises(ValueError, match="source changed"):
        screen.main(cli(prepared))
    protocol = dict(screen.EXPECTED, independent_replay_atol="0.00001")
    prepared[2].write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="independent_replay_atol"):
        screen.main(cli(prepared))


def test_real_admission_blocks_unbudgeted_child(prepared, monkeypatch):
    monkeypatch.setattr(campaign, "campaign_manifest_guard", lambda *a, **k: None)
    monkeypatch.setattr(pmm_ion_features, "verify_frozen_feature_inventory", lambda *a: None)
    monkeypatch.setattr(campaign, "execute_unit", lambda *a, **k: pytest.fail("Admission must precede child"))
    args = runtime_cli(prepared)
    args[args.index("--estimated-fit-seconds") + 1] = "9000"
    with pytest.raises(ExecutionBlocked, match="remain"):
        screen.main(args)


def test_tiny_cpu_five_fit_replays_and_original_strict_threshold_refuses_drift(prepared, monkeypatch):
    """Actual synthetic one-epoch fit/replay, not campaign evidence or a GPU smoke."""
    from pmm_campaign_fixtures import write_fake_embeddings
    from training.run import run_training
    from training.campaign_runtime import CampaignContractError, replay_campaign_run
    import platform

    monkeypatch.setattr(platform, "platform", lambda: "synthetic CPU fixture")
    # The in-process CPU trainer records git metadata using short subprocesses.
    monkeypatch.setattr(subprocess, "Popen", ORIGINAL_POPEN)

    train, paths, protocol, weights = prepared
    esm_dir = Path(campaign.read_json(paths.feature_inventory)["esm"]["embeddings_dir"])
    write_fake_embeddings(train, esm_dir, dim=8)
    args = screen.parse_args(cli(prepared))
    command, _, identity = screen.build_command(paths, args, "only_esm", screen.file_sha(protocol))
    for flag, value in (("--epochs", "1"), ("--esm-dim", "8"), ("--hidden-s", "16"),
                        ("--hidden-v", "4"), ("--edge-hidden", "8"), ("--gvp-layers", "2"),
                        ("--esm-fusion-dim", "8")):
        command[command.index(flag) + 1] = value
    command += ["--load-workers", "1"]
    identity.update(epochs=1, synthetic_fixture=True)
    identity["resolved_config_sha256"] = campaign.stable_hash({k: v for k, v in
        config_to_payload(parse_args(command[3:])).items() if k not in campaign.NON_IDENTITY_CONFIG_KEYS})
    command[command.index("--campaign-run-identity") + 1] = json.dumps(identity)
    run = run_training(parse_args(command[3:]))
    receipt = replay_campaign_run(run)
    assert receipt["independent_replay"] and not receipt["normalization_refitted"]
    checked = screen.validate_completed_unit(paths, "only_esm", identity)
    assert checked["status"] == "passed"
    metadata = campaign.read_json(run / "run_metadata.json")
    expected = weights["folds"]["0"]["four_class_multipliers"]
    assert metadata["metal_class_weights"] == pytest.approx({
        "Mn": expected["mn"], "Cu": expected["cu"], "Zn": expected["zn"],
        "Fe": expected["class_viii"], "Class VIII": expected["class_viii"]})
    assert metadata["test_report"] is None
    prediction_path = run / "val_predictions.csv"
    with prediction_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["p_native_Mn"] = str(float(rows[0]["p_native_Mn"]) + 0.000005)
    with prediction_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    selected_path = run / "selected_checkpoint.json"
    selected = campaign.read_json(selected_path)
    selected["validation_predictions"]["sha256"] = screen.file_sha(prediction_path)
    campaign.write_json(selected_path, selected)
    with pytest.raises(CampaignContractError, match="Independent replay differs"):
        replay_campaign_run(run, output_dir=run / "strict_failure")
    assert not (run / "strict_failure/replay_receipt.json").exists()


@pytest.mark.parametrize("status,validation_failure", [("completed", False), ("failed_independent_replay", False),
                                                        ("completed", True)])
def test_real_persistence_retains_success_or_refusal_without_retry(prepared, monkeypatch, capsys, status, validation_failure):
    paths = prepared[1]
    monkeypatch.setattr(campaign, "campaign_manifest_guard", lambda *a, **k: None)
    monkeypatch.setattr(pmm_ion_features, "verify_frozen_feature_inventory", lambda *a: None)
    calls = []

    def unit(paths, command, env, identity, runs_dir, name, *, execution):
        assert execution.state["pending_unit"]["run_name"] == name
        calls.append(command)
        (runs_dir / name).mkdir(parents=True)
        (runs_dir / name / "fixture.txt").write_text("synthetic terminal artifact")
        return {"run_name": name, "status": status, "elapsed_seconds": 1}

    def validate_unit(*args):
        if validation_failure:
            raise ValueError("incorrect collapse")
        return {"status": "passed"}

    monkeypatch.setattr(campaign, "execute_unit", unit)
    monkeypatch.setattr(screen, "validate_completed_unit", validate_unit)
    passed = status == "completed" and not validation_failure
    assert screen.main(runtime_cli(prepared)) == (0 if passed else 1)
    result = json.loads(capsys.readouterr().out)["unit"]
    expected = "failed_five_class_validation" if validation_failure else status
    assert result["status"] == expected and len(calls) == 1
    backup = paths.root.parent / "independent"
    assert (backup / "runs" / screen.unit_name("only_gvp") / "fixture.txt").read_text() == "synthetic terminal artifact"
    saved = json.loads((backup / screen.artifact_paths(paths, "only_gvp")[-1].name).read_text())
    assert saved["units"][0]["status"] == expected
    assert not saved["promotion_ready"]
    with pytest.raises(ValueError, match="automatic retry"):
        screen.main(runtime_cli(prepared))
