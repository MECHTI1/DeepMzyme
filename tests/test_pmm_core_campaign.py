"""Core continuation contracts; synthetic paths, no GPU or real held-out data."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

spec = importlib.util.spec_from_file_location("pmm_core_frontend", Path(__file__).parents[1] / "run_pmm_core_campaign.py")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


@pytest.fixture
def scope_args(tmp_path, monkeypatch):
    policy = tmp_path / "scope.json"
    policy.write_text(json.dumps(core.EXPECTED))
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "campaign_manifest.json").write_text(json.dumps({"campaign_id": core.EXPECTED["campaign_id"]}))
    train = tmp_path / "dataset/train"
    train.mkdir(parents=True)
    monkeypatch.setattr(core, "source_tree_sha256", lambda: core.EXPECTED["source_tree_sha256"])
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("Unexpected child process"))
    return ["--scope-file", str(policy), "--campaign-dir", str(campaign), "--train-dir", str(train)]


def run_args(scope_args):
    return scope_args + ["--action", "run", "--family", "only_gvp", "--target", "six_class", "--fold", "1",
                         "--session-id", "fixture", "--execution-deadline", "10000", "--allocation-started", "1000",
                         "--execution-max-seconds", "5000", "--estimated-fit-seconds", "2000",
                         "--durable-root", "/fixture/independent", "--persistence-mode", "host_pull"]


def test_default_plan_has_exact_36_units_without_child_reads_or_writes(scope_args, tmp_path, capsys, monkeypatch):
    original_read_bytes, original_read_text = Path.read_bytes, Path.read_text

    def forbid_run_read(path):
        assert not any(part in {"runs", "commands", "test", "reference"} for part in path.parts)

    def read_bytes(path, *args, **kwargs):
        forbid_run_read(path)
        return original_read_bytes(path, *args, **kwargs)

    def read_text(path, *args, **kwargs):
        forbid_run_read(path)
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(Path, "read_text", read_text)
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    assert core.main(scope_args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["required_core_fits"] == 45 and result["candidate_units"] == 36
    assert result["preserved_core_fold0_units"] == 9 and result["historical_awareness_fold0_units"] == 3
    assert result["paused_awareness_remaining_units"] == 12 and result["legacy_full_grid_fits"] == 45
    assert not result["execution_ready"] and not result["run_artifacts_examined"]
    assert not result["child_invoked"] and not result["files_written"]
    assert len({unit["run_name"] for unit in result["units"]}) == 36
    for unit in result["units"]:
        argv = unit["preview_argv_without_allocation_fields"]
        assert argv[1].endswith("run_pmm_core_campaign.py")
        assert argv[argv.index("--target") + 1] == unit["target"]
        assert argv[argv.index("--family") + 1] == unit["family"]
        assert argv[argv.index("--fold") + 1] == str(unit["fold"])
        assert unit["fold"] in (1, 2, 3, 4)
        assert unit["seed"] == 42
        assert "first_shell_bias" not in argv and "--session-id" not in argv
    assert sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*")) == before


@pytest.mark.parametrize("arguments", [["--readouts", "first_shell_bias"], ["--readout", "none"],
                                       ["--campaign-action", "assess"], ["--action", "refit"],
                                       ["--action", "test"], ["--action", "folds"], ["--action", "smoke"],
                                       ["--epochs", "1"], ["--no-skip-existing"], ["--fold", "5"],
                                       ["--family", "unplanned"], ["--target", "seven_class"],
                                       ["--camp", "truncated-flag"]])
def test_forbidden_actions_targets_and_passthrough_rejected(scope_args, arguments):
    with pytest.raises(SystemExit):
        core.main(scope_args + arguments)


def test_actual_run_is_blocked_until_readiness_exists(scope_args):
    with pytest.raises(ValueError, match="readiness file"):
        core.main(run_args(scope_args))


@pytest.mark.parametrize("artifact", ["directory", "log", "command", "dangling_symlink"])
def test_existing_unit_artifacts_never_trigger_reuse_or_retry(scope_args, artifact):
    campaign = Path(scope_args[scope_args.index("--campaign-dir") + 1])
    name = "only_gvp__six_class__none__fold1__seed42"
    if artifact == "command":
        path = campaign / "commands" / f"{name}.json"
    elif artifact == "log":
        path = campaign / "runs" / f"{name}.log"
    else:
        path = campaign / "runs" / name
    path.parent.mkdir()
    if artifact == "directory":
        path.mkdir()
    elif artifact == "dangling_symlink":
        path.symlink_to(campaign / "missing")
    else:
        path.write_text("preserved")
    with pytest.raises(ValueError, match="Existing unit artifact"):
        core.main(run_args(scope_args))


def test_run_requires_unit_and_allocation_fields(scope_args):
    with pytest.raises(ValueError, match="Select one"):
        core.main(scope_args + ["--action", "run"])
    with pytest.raises(ValueError, match="allocation/persistence"):
        core.main(scope_args + ["--action", "run", "--family", "only_gvp", "--target", "six_class", "--fold", "1"])


@pytest.mark.parametrize("key,value", [("source_tree_sha256", "changed"), ("readouts", ["first_shell_bias"]),
                                     ("queued_folds_by_target", {}), ("required_core_fits", 30),
                                     ("epochs", 1), ("model_seeds", [42, 43]), ("schema_version", True)])
def test_changed_scope_is_rejected(scope_args, key, value):
    path = Path(scope_args[1])
    data = json.loads(path.read_text())
    data[key] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Unsupported core scope field"):
        core.main(scope_args)


def test_actual_source_drift_campaign_mismatch_and_nontraining_path_refused(scope_args, monkeypatch):
    monkeypatch.setattr(core, "source_tree_sha256", lambda: "changed")
    with pytest.raises(ValueError, match="scientific source changed"):
        core.main(scope_args)
    monkeypatch.setattr(core, "source_tree_sha256", lambda: core.EXPECTED["source_tree_sha256"])
    campaign = Path(scope_args[scope_args.index("--campaign-dir") + 1])
    (campaign / "campaign_manifest.json").write_text('{"campaign_id":"other"}')
    with pytest.raises(ValueError, match="Campaign identity differs"):
        core.main(scope_args)
    with pytest.raises(ValueError, match="training-side"):
        core.main(scope_args[:-1] + [str(campaign / "test")])


def test_source_fingerprint_matches_frozen_scope_and_detects_both_source_locations(tmp_path):
    # Same path/content hashing rule as the frozen campaign, without importing torch.
    (tmp_path / "src").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "src/model.py").write_text("original")
    (tmp_path / "scripts/runner.py").write_text("original")
    first = core.source_tree_sha256(tmp_path)
    (tmp_path / "root_diagnostic.py").write_text("outside frozen source")
    assert core.source_tree_sha256(tmp_path) == first
    (tmp_path / "src/model.py").write_text("changed")
    second = core.source_tree_sha256(tmp_path)
    assert first != second
    (tmp_path / "scripts/runner.py").write_text("changed")
    assert core.source_tree_sha256(tmp_path) != second


@pytest.mark.parametrize("arguments", [["--family", "only_esm"], ["--session-id", "fixture"], ["--load-workers", "4"]])
def test_plan_does_not_silently_accept_partial_scope_or_allocation(scope_args, arguments):
    with pytest.raises(ValueError):
        core.main(scope_args + arguments)


@pytest.mark.parametrize("target", core.EXPECTED["targets"])
def test_no_fold_zero_can_be_retrained(scope_args, target):
    args = run_args(scope_args)
    args[args.index("--target") + 1] = target
    args[args.index("--fold") + 1] = "0"
    with pytest.raises(ValueError, match="all existing fold 0"):
        core.main(args)


@pytest.mark.parametrize("target", core.EXPECTED["targets"])
def test_native_command_recipe_is_preserved_and_policy_bound(tmp_path, monkeypatch, target):
    import sys
    from types import SimpleNamespace
    sys.path[:0] = [str(Path(__file__).parents[1]), str(Path(__file__).parents[1] / "src"),
                    str(Path(__file__).parent)]
    from benchmarking import pmm_ion_campaign as campaign
    from training.config import config_to_payload, parse_args
    from test_pmm_ion_campaign import _campaign_with_folds, _full_inventory
    from pmm_core_replay import POLICY_ID, POLICY_SHA256
    train, paths, weights = _campaign_with_folds(tmp_path, monkeypatch)
    _full_inventory(paths, tmp_path)
    args = SimpleNamespace(family="only_gvp", target=target, fold=1, python_bin=sys.executable,
                           train_dir=train, action="plan", load_workers=None)
    before = dict(campaign.TARGET_SCHEMES)
    argv, env, identity = core.build_command(paths, args)
    config = config_to_payload(parse_args(argv[3:]))
    assert config["metal_label_scheme"] == {"four_class": "merge_fe_class_viii",
                                            "five_class": "five_class", "six_class": "split_all_metals"}[target]
    assert config["metal_example_unit"] == "ion" and config["metal_eligibility_scheme"] == "six_class"
    assert (config["epochs"], config["batch_size"], config["fold_index"], config["seed"]) == (50, 16, 1, 42)
    assert config["selection_metric"] == "val_metal_balanced_acc" and not config["run_test_eval"]
    assert config["binding_residue_pooling"] == "none"
    assert identity["core_replay_contract"] == {"policy_id": POLICY_ID, "policy_sha256": POLICY_SHA256}
    assert "five_class_screen" not in identity
    assert identity["resolved_config_sha256"] == campaign.stable_hash({
        k: v for k, v in config.items() if k not in campaign.NON_IDENTITY_CONFIG_KEYS})
    if target == "five_class":
        expected = weights["folds"]["1"]["four_class_multipliers"]
        for label in ("mn", "cu", "zn", "class_viii"):
            assert config[label + "_loss_multiplier"] == expected[label]
        assert config["fe_loss_multiplier"] == expected["class_viii"]
    assert campaign.TARGET_SCHEMES == before


def test_readiness_revalidates_all_historical_units_and_rejects_omitted_evidence(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    sys.path.insert(0, str(Path(__file__).parents[1]))
    import pmm_core_replay as replay
    args = SimpleNamespace(campaign_dir=tmp_path, train_dir=tmp_path / "train",
                           readiness_file=tmp_path / "ready.json")
    calls = []
    monkeypatch.setattr(replay, "validate_core_unit", lambda *a, **k: calls.append(a[2:5]))
    monkeypatch.setattr(core, "readiness_hashes", lambda c: {"code": "fixed"})
    names = [core.unit_name(f, t, 0) for f in core.EXPECTED["families"] for t in core.EXPECTED["targets"]]
    evidence = {}
    for name in names:
        for artifact in ("best_model_checkpoint.pt", "run_config.json", "run_metadata.json",
                         "selected_checkpoint.json", "val_predictions.csv", "independent_validation_replay/val_predictions.csv"):
            path = tmp_path / "runs" / name / artifact
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture")
            evidence[str(path.relative_to(tmp_path))] = replay.sha(path)
    record = dict(schema="pmm-core-readiness-v1", campaign_id=core.EXPECTED["campaign_id"],
                  scope_sha256="scope", source_tree_sha256=core.EXPECTED["source_tree_sha256"],
                  policy_id=replay.POLICY_ID, policy_sha256=replay.POLICY_SHA256,
                  implementation_sha256={"code": "fixed"}, historical_units=[{"run_name": n} for n in names],
                  historical_evidence_sha256=evidence)
    args.readiness_file.write_text(json.dumps(record))
    core.verify_readiness(args, "scope")
    assert len(calls) == 9 and set(calls) == {(f, t, 0) for f in core.EXPECTED["families"] for t in core.EXPECTED["targets"]}
    record["historical_evidence_sha256"] = {}
    args.readiness_file.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="coverage"):
        core.verify_readiness(args, "scope")


@pytest.mark.parametrize("action", ["assess", "refit-preview"])
def test_assessment_actions_forward_explicit_scope_file(scope_args, tmp_path, monkeypatch, action):
    import sys
    sys.path.insert(0, str(Path(__file__).parents[1]))
    import pmm_core_assessment as assessment
    calls = []

    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "synthetic_adapter_result"}

    name = "assess_core" if action == "assess" else "preview_core_refit"
    monkeypatch.setattr(assessment, name, capture)
    argv = scope_args + ["--action", action, "--output-dir", str(tmp_path / "report")]
    if action == "refit-preview":
        argv += ["--decision-path", str(tmp_path / "core_validation_decision.json"),
                 "--test-route", "zenodo_pmm_secondary_reference"]
    assert core.main(argv) == 0
    assert len(calls) == 1
    assert calls[0][1]["scope_path"] == Path(scope_args[scope_args.index("--scope-file") + 1])
    assert calls[0][1]["scope_path"] != assessment.SCOPE_PATH


def test_early_child_failure_keeps_core_status_in_host_pull_manifest_under_lock(scope_args, tmp_path, monkeypatch):
    import fcntl
    import sys
    import time
    sys.path[:0] = [str(Path(__file__).parents[1]), str(Path(__file__).parents[1] / "src")]
    from benchmarking import pmm_ion_campaign as campaign
    from benchmarking import pmm_ion_features as features
    from training import access_guard

    args = core.parse_args(run_args(scope_args) + ["--readiness-file", str(tmp_path / "ready.json")])
    args.allocation_started = time.time()
    args.execution_deadline = args.allocation_started + 3600
    args.execution_max_seconds = 3600
    args.estimated_fit_seconds = 100
    args.durable_root = tmp_path / "independent_host"
    paths = campaign.CampaignPaths(args.campaign_dir)
    identity = {"synthetic": "failure before run-directory creation"}
    readiness_checked = []

    def locked_readiness(*_args):
        with (paths.root / "execution.lock").open("a+") as competing_lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(competing_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        readiness_checked.append(True)

    def failed_child(paths, command, env, admitted_identity, runs, name, **kwargs):
        assert readiness_checked and admitted_identity == identity
        runs.mkdir(exist_ok=True)
        (runs / f"{name}.log").write_text("synthetic argument/parser failure before run creation\n")
        campaign.write_json(paths.commands / f"{name}.json", {"identity": identity})
        assert not (runs / name).exists()
        return {"run_name": name, "status": "failed", "return_code": 2}

    monkeypatch.setattr(core, "verify_readiness", locked_readiness)
    monkeypatch.setattr(core, "build_command", lambda *a: ([], {}, identity))
    monkeypatch.setattr(campaign, "campaign_manifest_guard", lambda *a, **k: None)
    monkeypatch.setattr(features, "verify_frozen_feature_inventory", lambda *a: None)
    monkeypatch.setattr(access_guard, "install_forbidden_read_guard", lambda *a: None)
    monkeypatch.setattr(campaign, "execute_unit", failed_child)

    assert core.execute_one(paths, args, [], {}, identity) == 1
    name = core.unit_name(args.family, args.target, args.fold)
    status_name = f"run_status_core_{name}.json"
    status = json.loads((paths.root / status_name).read_text())
    assert status["status"] == status["original_strict_execution_status"] == "failed"
    state = json.loads((paths.root / "execution_state.json").read_text())
    assert state["pending_unit"]["status"] == "failed" and not state["pending_unit"]["persisted"]
    manifest = json.loads((paths.root / state["pending_transfer"]["manifest_path"]).read_text())
    transferred = {entry["path"] for entry in manifest["files"]}
    assert {status_name, f"runs/{name}.log", f"commands/{name}.json"} <= transferred
    assert not (paths.runs / name).exists()
