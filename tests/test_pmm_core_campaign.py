"""Core-only preview and fail-closed execution; synthetic paths, no GPU/data."""
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


def test_default_plan_has_exact_24_units_without_child_reads_or_writes(scope_args, tmp_path, capsys, monkeypatch):
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
    assert result["required_core_fits"] == 30 and result["candidate_units"] == 24
    assert result["preserved_core_fold0_units"] == 6 and result["historical_awareness_fold0_units"] == 3
    assert result["paused_awareness_remaining_units"] == 12 and result["legacy_full_grid_fits"] == 45
    assert not result["execution_ready"] and not result["run_artifacts_examined"]
    assert not result["child_invoked"] and not result["files_written"]
    assert len({unit["run_name"] for unit in result["units"]}) == 24
    for unit in result["units"]:
        argv = unit["preview_argv_without_allocation_fields"]
        assert argv[argv.index("--readouts") + 1] == "none"
        assert unit["fold"] in (1, 2, 3, 4) and unit["seed"] == 42
        assert "first_shell_bias" not in argv and "--session-id" not in argv
    assert sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*")) == before


@pytest.mark.parametrize("arguments", [["--readouts", "first_shell_bias"], ["--readout", "none"],
                                       ["--campaign-action", "assess"], ["--action", "refit"],
                                       ["--action", "test"], ["--action", "folds"], ["--action", "smoke"],
                                       ["--epochs", "1"], ["--no-skip-existing"], ["--fold", "0"],
                                       ["--family", "unplanned"], ["--target", "five_class"],
                                       ["--camp", "truncated-flag"]])
def test_forbidden_actions_targets_and_passthrough_rejected(scope_args, arguments):
    with pytest.raises(SystemExit):
        core.main(scope_args + arguments)


def test_actual_run_is_blocked_until_policy_and_core_bridges_exist(scope_args):
    with pytest.raises(ValueError, match="TECH-023.*TECH-025"):
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
                                     ("queued_folds", [0, 1, 2, 3, 4]), ("required_core_fits", 45),
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
