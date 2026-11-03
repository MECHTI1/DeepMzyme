"""Cached bootstrap execution preserves the complete published provider lineage."""
import json

import pytest

from ec_clean30 import phase3
from test_ec_clean30_execution_v2 import automation_source
from test_ec_clean30_execution_v3 import runpod_source
from test_ec_clean30_phase3 import execution_source, _git, _sha


@pytest.fixture
def bootstrap_source(runpod_source):
    repo, original, automation, runpod = runpod_source
    _git(repo, "add", "src", "scripts", "docs")
    _git(repo, "commit", "--quiet", "-m", "Fixture published provider extension")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    commit = _git(repo, "rev-parse", "HEAD")
    for name in set(phase3.BOOTSTRAP_FILES) - set(phase3.RUNPOD_FILES):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# cached bootstrap helper\n")
    (repo / phase3.EXECUTION_FILES[0]).write_text("# versioned bootstrap authentication\n")
    out = runpod.with_name("bootstrap.json")
    phase3.make_bootstrap_manifest(runpod, out, base_commit=commit, repo=repo)
    return repo, original, automation, runpod, out, commit


def test_bootstrap_extension_preserves_published_predecessors_and_requires_publication(bootstrap_source):
    repo, original, automation, runpod, out, commit = bootstrap_source
    predecessors = {path: path.read_bytes() for path in (original, automation, runpod)}
    record = phase3.verify_execution_source(out, repo=repo, require_published=False)
    assert record["format"] == phase3.BOOTSTRAP_FORMAT
    assert set(record["source_sha256"]) == set(phase3.BOOTSTRAP_FILES)
    assert len(phase3.RUNPOD_FILES) == 8 and len(phase3.BOOTSTRAP_FILES) == 9
    assert record["base_runpod"] == {"commit": commit, "path": runpod.relative_to(repo).as_posix(),
                                     "sha256": _sha(runpod)}
    assert record["hardware_profiles"] == sorted(phase3.GPU_PROFILES)
    with pytest.raises(phase3.Phase3Error, match="published"):
        phase3.verify_execution_source(out, repo=repo)
    _git(repo, "add", "src", "scripts", "docs")
    _git(repo, "commit", "--quiet", "-m", "Fixture cached bootstrap extension")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    assert phase3.verify_execution_source(out, repo=repo)["published"]
    assert all(path.read_bytes() == content for path, content in predecessors.items())


@pytest.mark.parametrize("case", [
    "wp13_bytes", "automation_bytes", "runpod_bytes", "base_hash", "base_commit", "base_path",
    "timing_inputs", "chain", "base_execution", "base_automation", "hardware_profiles",
    "bootstrap_source", "provider_source", "frozen_source", "extra_pin", "missing_pin",
])
def test_bootstrap_extension_refuses_tampered_lineage_inputs_or_source(bootstrap_source, case):
    repo, original, automation, runpod, out, _ = bootstrap_source
    record = json.loads(out.read_text())
    if case in {"wp13_bytes", "automation_bytes", "runpod_bytes"}:
        path = {"wp13_bytes": original, "automation_bytes": automation, "runpod_bytes": runpod}[case]
        path.write_bytes(path.read_bytes() + b"\n")
    elif case == "base_hash":
        record["base_runpod"]["sha256"] = "0" * 64
    elif case == "base_commit":
        record["base_runpod"]["commit"] = "HEAD"
    elif case == "base_path":
        record["base_runpod"]["path"] = "../runpod.json"
    elif case == "timing_inputs":
        record["timing_inputs_sha256"] = "0" * 64
    elif case == "chain":
        record["amendment_chain"] = ["0" * 64]
    elif case in {"base_execution", "base_automation"}:
        record[case]["sha256"] = "0" * 64
    elif case == "hardware_profiles":
        record["hardware_profiles"].append("unreviewed_gpu")
    elif case == "bootstrap_source":
        (repo / "src/ec_clean30/runpod_bootstrap.py").write_text("changed bootstrap\n")
    elif case == "provider_source":
        (repo / "src/ec_clean30/runpod_controller.py").write_text("changed provider control\n")
    elif case == "frozen_source":
        (repo / "src/frozen.py").write_text("changed scientific source\n")
    elif case == "extra_pin":
        extra = repo / "src/unapproved.py"
        extra.write_text("# unapproved\n")
        record["source_sha256"]["src/unapproved.py"] = _sha(extra)
    else:
        record["source_sha256"].pop("src/ec_clean30/runpod_bootstrap.py")
    out.write_text(json.dumps(record))
    with pytest.raises(phase3.Phase3Error):
        phase3.verify_execution_source(out, repo=repo, require_published=False)


def test_bootstrap_base_must_be_published(bootstrap_source):
    repo, _, _, _, out, _ = bootstrap_source
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD~1")
    with pytest.raises(phase3.Phase3Error, match="Runpod base must be published"):
        phase3.verify_execution_source(out, repo=repo, require_published=False)


def test_bootstrap_base_must_be_an_ancestor_of_the_execution_checkout(bootstrap_source):
    repo, _, _, _, out, original_commit = bootstrap_source
    (repo / "branch-marker.txt").write_text("published side branch\n")
    _git(repo, "add", "branch-marker.txt")
    _git(repo, "commit", "--quiet", "-m", "Fixture published descendant")
    descendant = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", "refs/remotes/origin/main", descendant)
    _git(repo, "checkout", "--quiet", "--detach", original_commit)
    record = json.loads(out.read_text())
    record["base_runpod"]["commit"] = descendant
    out.write_text(json.dumps(record))
    with pytest.raises(phase3.Phase3Error, match="ancestor"):
        phase3.verify_execution_source(out, repo=repo, require_published=False)


def test_bootstrap_authenticates_historical_provider_source_pins(bootstrap_source):
    repo, _, _, runpod, out, _ = bootstrap_source
    base = json.loads(runpod.read_text())
    base["source_sha256"]["src/ec_clean30/runpod_controller.py"] = "0" * 64
    runpod.write_text(json.dumps(base))
    _git(repo, "add", str(runpod.relative_to(repo)))
    _git(repo, "commit", "--quiet", "-m", "Fixture invalid historical source pin")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    record = json.loads(out.read_text())
    record["base_runpod"].update(commit=_git(repo, "rev-parse", "HEAD"), sha256=_sha(runpod))
    out.write_text(json.dumps(record))
    with pytest.raises(phase3.Phase3Error, match="Published Runpod source pin differs"):
        phase3.verify_execution_source(out, repo=repo, require_published=False)


def test_bootstrap_maker_refuses_non_provider_base_before_writing(bootstrap_source):
    repo, _, automation, _, _, commit = bootstrap_source
    out = automation.with_name("invalid_bootstrap.json")
    with pytest.raises(phase3.Phase3Error, match="eight-file Runpod"):
        phase3.make_bootstrap_manifest(automation, out, base_commit=commit, repo=repo)
    assert not out.exists()


@pytest.mark.parametrize("profile", ["runpod_l4", "runpod_rtx4090"])
@pytest.mark.parametrize("format_name", [phase3.RUNPOD_FORMAT, phase3.BOOTSTRAP_FORMAT])
def test_runpod_profile_accepts_only_explicit_provider_extensions(profile, format_name):
    source = {"format": format_name, "hardware_profiles": [profile]}
    assert phase3._approved_runtime_profile(source, profile)["name"] == profile
    source["hardware_profiles"] = []
    with pytest.raises(phase3.Phase3Error, match="authenticated provider extension"):
        phase3._approved_runtime_profile(source, profile)


def test_bootstrap_manifest_cli_dispatches_the_explicit_base_and_commit(tmp_path, monkeypatch, capsys):
    base, out, commit = tmp_path / "provider.json", tmp_path / "bootstrap.json", "a" * 40
    calls = []

    def make(actual_base, actual_out, *, base_commit):
        calls.append((actual_base, actual_out, base_commit))
        return {"format": phase3.BOOTSTRAP_FORMAT}

    monkeypatch.setattr(phase3, "make_bootstrap_manifest", make)
    assert phase3.main(["bootstrap-execution-manifest", "--base-runpod", str(base),
                        "--base-commit", commit, "--out", str(out)]) == 0
    assert calls == [(base, out, commit)]
    assert json.loads(capsys.readouterr().out)["format"] == phase3.BOOTSTRAP_FORMAT
