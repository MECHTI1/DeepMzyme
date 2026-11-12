"""Execution-v4 binds private launch recipes to the complete published locks."""
import copy
import json
from pathlib import Path

import pytest

from ec_clean30 import phase3, runpod_controller, workstation as ws


def test_published_runtime_document_pin_covers_both_complete_package_maps():
    path = phase3.REPO_ROOT / ws.PUBLISHED_RUNTIME_LOCKS_PATH
    assert ws._sha(path) == ws.PUBLISHED_RUNTIME_LOCKS_SHA256
    locks = json.loads(path.read_text())["runtimes"]
    assert len(locks["fit"]["packages"]) == 187
    assert len(locks["esm1b"]["packages"]) == 26


@pytest.fixture
def locked_config(tmp_path, monkeypatch):
    document = json.loads((phase3.REPO_ROOT / ws.PUBLISHED_RUNTIME_LOCKS_PATH).read_text())
    repo = tmp_path / "repo"
    lock = repo / ws.PUBLISHED_RUNTIME_LOCKS_PATH
    lock.parent.mkdir(parents=True)
    stage, runtimes = [], copy.deepcopy(document["runtimes"])
    for index, (key, runtime) in enumerate(runtimes.items()):
        # Keep the complete real package maps; tiny archive stand-ins suffice
        # because this guard checks metadata before the full content preflight.
        runtime["bytes"] = document["runtimes"][key]["bytes"] = 8 + index
        runtime["archive"] = f"/workspace/relocated/{key}.tar.gz"
        local = tmp_path / f"{key}.tar.gz"
        local.write_bytes(b"x" * runtime["bytes"])
        stage.append({"local": str(local), "remote": runtime["archive"],
                      "sha256": runtime["archive_sha256"], "conda_unpack": True})
    lock.write_text(json.dumps(document))
    monkeypatch.setattr(ws, "PUBLISHED_RUNTIME_LOCKS_SHA256", ws._sha(lock))
    monkeypatch.setattr(phase3, "REPO_ROOT", repo)
    return {"provider": "runpod", "artifact_cache": {"fixture": True},
            "state_root": str(tmp_path / "state"), "execution_manifest": str(repo / "manifest.json"),
            "repo_url": "https://github.com/fixture/repo.git", "hardware_profile": "runpod_rtx4090",
            "runtime_environments": runtimes, "stage": stage}, lock


def test_v4_accepts_complete_published_recipes_with_relocated_archive_paths(locked_config):
    config, _ = locked_config
    ws._validate_published_runtime_locks(config, {"format": phase3.BOOTSTRAP_FORMAT})


@pytest.mark.parametrize("key", ["fit", "esm1b"])
@pytest.mark.parametrize("field", ["package_changed", "package_removed", "package_added", "python_version",
                                   "torch_cuda", "archive_sha256", "bytes"])
def test_v4_refuses_private_runtime_recipe_changes(locked_config, key, field):
    config, _ = locked_config
    runtime = config["runtime_environments"][key]
    if field == "package_changed":
        runtime["packages"]["torch"] = "changed"
    elif field == "package_removed":
        runtime["packages"].pop("torch")
    elif field == "package_added":
        runtime["packages"]["unexpected-package"] = "1.0"
    elif field == "bytes":
        runtime[field] += 1
    else:
        runtime[field] = "changed"
    with pytest.raises(ws.WorkstationError, match="complete published recipe"):
        ws._validate_published_runtime_locks(config, {"format": phase3.BOOTSTRAP_FORMAT})


@pytest.mark.parametrize("problem", ["document_bytes", "document_symlink", "stage_hash", "stage_size", "duplicate_stage"])
def test_v4_refuses_changed_document_or_archive_binding(locked_config, problem):
    config, lock = locked_config
    if problem == "document_bytes":
        lock.write_bytes(lock.read_bytes() + b"\n")
    elif problem == "document_symlink":
        moved = lock.with_suffix(".original")
        lock.rename(moved)
        lock.symlink_to(moved.name)
    elif problem == "stage_hash":
        config["stage"][0]["sha256"] = "0" * 64
    elif problem == "stage_size":
        Path(config["stage"][0]["local"]).write_bytes(b"changed size")
    else:
        config["stage"].append(dict(config["stage"][0]))
    with pytest.raises(ws.WorkstationError, match="published runtime|published recipe|runtime lock document"):
        ws._validate_published_runtime_locks(config, {"format": phase3.BOOTSTRAP_FORMAT})


@pytest.mark.parametrize("format_name", [phase3.EXECUTION_FORMAT, phase3.AUTOMATION_FORMAT, phase3.RUNPOD_FORMAT])
def test_historical_execution_formats_keep_their_existing_runtime_contract(format_name, monkeypatch):
    monkeypatch.setattr(ws, "_sha", lambda *_: pytest.fail("Historical format must not acquire a new document pin"))
    ws._validate_published_runtime_locks({}, {"format": format_name})


@pytest.mark.parametrize("entrypoint", ["preflight", "_prepared_preflight"])
def test_both_preflight_paths_bind_runtime_before_storage_or_archive_io(locked_config, monkeypatch, entrypoint):
    config, _ = locked_config
    config["runtime_environments"]["fit"]["packages"].pop("torch")
    monkeypatch.setattr(ws, "_validate_runpod_structure", lambda *_: None)
    monkeypatch.setattr(runpod_controller, "validate_config", lambda *_, **__: {})
    monkeypatch.setattr(phase3, "verify_execution_source", lambda *a, **kw: {"format": phase3.BOOTSTRAP_FORMAT})
    monkeypatch.setattr(ws.subprocess, "check_output", lambda *a, **kw: config["repo_url"])
    monkeypatch.setattr(ws, "_local_storage", lambda *_: pytest.fail("Runtime binding must precede storage and archive reads"))
    with pytest.raises(ws.WorkstationError, match="complete published recipe"):
        getattr(ws, entrypoint)(config, Path(config["execution_manifest"]), published=True)
