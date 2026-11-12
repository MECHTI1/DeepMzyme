"""Local ledger tests only; no provider, background process or GPU operations."""
import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ec_clean30 import execution_v5 as v5


AUTH = "a" * 64
SOURCE = "b" * 64


@pytest.fixture
def ledger(tmp_path):
    directory = tmp_path / "shared"
    directory.mkdir(mode=0o700)
    path = directory / "launch_windows.json"
    path.write_text(json.dumps(v5.empty_failure_ledger(path, AUTH)))
    path.chmod(0o600)
    return v5.FailedWindowLedger(path, AUTH), tmp_path


def begin(ledger, root, name, *, config="c" * 64):
    return ledger.begin_window(name, config_sha256=config, execution_manifest_sha256=SOURCE, pass_root=root / name)


def fail(ledger, name):
    return ledger.finish_window(name, outcome="failed", closeout_sha256="d" * 64, provider_terminal="verified_absent")


def test_two_failed_windows_persist_across_new_instances_passes_and_configs(ledger):
    current, root = ledger
    begin(current, root, "pass1")
    fail(current, "pass1")
    restarted = v5.FailedWindowLedger(current.path, AUTH)
    begin(restarted, root, "pass2", config="e" * 64)
    fail(restarted, "pass2")
    again = v5.FailedWindowLedger(current.path, AUTH)
    with pytest.raises(v5.LaunchWindowError, match="Two additional"):
        begin(again, root, "pass3", config="f" * 64)
    assert again.status()["failed_windows"] == 2
    assert len(json.loads(current.path.read_text())["windows"]) == 2


def test_unresolved_window_blocks_fresh_launch_but_exact_recovery_does_not_reset(ledger):
    current, root = ledger
    first = begin(current, root, "first")
    original = current.path.read_bytes()
    assert begin(v5.FailedWindowLedger(current.path, AUTH), root, "first") == first
    assert current.path.read_bytes() == original
    with pytest.raises(v5.LaunchWindowError, match="unresolved"):
        begin(current, root, "second", config="e" * 64)
    with pytest.raises(v5.LaunchWindowError, match="differently bound"):
        begin(current, root, "first", config="e" * 64)
    assert current.status()["unresolved_window"]["window_id"] == "first"


def test_recovery_query_matches_only_the_exact_open_binding_and_never_writes(ledger):
    current, root = ledger

    def holds(name, *, config="c" * 64, manifest=SOURCE, pass_root=None):
        return current.holds_open(name, config_sha256=config, execution_manifest_sha256=manifest,
                                  pass_root=root / name if pass_root is None else pass_root)

    assert not holds("first")
    begin(current, root, "first")
    original = current.path.read_bytes()
    assert holds("first")
    assert not holds("first", config="e" * 64) and not holds("first", manifest="e" * 64)
    assert not holds("first", pass_root=root / "other") and not holds("second")
    assert current.path.read_bytes() == original
    fail(current, "first")
    assert not holds("first")
    with pytest.raises(v5.LaunchWindowError, match="inside a pass directory"):
        holds("bad", pass_root=current.path.parent)


def test_completed_window_does_not_reset_previous_failures_and_cannot_restart(ledger):
    current, root = ledger
    begin(current, root, "failed")
    fail(current, "failed")
    begin(current, root, "completed")
    current.finish_window("completed", outcome="completed", closeout_sha256="e" * 64,
                          provider_terminal="verified_stopped")
    assert current.status()["failed_windows"] == 1
    with pytest.raises(v5.LaunchWindowError, match="closed"):
        begin(current, root, "completed")
    begin(current, root, "last")
    fail(current, "last")
    assert current.status()["remaining_failed_windows"] == 0


def test_closeout_is_idempotent_and_cannot_change_evidence_or_outcome(ledger):
    current, root = ledger
    begin(current, root, "window")
    first = fail(current, "window")
    assert fail(current, "window") == first
    with pytest.raises(v5.LaunchWindowError, match="cannot be changed"):
        current.finish_window("window", outcome="completed", closeout_sha256="d" * 64,
                              provider_terminal="verified_absent")
    with pytest.raises(v5.LaunchWindowError, match="unreserved"):
        fail(current, "unknown")


@pytest.mark.parametrize("problem", ["missing", "authorization", "copied", "limit", "permissions", "alias"])
def test_missing_or_unbound_ledger_fails_closed(ledger, problem):
    current, root = ledger
    path, auth = current.path, AUTH
    if problem == "missing": path.unlink()
    elif problem == "authorization": auth = "e" * 64
    elif problem == "copied":
        path = current.path.with_name("copy.json")
        path.write_bytes(current.path.read_bytes())
        path.chmod(0o600)
    elif problem == "limit":
        document = json.loads(path.read_text())
        document["failed_window_limit"] = 3
        path.write_text(json.dumps(document))
    elif problem == "permissions": path.chmod(0o644)
    elif problem == "alias": os.link(path, path.with_name("hardlink.json"))
    with pytest.raises(v5.LaunchWindowError):
        v5.FailedWindowLedger(path, auth).status()


def test_no_implicit_ledger_creation_or_pass_scoped_reset(tmp_path, ledger):
    missing = tmp_path / "missing.json"
    with pytest.raises(v5.LaunchWindowError, match="missing"):
        v5.FailedWindowLedger(missing, AUTH)
    assert not missing.exists()
    current, _ = ledger
    with pytest.raises(v5.LaunchWindowError, match="inside a pass directory"):
        current.begin_window("bad", config_sha256="c" * 64, execution_manifest_sha256=SOURCE,
                             pass_root=current.path.parent)


def test_outcome_requires_explicit_terminal_evidence(ledger):
    current, root = ledger
    begin(current, root, "window")
    with pytest.raises(v5.LaunchWindowError, match="terminal"):
        current.finish_window("window", outcome="failed", closeout_sha256="d" * 64, provider_terminal="unknown")
    assert current.status()["unresolved_window"] is not None
