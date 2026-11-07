"""Small local fixtures for terminated-child restoration; no models or subprocesses."""
import json
import os
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ec_clean30 import hot_storage as hot


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else value.encode())
    return path


@pytest.fixture
def attempt(tmp_path):
    canonical = tmp_path / "campaign" / "runs" / "clean_unit" / "attempt1"
    prepared = hot.prepare_hot_attempt(tmp_path / "hot", canonical, session_id="session", unit_id="clean_unit",
        attempt=1, execution_manifest_sha256="a" * 64, g3_sha256="b" * 64)
    source = Path(prepared["hot_attempt"])
    write(source / "aaa.log", "terminal child log\n")
    write(source / "work/fit_manifest.json", b'{ "original_receipt" : true, "path" : "/original/hot/path" }\n')
    export = write(source / "validation.npz", b"small synthetic export bytes")
    write(source / "validation_export_receipt.json", b'{"immutable_export_sidecar":true}\n')
    launcher = write(source / "capped_clean_python", b"#!/bin/sh\nexit 0\n")
    launcher.chmod(0o700)
    result = {"status": "passed", "kind": "clean", "unit_id": "clean_unit",
              "export": str(canonical / export.name), "export_sha256": hot._sha(export)}
    write(source / hot.RESULT, json.dumps(result))
    return {**prepared, "source": source, "canonical": canonical, "execution_root": canonical.parent.parent.parent}


def test_restore_verifies_all_bytes_preserves_receipts_and_publishes_result_last(attempt, monkeypatch):
    events = []
    copy, atomic = hot._copy_regular, hot._atomic
    def traced_copy(source, destination, *args):
        events.append(("copy", destination.name))
        return copy(source, destination, *args)
    def traced_atomic(path, content, **kwargs):
        events.append(("marker" if path.name == hot.MARKER else "metadata", path.name))
        return atomic(path, content, **kwargs)
    monkeypatch.setattr(hot, "_copy_regular", traced_copy)
    monkeypatch.setattr(hot, "_atomic", traced_atomic)
    receipt = hot.restore_hot_attempt(attempt["marker_path"], worker_terminated=True)
    assert receipt["successful_result"] is True
    assert [name for kind, name in events if kind == "copy"][-1] == hot.RESULT
    assert events[-1] == ("marker", hot.MARKER)
    for row in receipt["files"]:
        assert (attempt["source"] / row["path"]).read_bytes() == (attempt["canonical"] / row["path"]).read_bytes()
    assert (attempt["canonical"] / "capped_clean_python").stat().st_mode & 0o100
    hot.require_restored_attempts(attempt["execution_root"])
    assert hot.restore_hot_attempt(attempt["marker_path"], worker_terminated=True) == receipt


def test_pending_marker_blocks_recovery_and_restore_requires_confirmed_termination(attempt):
    with pytest.raises(hot.HotStorageError, match="Pending hot data"):
        hot.require_restored_attempts(attempt["execution_root"])
    with pytest.raises(hot.HotStorageError, match="confirm worker termination"):
        hot.restore_hot_attempt(attempt["marker_path"], worker_terminated=False)
    assert not (attempt["canonical"] / hot.RESULT).exists()


@pytest.mark.parametrize("field", [None, "session_id", "unit_id", "attempt", "execution_manifest_sha256", "g3_sha256"])
def test_worker_accepts_only_its_exact_pending_identity(attempt, field):
    identity = {"session_id": "session", "unit_id": "clean_unit", "attempt": 1,
                "execution_manifest_sha256": "a" * 64, "g3_sha256": "b" * 64}
    if field:
        identity[field] = 2 if field == "attempt" else "changed"
        with pytest.raises(hot.HotStorageError, match="Worker identity"):
            hot.worker_hot_attempt(attempt["marker_path"], **identity)
    else:
        assert hot.worker_hot_attempt(attempt["marker_path"], **identity) == attempt["source"]
        hot.restore_hot_attempt(attempt["marker_path"], worker_terminated=True)
        with pytest.raises(hot.HotStorageError, match="pending"):
            hot.worker_hot_attempt(attempt["marker_path"], **identity)


def test_copy_failure_leaves_pending_and_retry_reuses_only_identical_partial_files(attempt, monkeypatch):
    original = hot._copy_regular
    def interrupted(source, destination, *args):
        if destination.name == "validation.npz":
            raise OSError("synthetic failed copy")
        return original(source, destination, *args)
    monkeypatch.setattr(hot, "_copy_regular", interrupted)
    with pytest.raises(OSError, match="failed copy"):
        hot.restore_hot_attempt(attempt["marker_path"], worker_terminated=True)
    assert (attempt["canonical"] / "aaa.log").is_file()
    assert not (attempt["canonical"] / hot.RESULT).exists()
    with pytest.raises(hot.HotStorageError, match="Pending hot data"):
        hot.require_restored_attempts(attempt["execution_root"])
    monkeypatch.setattr(hot, "_copy_regular", original)
    hot.restore_hot_attempt(attempt["marker_path"], worker_terminated=True)


def test_hot_mutation_during_copy_cannot_publish_completion(attempt, monkeypatch):
    original = hot._copy_regular
    def changed(source, destination, *args):
        result = original(source, destination, *args)
        if destination.name == "aaa.log":
            write(attempt["source"] / "late_output.txt", "changed while copying")
        return result
    monkeypatch.setattr(hot, "_copy_regular", changed)
    with pytest.raises(hot.HotStorageError, match="mutated during restoration"):
        hot.restore_hot_attempt(attempt["marker_path"], worker_terminated=True)
    assert not (attempt["canonical"] / hot.RESULT).exists()
    assert json.loads(Path(attempt["marker_path"]).read_text())["status"] == "pending"


def farm_link(attempt):
    source = attempt["source"]
    target = write(source / "farms/base/P1.pt", b"verified farm embedding")
    content = "\t".join(hot.FARM_COLUMNS) + "\n"
    content += "\t".join(["P1", "", "e" * 64, "esm1b_l33_mean", "33", "1280", hot._sha(target)]) + "\n"
    write(target.parent / hot.FARM_MANIFEST, content)
    link = source / "work/data/esm_data/P1.pt"
    link.parent.mkdir(parents=True)
    link.symlink_to("../../../farms/base/P1.pt")
    return link, target


def test_interrupted_internal_farm_links_are_verified_then_materialized(attempt):
    link, target = farm_link(attempt)
    receipt = hot.restore_hot_attempt(attempt["marker_path"], worker_terminated=True)
    restored = attempt["canonical"] / link.relative_to(attempt["source"])
    assert restored.is_file() and not restored.is_symlink() and restored.read_bytes() == target.read_bytes()
    entry = next(row for row in receipt["files"] if row["path"] == "work/data/esm_data/P1.pt")
    assert entry["materialized_from"] == "farms/base/P1.pt"


@pytest.mark.parametrize("problem", ["external", "bad_hash", "directory", "unexpected_link", "fifo"])
def test_unsafe_links_and_special_files_fail_closed(attempt, tmp_path, problem):
    link, target = farm_link(attempt)
    if problem == "external":
        link.unlink()
        link.symlink_to(write(tmp_path / "P1.pt", b"outside hot attempt"))
    elif problem == "bad_hash":
        target.write_bytes(b"changed farm embedding")
    elif problem == "directory":
        (attempt["source"] / "linked_directory").symlink_to(target.parent, target_is_directory=True)
    elif problem == "unexpected_link":
        (attempt["source"] / "P1.pt").symlink_to(target)
    elif problem == "fifo":
        os.mkfifo(attempt["source"] / "pipe")
    with pytest.raises(hot.HotStorageError):
        hot.restore_hot_attempt(attempt["marker_path"], worker_terminated=True)
    assert not (attempt["canonical"] / hot.RESULT).exists()


def test_successful_result_must_already_reference_canonical_export(attempt):
    path = attempt["source"] / hot.RESULT
    result = json.loads(path.read_text())
    result["export"] = str(attempt["source"] / "validation.npz")
    path.write_text(json.dumps(result))
    with pytest.raises(hot.HotStorageError, match="canonical CLEAN export"):
        hot.restore_hot_attempt(attempt["marker_path"], worker_terminated=True)


def test_cleanup_requires_ack_and_result_and_leaves_canonical_bytes_unchanged(attempt):
    hot.restore_hot_attempt(attempt["marker_path"], worker_terminated=True)
    before = {path.relative_to(attempt["canonical"]).as_posix(): path.read_bytes()
              for path in attempt["canonical"].rglob("*") if path.is_file()}
    with pytest.raises(hot.HotStorageError, match="HostPull acknowledgment"):
        hot.cleanup_hot_attempt(attempt["marker_path"], host_ack_verified=False, result_verified=True)
    assert hot.cleanup_hot_attempt(attempt["marker_path"], host_ack_verified=True, result_verified=True)["status"] == "removed"
    assert not attempt["source"].exists()
    after = {path.relative_to(attempt["canonical"]).as_posix(): path.read_bytes()
             for path in attempt["canonical"].rglob("*") if path.is_file()}
    assert before == after
    hot.require_restored_attempts(attempt["execution_root"])
    assert hot.cleanup_hot_attempt(attempt["marker_path"], host_ack_verified=True, result_verified=True)["status"] == "already_absent"


@pytest.mark.parametrize("tree", ["source", "canonical"])
def test_changed_hot_or_canonical_bytes_block_cleanup(attempt, tree):
    hot.restore_hot_attempt(attempt["marker_path"], worker_terminated=True)
    (attempt[tree] / "validation.npz").write_bytes(b"changed after restore")
    with pytest.raises(hot.HotStorageError, match="changed|no longer matches"):
        hot.cleanup_hot_attempt(attempt["marker_path"], host_ack_verified=True, result_verified=True)
    assert attempt["source"].is_dir()


def test_failure_attempt_is_fully_restored_but_never_cleaned(attempt):
    (attempt["source"] / hot.RESULT).unlink()
    receipt = hot.restore_hot_attempt(attempt["marker_path"], worker_terminated=True)
    assert receipt["successful_result"] is False
    assert (attempt["canonical"] / "aaa.log").read_bytes() == (attempt["source"] / "aaa.log").read_bytes()
    hot.require_restored_attempts(attempt["execution_root"])
    with pytest.raises(hot.HotStorageError, match="Failure attempts"):
        hot.cleanup_hot_attempt(attempt["marker_path"], host_ack_verified=True, result_verified=True)
    assert attempt["source"].is_dir()


@pytest.mark.parametrize("problem", ["overlap", "symlink", "not_private", "existing"])
def test_prepare_rejects_unsafe_placement_and_never_restarts_attempt(tmp_path, problem):
    canonical = tmp_path / "campaign/runs/unit/attempt1"
    root = tmp_path / "hot"
    if problem == "overlap": root = canonical.parent
    elif problem == "symlink":
        (tmp_path / "real").mkdir()
        root.symlink_to(tmp_path / "real", target_is_directory=True)
    elif problem == "not_private": root.mkdir(mode=0o755)
    elif problem == "existing": canonical.mkdir(parents=True)
    with pytest.raises(hot.HotStorageError):
        hot.prepare_hot_attempt(root, canonical, session_id="session", unit_id="unit", attempt=1,
                                execution_manifest_sha256="a" * 64, g3_sha256="b" * 64)


def test_relocated_verification_reads_only_local_files_and_preserves_remote_identity(attempt, tmp_path, monkeypatch):
    farm_link(attempt)
    hot.restore_hot_attempt(attempt["marker_path"], worker_terminated=True)
    local = tmp_path / "independent_copy"
    shutil.copytree(attempt["canonical"], local)
    original = str(attempt["canonical"])
    remote_paths = [attempt["canonical"], Path(json.loads((local / hot.MARKER).read_text())["identity"]["hot_root"])]
    shutil.rmtree(attempt["canonical"])
    shutil.rmtree(remote_paths[1])
    originals = {name: getattr(Path, name) for name in ("stat", "lstat", "resolve")}
    for name, method in originals.items():
        def local_only(path, *args, _method=method, **kwargs):
            assert not any(path == root or root in path.parents for root in remote_paths), "Recorded remote path was accessed"
            return _method(path, *args, **kwargs)
        monkeypatch.setattr(Path, name, local_only)
    before = (local / hot.MARKER).read_bytes()
    result = hot.verify_preserved_attempt(local, original_attempt=original)
    assert result["status"] == "verified" and result["successful_result"] is True
    assert result["original_attempt"] == original and (local / hot.MARKER).read_bytes() == before


@pytest.mark.parametrize("problem", ["pending", "original", "identity", "invalid_source", "manifest_hash",
                                     "artifact", "missing", "extra", "symlink"])
def test_relocated_verifier_rejects_partial_foreign_or_changed_preservation(attempt, tmp_path, problem):
    hot.restore_hot_attempt(attempt["marker_path"], worker_terminated=True)
    local = tmp_path / "preserved"
    shutil.copytree(attempt["canonical"], local)
    original = attempt["canonical"]
    marker_path = local / hot.MARKER
    marker = json.loads(marker_path.read_text())
    if problem == "pending":
        for path in list(local.iterdir()):
            if path == marker_path:
                continue
            shutil.rmtree(path) if path.is_dir() else path.unlink()
        marker["status"] = "pending"
        marker.pop("restore_manifest_sha256")
        marker_path.write_bytes(hot._bytes(marker))
    elif problem == "original": original = original.parent / "attempt2"
    elif problem == "identity":
        marker["identity"]["g3_sha256"] = "c" * 64
        marker_path.write_bytes(hot._bytes(marker))
    elif problem == "invalid_source":
        marker["identity"]["execution_manifest_sha256"] = "not-a-digest"
        marker["identity_sha256"] = hot._digest(marker["identity"])
        marker_path.write_bytes(hot._bytes(marker))
    elif problem == "manifest_hash":
        (local / hot.RESTORE_MANIFEST).write_text("{}")
    elif problem == "artifact": (local / "validation.npz").write_bytes(b"changed bytes")
    elif problem == "missing": (local / "validation_export_receipt.json").unlink()
    elif problem == "extra": write(local / "unrecorded.txt", "foreign output")
    elif problem == "symlink":
        (local / "validation.npz").unlink()
        (local / "validation.npz").symlink_to(attempt["source"] / "validation.npz")
    with pytest.raises((hot.HotStorageError, OSError)):
        hot.verify_preserved_attempt(local, original_attempt=original)


def test_relocated_verifier_accepts_restored_failed_attempt_without_success_claim(attempt, tmp_path):
    (attempt["source"] / hot.RESULT).unlink()
    hot.restore_hot_attempt(attempt["marker_path"], worker_terminated=True)
    local = tmp_path / "preserved_failure"
    shutil.copytree(attempt["canonical"], local)
    result = hot.verify_preserved_attempt(local, original_attempt=attempt["canonical"])
    assert result["status"] == "verified" and result["successful_result"] is False


def test_relocated_verifier_checks_materialized_link_provenance_even_with_matching_inventory_hashes(attempt, tmp_path):
    farm_link(attempt)
    hot.restore_hot_attempt(attempt["marker_path"], worker_terminated=True)
    local = tmp_path / "preserved"
    shutil.copytree(attempt["canonical"], local)
    receipt_path = local / hot.RESTORE_MANIFEST
    receipt = json.loads(receipt_path.read_text())
    row = next(row for row in receipt["files"] if "materialized_from" in row)
    row["materialized_from"] = "/outside/P1.pt"
    receipt_path.write_bytes(hot._bytes(receipt))
    marker_path = local / hot.MARKER
    marker = json.loads(marker_path.read_text())
    marker["restore_manifest_sha256"] = hot._sha(receipt_path)
    marker_path.write_bytes(hot._bytes(marker))
    with pytest.raises(hot.HotStorageError, match="materialized embedding link"):
        hot.verify_preserved_attempt(local, original_attempt=attempt["canonical"])
