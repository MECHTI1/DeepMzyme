"""Restore one terminated CLEAN hot attempt before normal campaign persistence.

Callers own worker termination, capacity admission, HostPull acknowledgment and
scientific result verification. This helper never executes a model or repairs a
pending attempt automatically.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
from typing import Any, Mapping
import uuid

MARKER = "hot_storage.json"
RESTORE_MANIFEST = "hot_restore.json"
RESULT = "unit_result.json"
FORMAT = "deepmzyme_ec_clean30_hot_storage_v1"
RESTORE_FORMAT = "deepmzyme_ec_clean30_hot_restore_v1"
FARM_MANIFEST = "embedding_farm_manifest.tsv"
FARM_COLUMNS = ("label", "seed", "sequence_sha256", "embedding", "layer", "dim", "file_sha256")


class HotStorageError(RuntimeError):
    """The hot attempt cannot safely be restored or removed."""


def _require(value: Any, message: str) -> None:
    if not value:
        raise HotStorageError(message)


def _bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


def _sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _safe_path(value: str | Path) -> Path:
    _require(isinstance(value, (str, Path)), "An explicit path is required.")
    path = Path(value)
    _require(path.is_absolute() and ".." not in path.parts and path == Path(os.path.abspath(path)),
             "Hot storage paths must be canonical and absolute.")
    for parent in (*reversed(path.parents), path):
        try:
            mode = parent.lstat().st_mode
        except FileNotFoundError:
            continue
        _require(not stat.S_ISLNK(mode), f"Symlink ancestor is forbidden: {parent}")
        _require(parent == path or stat.S_ISDIR(mode), f"Non-directory ancestor: {parent}")
    return path


def _directory(path: Path, *, private: bool, create: bool = False) -> None:
    _safe_path(path)
    if create:
        path.mkdir(parents=True, mode=0o700, exist_ok=True)
    info = path.lstat()
    _require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid(), "Attempt directories must be owned directories.")
    _require(not private or stat.S_IMODE(info.st_mode) == 0o700,
             "The hot storage root and managed directories need genuine private POSIX mode 0700.")


def _regular(path: Path) -> os.stat_result:
    _safe_path(path)
    info = path.lstat()
    _require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid(), f"Expected an owned regular file: {path}")
    return info


def _atomic(path: Path, content: bytes, *, replace: bool = False) -> None:
    _safe_path(path)
    if path.exists():
        _regular(path)
        if not replace:
            _require(path.read_bytes() == content, f"Existing immutable bytes differ: {path}")
            return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _identity(hot_root, canonical_attempt, *, session_id, unit_id, attempt,
              execution_manifest_sha256, g3_sha256) -> dict[str, Any]:
    hot_root, canonical = _safe_path(hot_root), _safe_path(canonical_attempt)
    _require(all(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value)
                 for value in (session_id, unit_id)), "Unsafe session or unit identifier.")
    _require(type(attempt) is int and attempt > 0, "Attempt must be a positive integer.")
    _require(all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
                 for value in (execution_manifest_sha256, g3_sha256)), "Source and G3 SHA-256 bindings are required.")
    _require(canonical.name == f"attempt{attempt}" and canonical.parent.name == unit_id
             and canonical.parent.parent.name == "runs", "Canonical attempt does not match its unit and attempt identity.")
    _require(hot_root != canonical and hot_root not in canonical.parents and canonical not in hot_root.parents,
             "Hot and canonical roots must not overlap.")
    hot = hot_root / session_id / unit_id / f"attempt{attempt}"
    _safe_path(hot)
    return {"hot_root": str(hot_root), "hot_attempt": str(hot), "canonical_attempt": str(canonical),
            "session_id": session_id, "unit_id": unit_id, "attempt": attempt,
            "execution_manifest_sha256": execution_manifest_sha256, "g3_sha256": g3_sha256}


def _marker(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {"format": FORMAT, "identity": dict(identity), "identity_sha256": _digest(identity), "status": "pending"}


def prepare_hot_attempt(hot_root: str | Path, canonical_attempt: str | Path, *, session_id: str, unit_id: str,
                        attempt: int, execution_manifest_sha256: str, g3_sha256: str) -> dict[str, Any]:
    """Create a fresh deterministic attempt and pending marker before starting its worker."""
    identity = _identity(hot_root, canonical_attempt, session_id=session_id, unit_id=unit_id, attempt=attempt,
                         execution_manifest_sha256=execution_manifest_sha256, g3_sha256=g3_sha256)
    hot, canonical = Path(identity["hot_attempt"]), Path(identity["canonical_attempt"])
    _require(not hot.exists() and not canonical.exists(), "An existing hot or canonical attempt requires explicit recovery.")
    for path in (Path(identity["hot_root"]), hot.parent.parent, hot.parent, hot):
        _directory(path, private=True, create=True)
    _directory(canonical, private=False, create=True)
    marker = _marker(identity)
    _atomic(canonical / MARKER, _bytes(marker))
    _atomic(hot / MARKER, _bytes(marker))
    return {"marker_path": str(canonical / MARKER), "hot_attempt": str(hot), "canonical_attempt": str(canonical),
            "identity_sha256": marker["identity_sha256"]}


def _load_marker(marker_path: str | Path) -> tuple[Path, dict[str, Any]]:
    path = _safe_path(marker_path)
    _regular(path)
    marker = json.loads(path.read_text())
    _require(isinstance(marker, dict) and marker.get("format") == FORMAT
             and marker.get("status") in {"pending", "restored"}, "Invalid hot-storage marker.")
    identity = marker.get("identity", {})
    _require(isinstance(identity, dict), "Invalid hot-attempt identity.")
    expected = _identity(identity.get("hot_root"), identity.get("canonical_attempt"),
        session_id=identity.get("session_id"), unit_id=identity.get("unit_id"), attempt=identity.get("attempt"),
        execution_manifest_sha256=identity.get("execution_manifest_sha256"), g3_sha256=identity.get("g3_sha256"))
    _require(identity == expected and marker.get("identity_sha256") == _digest(identity)
             and path == Path(identity["canonical_attempt"]) / MARKER, "Hot-storage marker identity changed.")
    keys = {"format", "identity", "identity_sha256", "status"}
    _require(set(marker) == (keys if marker["status"] == "pending" else keys | {"restore_manifest_sha256"}),
             "Hot-storage marker contains an incomplete transition.")
    return path, marker


def _fingerprint(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def worker_hot_attempt(marker_path: str | Path, *, session_id: str, unit_id: str, attempt: int,
                       execution_manifest_sha256: str, g3_sha256: str) -> Path:
    """Give the worker only its parent's pending, exactly bound, already-created directory."""
    _, marker = _load_marker(marker_path)
    expected = {"session_id": session_id, "unit_id": unit_id, "attempt": attempt,
                "execution_manifest_sha256": execution_manifest_sha256, "g3_sha256": g3_sha256}
    _require(type(attempt) is int and attempt > 0 and marker["status"] == "pending"
             and all(marker["identity"].get(key) == value for key, value in expected.items()),
             "Worker identity differs from its pending hot-attempt allocation.")
    hot = Path(marker["identity"]["hot_attempt"])
    _directory(hot, private=True)
    _regular(hot / MARKER)
    _require((hot / MARKER).read_bytes() == _bytes(_marker(marker["identity"])), "Private hot-attempt marker changed.")
    return hot


def _farm_target(link: Path, hot: Path) -> Path:
    _require(link.parent == hot / "work/data/esm_data", "Only CLEAN's internal embedding-file links may be materialized.")
    target = _safe_path(Path(os.path.abspath(link.parent / os.readlink(link))))
    _require(hot in target.parents and target.parent in {hot / "farms/base", hot / "farms/masked"}
             and target.name == link.name and target.suffix == ".pt", "Embedding link escapes its exact hot farm.")
    _regular(target)
    manifest = target.parent / FARM_MANIFEST
    _regular(manifest)
    rows = {}
    with manifest.open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        _require(tuple(next(reader, ())) == FARM_COLUMNS, "The internal embedding farm manifest has an invalid header.")
        for row in reader:
            _require(len(row) == len(FARM_COLUMNS) and row[0] not in rows
                     and re.fullmatch(r"[A-Za-z0-9_.-]+", row[0])
                     and re.fullmatch(r"[0-9a-f]{64}", row[-1]), "Malformed or duplicate internal farm entry.")
            rows[row[0]] = row[-1]
    _require(rows.get(target.stem) == _sha(target), "The internal embedding link target differs from its farm manifest.")
    return target


def _snapshot(root: Path, *, materialize_links: bool) -> tuple[list[dict[str, Any]], dict[str, tuple[Path, tuple[int, ...]]]]:
    _directory(root, private=materialize_links)
    records, sources = [], {}
    for parent, directories, files in os.walk(root, followlinks=False):
        directory = Path(parent)
        _safe_path(directory)
        for name in directories:
            _directory(directory / name, private=False)
        for name in files:
            path = directory / name
            relative = path.relative_to(root).as_posix()
            if relative in {MARKER, RESTORE_MANIFEST}:
                continue
            info = path.lstat()
            source, target = path, None
            if stat.S_ISLNK(info.st_mode):
                _require(materialize_links, "The canonical attempt cannot contain links.")
                source = _farm_target(path, root)
                target = source.relative_to(root).as_posix()
            else:
                _regular(path)
            source_info = _regular(source)
            record = {"path": relative, "size_bytes": source_info.st_size, "sha256": _sha(source)}
            if target is not None:
                record["materialized_from"] = target
            records.append(record)
            sources[relative] = (source, _fingerprint(source_info))
    return sorted(records, key=lambda row: row["path"]), sources


def _copy_regular(source: Path, destination: Path, expected: Mapping[str, Any], fingerprint: tuple[int, ...]) -> None:
    _safe_path(destination)
    _directory(destination.parent, private=False, create=True)
    if destination.exists():
        _regular(destination)
        _require(destination.stat().st_size == expected["size_bytes"] and _sha(destination) == expected["sha256"],
                 f"Existing restored bytes differ: {destination}")
        return
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.partial")
    try:
        source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(source_fd, "rb") as incoming:
            before = os.fstat(incoming.fileno())
            _require(stat.S_ISREG(before.st_mode) and _fingerprint(before) == fingerprint,
                     "The terminated hot attempt changed before copying.")
            target_fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(target_fd, "wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
                outgoing.flush()
                os.fsync(outgoing.fileno())
                os.fchmod(outgoing.fileno(), stat.S_IMODE(before.st_mode) & 0o777)
            _require(_fingerprint(os.fstat(incoming.fileno())) == fingerprint, "The hot file changed while copying.")
        _require(temporary.stat().st_size == expected["size_bytes"] and _sha(temporary) == expected["sha256"],
                 "Restored file hash differs from the terminated hot attempt.")
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_canonical(canonical: Path, receipt: Mapping[str, Any]) -> None:
    actual, _ = _snapshot(canonical, materialize_links=False)
    expected = [{key: value for key, value in row.items() if key != "materialized_from"} for row in receipt["files"]]
    _require(actual == expected, "Canonical attempt no longer matches the complete restored inventory.")


def _restoration(path: Path, marker: Mapping[str, Any]) -> dict[str, Any]:
    receipt_path = path.parent / RESTORE_MANIFEST
    _regular(receipt_path)
    _require(marker["status"] == "restored" and _sha(receipt_path) == marker.get("restore_manifest_sha256"),
             "Hot restoration is incomplete or its receipt changed.")
    receipt = json.loads(receipt_path.read_text())
    _require(receipt.get("format") == RESTORE_FORMAT and receipt.get("identity_sha256") == marker["identity_sha256"],
             "Restoration receipt belongs to another hot attempt.")
    return receipt


def restore_hot_attempt(marker_path: str | Path, *, worker_terminated: bool) -> dict[str, Any]:
    """Copy only after the parent positively confirms termination; failed copies stay pending."""
    _require(worker_terminated is True, "The parent must confirm worker termination before restoring hot files.")
    path, marker = _load_marker(marker_path)
    canonical, hot = path.parent, Path(marker["identity"]["hot_attempt"])
    if marker["status"] == "restored":
        receipt = _restoration(path, marker)
        _verify_canonical(canonical, receipt)
        return receipt
    _regular(hot / MARKER)
    _require((hot / MARKER).read_bytes() == _bytes(_marker(marker["identity"])), "The private hot-attempt identity changed.")
    _require(not (hot / RESTORE_MANIFEST).exists() and not (hot / RESTORE_MANIFEST).is_symlink(),
             "A worker cannot supply the parent restoration receipt.")
    records, sources = _snapshot(hot, materialize_links=True)
    names = {row["path"]: row for row in records}
    successful = False
    if RESULT in names:
        try:
            result = json.loads((hot / RESULT).read_text())
        except (ValueError, UnicodeError):
            result = {}
        successful = isinstance(result, dict) and result.get("status") == "passed"
        if successful:
            export = _safe_path(result.get("export"))
            _require(result.get("kind") == "clean" and result.get("unit_id") == marker["identity"]["unit_id"]
                     and canonical in export.parents, "Successful hot result must already name its canonical CLEAN export.")
            relative = export.relative_to(canonical).as_posix()
            _require(relative in names and names[relative]["sha256"] == result.get("export_sha256"),
                     "Canonical export binding differs from the hot result.")
    existing, _ = _snapshot(canonical, materialize_links=False)
    _require(all(row["path"] in names and row["size_bytes"] == names[row["path"]]["size_bytes"]
                 and row["sha256"] == names[row["path"]]["sha256"] for row in existing),
             "Canonical attempt contains foreign or changed partial output.")
    receipt = {"format": RESTORE_FORMAT, "identity_sha256": marker["identity_sha256"], "files": records,
               "total_bytes": sum(row["size_bytes"] for row in records), "successful_result": successful,
               "worker_terminated": True}
    for row in records:
        if row["path"] != RESULT:
            source, fingerprint = sources[row["path"]]
            _copy_regular(source, canonical / row["path"], row, fingerprint)
    after, after_sources = _snapshot(hot, materialize_links=True)
    _require(after == records and after_sources == sources, "The hot attempt mutated during restoration.")
    _atomic(canonical / RESTORE_MANIFEST, _bytes(receipt))
    if RESULT in names:
        source, fingerprint = sources[RESULT]
        _copy_regular(source, canonical / RESULT, names[RESULT], fingerprint)
    _verify_canonical(canonical, receipt)
    _require(_load_marker(path)[1] == marker, "The canonical identity marker changed during restoration.")
    _atomic(path, _bytes({**marker, "status": "restored", "restore_manifest_sha256": _sha(canonical / RESTORE_MANIFEST)}),
            replace=True)
    return receipt


def require_restored_attempts(execution_root: str | Path) -> None:
    """Block campaign recovery/preservation before it can persist a pending marker alone."""
    root = _safe_path(execution_root)
    runs = root / "runs"
    if not runs.exists():
        _require(not runs.is_symlink(), "The execution runs directory cannot be a symlink.")
        return
    _directory(runs, private=False)
    for unit in runs.iterdir():
        if not unit.is_dir():
            _require(not unit.is_symlink(), "Execution unit paths cannot be symlinks.")
            continue
        _directory(unit, private=False)
        for attempt in unit.iterdir():
            if not re.fullmatch(r"attempt[1-9][0-9]*", attempt.name):
                continue
            _directory(attempt, private=False)
            marker_path = attempt / MARKER
            if marker_path.exists() or marker_path.is_symlink():
                path, marker = _load_marker(marker_path)
                _require(marker["status"] == "restored", "Pending hot data blocks automatic recovery and provider deletion.")
                _restoration(path, marker)


def verify_preserved_attempt(local_attempt: str | Path, *, original_attempt: str | Path) -> dict[str, Any]:
    """Verify a relocated complete copy without resolving or accessing any recorded remote path."""
    local = _safe_path(local_attempt)
    _directory(local, private=False)

    def remote_path(value):
        _require(isinstance(value, str) and value and not any(char in value for char in "\0\r\n"),
                 "Recorded remote paths must be explicit POSIX paths.")
        path = PurePosixPath(value)
        _require(path.is_absolute() and ".." not in path.parts and str(path) == value,
                 "Recorded remote paths must be canonical absolute POSIX paths.")
        return path

    original = remote_path(str(original_attempt))
    marker_path, receipt_path = local / MARKER, local / RESTORE_MANIFEST
    _regular(marker_path)
    marker = json.loads(marker_path.read_text())
    _require(isinstance(marker, dict) and set(marker) ==
             {"format", "identity", "identity_sha256", "status", "restore_manifest_sha256"}
             and marker.get("format") == FORMAT and marker.get("status") == "restored",
             "Preserved hot data is pending or has an invalid restored marker.")
    identity = marker["identity"]
    _require(isinstance(identity, dict) and set(identity) ==
             {"hot_root", "hot_attempt", "canonical_attempt", "session_id", "unit_id", "attempt",
              "execution_manifest_sha256", "g3_sha256"}, "Preserved hot identity has an invalid schema.")
    _require(all(isinstance(identity[key], str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", identity[key])
                 for key in ("session_id", "unit_id"))
             and type(identity["attempt"]) is int and identity["attempt"] > 0,
             "Preserved hot identity has an invalid session, unit or attempt.")
    _require(all(isinstance(identity[key], str) and re.fullmatch(r"[0-9a-f]{64}", identity[key])
                 for key in ("execution_manifest_sha256", "g3_sha256"))
             and marker["identity_sha256"] == _digest(identity), "Preserved hot source, G3 or identity digest differs.")
    canonical = remote_path(identity["canonical_attempt"])
    hot_root, hot_attempt = remote_path(identity["hot_root"]), remote_path(identity["hot_attempt"])
    _require(canonical == original and canonical.name == f"attempt{identity['attempt']}"
             and canonical.parent.name == identity["unit_id"] and canonical.parent.parent.name == "runs"
             and hot_attempt == hot_root / identity["session_id"] / identity["unit_id"] / canonical.name
             and hot_root != canonical and hot_root not in canonical.parents and canonical not in hot_root.parents,
             "Preserved hot paths differ from the original attempt or deterministic placement.")
    _regular(receipt_path)
    _require(isinstance(marker["restore_manifest_sha256"], str)
             and re.fullmatch(r"[0-9a-f]{64}", marker["restore_manifest_sha256"])
             and _sha(receipt_path) == marker["restore_manifest_sha256"], "Preserved restoration manifest checksum differs.")
    receipt = json.loads(receipt_path.read_text())
    _require(isinstance(receipt, dict) and set(receipt) ==
             {"format", "identity_sha256", "files", "total_bytes", "successful_result", "worker_terminated"}
             and receipt.get("format") == RESTORE_FORMAT and receipt.get("identity_sha256") == marker["identity_sha256"]
             and receipt.get("worker_terminated") is True and type(receipt.get("successful_result")) is bool,
             "Preserved restoration receipt has an invalid identity or completion state.")
    records = receipt["files"]
    _require(isinstance(records, list), "Preserved restoration inventory is missing.")
    members = {}
    for row in records:
        _require(isinstance(row, dict) and set(row) in
                 ({"path", "size_bytes", "sha256"}, {"path", "size_bytes", "sha256", "materialized_from"}),
                 "Invalid preserved restoration member.")
        name = row["path"]
        relative = PurePosixPath(name) if isinstance(name, str) else PurePosixPath("/")
        _require(isinstance(name, str) and name not in {"", ".", MARKER, RESTORE_MANIFEST}
                 and not relative.is_absolute() and ".." not in relative.parts and str(relative) == name
                 and name not in members and "\\" not in name,
                 "Unsafe or duplicate preserved restoration member.")
        _require(type(row["size_bytes"]) is int and row["size_bytes"] >= 0
                 and isinstance(row["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]),
                 "Invalid preserved restoration size or checksum.")
        members[name] = row
    _require([row["path"] for row in records] == sorted(members)
             and type(receipt["total_bytes"]) is int
             and receipt["total_bytes"] == sum(row["size_bytes"] for row in records),
             "Preserved restoration inventory order or total bytes differs.")
    _verify_canonical(local, receipt)
    for name, row in members.items():
        if "materialized_from" not in row:
            continue
        target = row["materialized_from"]
        source_path = PurePosixPath(target) if isinstance(target, str) else PurePosixPath("/")
        _require(PurePosixPath(name).parent == PurePosixPath("work/data/esm_data")
                 and source_path.parent in {PurePosixPath("farms/base"), PurePosixPath("farms/masked")}
                 and source_path.name == PurePosixPath(name).name and source_path.suffix == ".pt"
                 and target in members and members[target]["sha256"] == row["sha256"]
                 and members[target]["size_bytes"] == row["size_bytes"],
                 "Preserved materialized embedding link differs from its internal farm file.")
        manifest = local / str(source_path.parent) / FARM_MANIFEST
        _regular(manifest)
        with manifest.open(newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            _require(tuple(next(reader, ())) == FARM_COLUMNS, "Preserved farm manifest header differs.")
            farm = {}
            for entry in reader:
                _require(len(entry) == len(FARM_COLUMNS) and entry[0] not in farm,
                         "Preserved farm manifest has malformed or duplicate entries.")
                farm[entry[0]] = entry[-1]
        _require(farm.get(source_path.stem) == row["sha256"], "Preserved materialized file differs from its farm manifest.")
    result = {}
    if RESULT in members:
        try:
            result = json.loads((local / RESULT).read_text())
        except (ValueError, UnicodeError):
            pass
    successful = isinstance(result, dict) and result.get("status") == "passed"
    _require(successful == receipt["successful_result"], "Preserved result status differs from its restoration receipt.")
    if successful:
        export = remote_path(result.get("export"))
        _require(result.get("kind") == "clean" and result.get("unit_id") == identity["unit_id"]
                 and canonical in export.parents, "Preserved result names another original CLEAN export.")
        relative = export.relative_to(canonical).as_posix()
        _require(relative in members and members[relative]["sha256"] == result.get("export_sha256"),
                 "Preserved export checksum differs from its terminal result.")
    return {"status": "verified", "identity_sha256": marker["identity_sha256"],
            "original_attempt": str(original), "file_count": len(records), "total_bytes": receipt["total_bytes"],
            "successful_result": successful, "execution_manifest_sha256": identity["execution_manifest_sha256"],
            "g3_sha256": identity["g3_sha256"]}


def cleanup_hot_attempt(marker_path: str | Path, *, host_ack_verified: bool, result_verified: bool) -> dict[str, Any]:
    """Remove only a successful, independently acknowledged, still-identical hot attempt."""
    _require(host_ack_verified is True and result_verified is True,
             "Hot cleanup requires verified HostPull acknowledgment and verified unit result.")
    path, marker = _load_marker(marker_path)
    receipt = _restoration(path, marker)
    _require(receipt.get("successful_result") is True, "Failure attempts are preserved; hot cleanup is forbidden.")
    _verify_canonical(path.parent, receipt)
    hot = _safe_path(marker["identity"]["hot_attempt"])
    if not hot.exists():
        return {"status": "already_absent", "identity_sha256": marker["identity_sha256"]}
    _require((hot / MARKER).read_bytes() == _bytes(_marker(marker["identity"])), "Hot cleanup identity differs.")
    records, _ = _snapshot(hot, materialize_links=True)
    _require(records == receipt["files"], "Hot files changed after restoration; cleanup is forbidden.")
    _require(shutil.rmtree.avoids_symlink_attacks, "This platform cannot safely remove the verified hot tree.")
    shutil.rmtree(hot)
    return {"status": "removed", "identity_sha256": marker["identity_sha256"], "bytes_preserved": receipt["total_bytes"]}
