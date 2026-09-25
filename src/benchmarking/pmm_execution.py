"""Small admission and persistence hooks for the existing PMM campaign runner.

This module never provisions, stops, or restarts compute. The provider/controller
owns billing and the hard stop. A worker receipt records observations, not proof
that a VM stopped. Completed scientific fits are reused by pmm_ion_campaign;
interrupted fits restart from their original seed, not an optimizer checkpoint.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import signal
import shutil
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ExecutionBlocked(RuntimeError):
    """Ownership, time allowance, or a preceding result blocks a launch."""


class PersistenceError(RuntimeError):
    """A terminal unit has not reached verified independent storage."""


@dataclass(frozen=True)
class ExecutionPolicy:
    deadline_unix: float
    max_total_seconds: float
    reserve_seconds: float = 900.0
    forecast_margin: float = 1.25
    max_consecutive_failures: int = 1
    allocation_started_unix: float | None = None

    def __post_init__(self) -> None:
        for name in ("deadline_unix", "max_total_seconds", "reserve_seconds", "forecast_margin"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.forecast_margin < 1 or self.max_consecutive_failures < 1:
            raise ValueError("forecast_margin and max_consecutive_failures must be at least one")
        if self.allocation_started_unix is not None and not math.isfinite(self.allocation_started_unix):
            raise ValueError("allocation_started_unix must be finite")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _process_identity(pid: int) -> dict[str, Any]:
    """Linux process identity protects recovery from PID reuse across boots."""
    stat = Path(f"/proc/{pid}/stat").read_text()
    fields = stat[stat.rfind(")") + 2:].split()
    if fields[0] in {"Z", "X"}:
        raise FileNotFoundError(f"Process {pid} has exited")
    return {"pid": pid, "start_ticks": fields[19],
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "hostname": socket.gethostname()}


def verify_host_pull(manifest_path: Path, downloaded_root: Path, ack_path: Path) -> dict[str, Any]:
    """Run on the host after downloading the manifest and every listed artifact.

    Only this independent local readback produces the acknowledgment to upload
    to the worker. The acknowledgment grants no provider start authorization.
    """
    manifest_path, downloaded_root = Path(manifest_path).resolve(), Path(downloaded_root).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("persistence_mode") != "host_pull" or manifest.get("schema_version") != 1:
        raise PersistenceError("Not a supported host-pull manifest")
    if str(downloaded_root) != manifest.get("destination_root"):
        raise PersistenceError("Host destination differs from the worker's recorded destination")
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        path = downloaded_root / relative
        if relative.is_absolute() or downloaded_root not in path.resolve().parents or path.is_symlink():
            raise PersistenceError(f"Unsafe host-pull member: {relative}")
        if not path.is_file() or path.stat().st_size != entry["size_bytes"] or _sha256(path) != entry["sha256"]:
            raise PersistenceError(f"Host artifact is missing or has a different checksum: {relative}")
    acknowledgement = {"schema_version": 1, "manifest_sha256": _sha256(manifest_path),
                       "verified_root": str(downloaded_root), "file_count": len(manifest["files"]),
                       "verified_unix": time.time(), "verified_by": socket.gethostname()}
    _atomic_json(Path(ack_path), acknowledgement)
    return acknowledgement


class CampaignExecution:
    """One worker owns a campaign directory and persists each terminal unit.

    ``max_total_seconds`` limits this allocation, measured from its real start
    when supplied; otherwise it limits runner time only. An absolute provider
    deadline is always required. Neither limit replaces the provider hard stop.
    In ``mounted`` mode, ``durable_root`` must be a verified independent mount;
    a different pathname alone does not establish independent storage. In
    ``host_pull`` mode it names the host's destination; the worker writes only a
    manifest, then blocks until independent host readback is acknowledged.
    """

    def __init__(self, campaign_root: Path, *, policy: ExecutionPolicy,
                 session_id: str, durable_root: Path, persistence_mode: str = "mounted"):
        if persistence_mode not in {"mounted", "host_pull"}:
            raise ValueError("persistence_mode must be mounted or host_pull")
        self.persistence_mode = persistence_mode
        self.root = Path(campaign_root).resolve()
        self.durable_root = Path(durable_root).resolve()
        if (self.root == self.durable_root or self.root in self.durable_root.parents
                or self.durable_root in self.root.parents):
            raise ValueError("Campaign and durable roots must be separate, non-nested directories")
        if not session_id.strip():
            raise ValueError("session_id must identify the owned allocation")
        self.policy, self.session_id = policy, session_id
        self.state_path = self.root / "execution_state.json"
        self.events_path = self.root / "execution_events.jsonl"
        self.state: dict[str, Any] = {}
        self._lock = None
        self._failures = 0

    def __enter__(self) -> "CampaignExecution":
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = (self.root / "execution.lock").open("a+")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock.close()
            self._lock = None
            raise ExecutionBlocked("Another worker owns this campaign directory") from exc
        try:
            self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {
                "schema_version": 1, "allocations": {}, "pending_unit": None,
            }
            self._accept_host_ack(required=True)
            child = self.state.get("active_child")
            if child:
                if child["hostname"] != socket.gethostname():
                    raise ExecutionBlocked("Previous child belongs to another host; reconcile that allocation first")
                try:
                    alive = _process_identity(int(child["pid"])) == child
                except FileNotFoundError:
                    alive = False
                if alive:
                    raise ExecutionBlocked(f"Previous training child PID {child['pid']} is still alive")
                self.state["active_child"] = None
            previous = self.state.get("pending_unit")
            if previous and not previous.get("persisted"):
                if previous["status"] == "running":
                    previous.update(status="interrupted", restart_semantics="restart original seed")
                self._event("recover_previous_terminal", run_name=previous["run_name"])
                self.persist(self._unit_paths(previous["run_name"]))
                if self.state.get("pending_transfer"):
                    raise ExecutionBlocked("Recovered attempt awaits verified host-pull acknowledgment")
            now = time.time()
            proposed_start = self.policy.allocation_started_unix
            if proposed_start is not None and proposed_start > now:
                raise ValueError("Allocation start cannot be in the future")
            allocation = self.state["allocations"].setdefault(self.session_id, {
                "started_unix": proposed_start if proposed_start is not None else now,
                "start_basis": "provider_allocation" if proposed_start is not None else "runner_only",
            })
            if proposed_start is not None and abs(allocation["started_unix"] - proposed_start) > 1:
                raise ExecutionBlocked("The same session ID has a different allocation-start receipt")
            allocation.update(deadline_unix=self.policy.deadline_unix,
                              max_total_seconds=self.policy.max_total_seconds,
                              provider_stop_verified=False)
            self.state.update(owner={"pid": os.getpid(), "hostname": socket.gethostname()},
                              active_session_id=self.session_id)
            self._event("worker_started")
            self._save()
            return self
        except BaseException:
            self._release()
            raise

    @property
    def should_stop(self) -> bool:
        return (self._failures >= self.policy.max_consecutive_failures
                or bool(self.state.get("pending_transfer")))

    def run_subprocess(self, command, *, check: bool = False, input=None,
                       timeout: float | None = None, **kwargs) -> subprocess.CompletedProcess:
        """Run an admitted child while retaining ownership if this parent dies.

        The inherited flock survives SIGKILL of the controller. Ordinary errors
        terminate and reap the child's process group before releasing ownership.
        This wraps subprocess execution only; it never changes provider state.
        """
        self._require_owner()
        pending = self.state.get("pending_unit")
        if not pending or pending["status"] != "running":
            raise ExecutionBlocked("Admit a running unit before starting its child")
        if self.state.get("active_child"):
            raise ExecutionBlocked("A child is already registered for this worker")
        if "start_new_session" in kwargs or "pass_fds" in kwargs:
            raise ValueError("The execution helper owns child process group and lock inheritance")
        allocation = self.state["allocations"][self.session_id]
        remaining = (min(self.policy.deadline_unix,
                         allocation["started_unix"] + self.policy.max_total_seconds)
                     - time.time() - self.policy.reserve_seconds)
        if remaining <= 0:
            raise ExecutionBlocked("Closeout reserve has begun; no child can start")
        if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("Child timeout must be finite and positive")
        timeout = min(remaining, timeout) if timeout is not None else remaining
        process = subprocess.Popen(command, pass_fds=(self._lock.fileno(),),
                                   start_new_session=True, **kwargs)
        try:
            try:
                child = _process_identity(process.pid)
            except FileNotFoundError:  # A very short child can finish before the /proc read.
                child = None
            self.state["active_child"] = child
            self._event("child_started", process=child, pid=process.pid)
            self._save()
            stdout, stderr = process.communicate(input=input, timeout=timeout)
            result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
            if check:
                result.check_returncode()
            return result
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise
        finally:
            self.state["active_child"] = None
            original = sys.exception()
            try:
                self._event("child_exited", pid=process.pid, return_code=process.returncode)
                self._save()
            except BaseException as save_error:
                if original is not None:
                    original.add_note(f"Child receipt could not be saved: {save_error}")
                else:
                    raise

    def admit(self, run_name: str, estimated_seconds: float) -> dict[str, Any]:
        self._require_owner()
        self._accept_host_ack(required=True)
        if not run_name or Path(run_name).name != run_name or run_name in {".", ".."}:
            raise ValueError("run_name must be one nonempty directory name")
        if not math.isfinite(estimated_seconds) or estimated_seconds <= 0:
            raise ValueError("A finite positive full-unit forecast is required")
        pending = self.state.get("pending_unit")
        if pending and not pending.get("persisted"):
            raise ExecutionBlocked("Persist and verify the preceding terminal unit before another launch")
        if self.should_stop:
            raise ExecutionBlocked("Execution stopped after a failed unit; diagnose before deliberate retry")
        allocation = self.state["allocations"][self.session_id]
        deadline = min(self.policy.deadline_unix,
                       allocation["started_unix"] + self.policy.max_total_seconds)
        required = estimated_seconds * self.policy.forecast_margin + self.policy.reserve_seconds
        remaining = deadline - time.time()
        if remaining < required:
            self._event("admission_refused", run_name=run_name,
                        required_seconds=required, remaining_seconds=remaining)
            self._save()
            raise ExecutionBlocked(f"{run_name} needs {required:.0f}s including margin/closeout; {remaining:.0f}s remain")
        pending = {"run_name": run_name, "status": "running", "persisted": False,
                   "session_id": self.session_id, "started_unix": time.time(),
                   "estimated_seconds": estimated_seconds, "required_seconds": required}
        self.state["pending_unit"] = pending
        self._event("unit_admitted", **pending)
        self._save()
        return dict(pending)

    def record_result(self, run_name: str, status: str, elapsed_seconds: float) -> None:
        self._require_owner()
        pending = self.state.get("pending_unit")
        if not pending or pending["run_name"] != run_name or pending["status"] != "running":
            raise ExecutionBlocked("Result does not match the admitted running unit")
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be finite and nonnegative")
        if status not in {"completed", "reused_verified", "interrupted"} and not status.startswith("failed"):
            raise ValueError(f"Unsupported terminal status {status!r}")
        pending.update(status=status, elapsed_seconds=elapsed_seconds, finished_unix=time.time())
        if status in {"completed", "reused_verified"}:
            self._failures = 0
        else:
            self._failures += 1
            pending["restart_semantics"] = "restart original seed"
        self._event("unit_terminal", **pending)
        self._save()

    def persist(self, paths: list[Path]) -> dict[str, Any]:
        """Atomically copy terminal artifacts, then independently hash their readback.

        Pass this unit's run directory, log, command and queue status, not the
        complete campaign/embedding directory. State and events are included.
        A manifest is published last; failure blocks further admission.
        """
        self._require_owner()
        pending = self.state.get("pending_unit")
        if pending and pending["status"] == "running":
            raise PersistenceError("A running unit is not a terminal result")
        if self.state.get("pending_transfer"):
            verified = self._accept_host_ack(required=False)
            return {"status": "verified_host" if verified else "awaiting_host_ack",
                    "transfer": self.state.get("pending_transfer") or self.state.get("last_host_transfer")}
        files: set[Path] = set()
        for item in paths:
            item = Path(item).resolve()
            if item == self.root or self.root not in item.parents:
                raise ValueError("Persist only unit artifacts located inside the campaign root")
            if not item.exists():
                raise PersistenceError(f"Required artifact is missing: {item}")
            files.update(item.rglob("*") if item.is_dir() else [item])
        try:
            if self.persistence_mode == "host_pull":
                if not files and (pending is None or pending.get("persisted")):
                    return {"status": "no_new_artifacts"}
                return self._request_host_pull(files)
            copied = [self._copy_verified(path) for path in sorted(files) if path.is_file()]
            receipt_name = f"persistence_receipts/{uuid.uuid4().hex}.json"
            if pending and (not pending.get("persisted") or files):
                pending.update(persisted=True, persistence_receipt=receipt_name)
            self._event("artifacts_verified", receipt=receipt_name, files=len(copied))
            self._save()
            copied += [self._copy_verified(self.state_path), self._copy_verified(self.events_path)]
            receipt = {"schema_version": 1, "session_id": self.session_id,
                       "verified_unix": time.time(), "durable_root": str(self.durable_root),
                       "run_name": pending["run_name"] if pending else None,
                       "files": copied, "provider_stop_verified": False}
            receipt_path = self.root / receipt_name
            _atomic_json(receipt_path, receipt)
            self._copy_verified(receipt_path)
            return receipt
        except BaseException as exc:
            if pending:
                pending["persisted"] = False
            if self.state.get("pending_transfer") and not self.state["pending_transfer"].get("manifest_sha256"):
                self.state["pending_transfer"] = None
            self._failures = self.policy.max_consecutive_failures
            self.state["persistence_error"] = str(exc)
            self._save()
            raise PersistenceError(f"Durable copy/readback failed; no further units admitted: {exc}") from exc

    def _request_host_pull(self, files: set[Path]) -> dict[str, Any]:
        token = uuid.uuid4().hex
        relative = f"persistence_receipts/{token}.json"
        transfer = {"manifest_path": relative, "ack_path": f"persistence_receipts/{token}.ack.json",
                    "destination_root": str(self.durable_root)}
        self.state["pending_transfer"] = transfer
        self._event("host_pull_requested", manifest=relative)
        # Immutable snapshots avoid changing the advertised hashes at worker exit.
        state_snapshot = self.root / f"persistence_receipts/{token}.state.json"
        events_snapshot = self.root / f"persistence_receipts/{token}.events.jsonl"
        _atomic_json(state_snapshot, self.state)
        with self.events_path.open("rb") as source, events_snapshot.open("wb") as target:
            shutil.copyfileobj(source, target)
            target.flush()
            os.fsync(target.fileno())
        files = (files - {self.state_path, self.events_path}) | {state_snapshot, events_snapshot}
        entries = []
        for path in sorted(files):
            if not path.is_file():
                continue
            if path.is_symlink() or self.root not in path.resolve().parents:
                raise PersistenceError(f"Unsafe transfer member: {path}")
            entries.append({"path": str(path.relative_to(self.root)), "sha256": _sha256(path),
                            "size_bytes": path.stat().st_size})
        manifest = {"schema_version": 1, "persistence_mode": "host_pull",
                    "session_id": self.session_id, "created_unix": time.time(),
                    "destination_root": str(self.durable_root), "ack_path": transfer["ack_path"],
                    "files": entries, "provider_stop_verified": False}
        _atomic_json(self.root / relative, manifest)
        transfer.update(manifest_sha256=_sha256(self.root / relative), file_count=len(entries))
        self._save()
        return {"status": "awaiting_host_ack", "transfer": dict(transfer)}

    def _accept_host_ack(self, *, required: bool) -> bool:
        transfer = self.state.get("pending_transfer")
        if not transfer:
            return False
        ack_path = self.root / transfer["ack_path"]
        if not ack_path.is_file():
            if required:
                raise ExecutionBlocked(f"Host must download/verify {transfer['manifest_path']} and upload {transfer['ack_path']}")
            return False
        ack = json.loads(ack_path.read_text())
        if (_sha256(self.root / transfer["manifest_path"]) != transfer["manifest_sha256"]
                or ack.get("manifest_sha256") != transfer["manifest_sha256"]
                or ack.get("verified_root") != str(self.durable_root)
                or ack.get("verified_root") != transfer["destination_root"]
                or ack.get("file_count") != transfer["file_count"]):
            raise PersistenceError("Host acknowledgment does not match the immutable transfer manifest/destination")
        pending = self.state.get("pending_unit")
        if pending:
            pending.update(persisted=True, persistence_receipt=transfer["manifest_path"],
                           host_acknowledgment=transfer["ack_path"])
        self.state["last_host_transfer"] = {**transfer, "ack_sha256": _sha256(ack_path)}
        self.state["pending_transfer"] = None
        self._event("host_pull_verified", manifest=transfer["manifest_path"], ack_sha256=_sha256(ack_path))
        self._save()
        return True

    def _copy_verified(self, path: Path) -> dict[str, Any]:
        resolved = path.resolve()
        if self.root not in resolved.parents or path.is_symlink():
            raise PersistenceError(f"Artifact escapes campaign or is a symlink: {path}")
        relative = path.relative_to(self.root)
        target = self.durable_root / relative
        if self.durable_root not in target.resolve().parents:
            raise PersistenceError(f"Durable destination escapes its root: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        before = _sha256(path)
        temporary = target.with_name(target.name + ".partial")
        with path.open("rb") as source, temporary.open("wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(target)
        if _sha256(target) != before or _sha256(path) != before:
            raise PersistenceError(f"Checksum mismatch or concurrent write: {relative}")
        return {"path": str(relative), "sha256": before, "size_bytes": target.stat().st_size}

    def _unit_paths(self, run_name: str) -> list[Path]:
        return [path for path in (self.root / "runs" / run_name,
                                  self.root / "runs" / f"{run_name}.log",
                                  self.root / "commands" / f"{run_name}.json") if path.exists()]

    def _require_owner(self) -> None:
        if self._lock is None:
            raise ExecutionBlocked("Enter the CampaignExecution context before using it")

    def _event(self, event: str, **fields: Any) -> None:
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": event, "at_unix": time.time(),
                                     "session_id": self.session_id, **fields}, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _save(self) -> None:
        if self.session_id in self.state.get("allocations", {}):
            self.state["allocations"][self.session_id]["last_observed_unix"] = time.time()
        _atomic_json(self.state_path, self.state)

    def _release(self) -> None:
        if self._lock is not None:
            fcntl.flock(self._lock, fcntl.LOCK_UN)
            self._lock.close()
            self._lock = None

    def __exit__(self, exc_type, exc, traceback) -> bool:
        try:
            pending = self.state.get("pending_unit")
            if pending and pending["status"] == "running":
                self.record_result(pending["run_name"], "interrupted",
                                   max(0, time.time() - pending["started_unix"]))
            self.state["owner"] = None
            self._event("worker_exited", error=str(exc) if exc else None,
                        provider_stop_verified=False)
            self._save()
            self.persist(self._unit_paths(pending["run_name"]) if pending and not pending.get("persisted") else [])
        except BaseException as save_error:
            if exc is not None:
                exc.add_note(f"Execution closeout also failed: {save_error}")
            else:
                raise
        finally:
            self._release()
        return False
