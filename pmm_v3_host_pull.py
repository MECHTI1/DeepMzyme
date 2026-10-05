#!/usr/bin/env python3
"""Workstation host-pull for the v3 campaign: per-lane download, independent readback, acknowledgment.

A v3 lane in ``host_pull`` mode ends each unit by writing an immutable transfer manifest and exits; the
lane admits nothing new until this tool has, on the workstation, downloaded every listed file, verified
each SHA-256 from the downloaded bytes (``benchmarking.pmm_execution.verify_host_pull``) and uploaded the
small acknowledgment back to the lane. It never starts, stops or extends a VM and never runs fits.

  python3 pmm_v3_host_pull.py --remote-root /home/mechti/deepmzyme_runs/pmm_ion_metal_v3 \
      --durable-root /media/mechti/Data1/DeepMzyme_Data/campaigns/pmm_ion_metal_v3/durable --lanes 0 1 2

Per lane K the VM root is ``<remote-root>/lanes/laneK`` and the workstation root ``<durable-root>/laneK``,
which must equal (as a string) the destination the worker recorded from ``--durable-root``. Lanes are
pulled in parallel, each under its own lock. Re-running is safe: an acknowledged transfer is never
verified and stamped again (the speed report measures persistence from the first acknowledgment); a
missing remote copy of an existing acknowledgment is re-uploaded unchanged after a fresh readback.

Standard library only: ``pmm_execution.py`` is loaded by file path (torch is never imported).
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent
DEFAULT_SSH_CONFIG = Path.home() / "deepmzyme-vm" / "state" / "ssh_config"
CONDA_PYTHON = "/home/mechti/miniconda3/envs/DeepMzyme/bin/python"  # gcloud's IAP proxy (ssh ProxyCommand)


class HostPullError(RuntimeError):
    """The transfer cannot be acknowledged; the lane stays closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HostPullError(message)


def load_verifier(source_root: Path) -> Callable[..., dict[str, Any]]:
    """``verify_host_pull`` from ``<source_root>/src/benchmarking/pmm_execution.py`` (standard library only)."""
    path = Path(source_root) / "src" / "benchmarking" / "pmm_execution.py"
    spec = importlib.util.spec_from_file_location("pmm_execution_host_side", path)
    require(spec is not None and spec.loader is not None, f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses need the module registered before execution
    spec.loader.exec_module(module)
    return module.verify_host_pull


def relative_member(value: str) -> str:
    path = PurePosixPath(value)
    require(bool(value) and not path.is_absolute() and ".." not in path.parts and value != "."
            and not any(c in value for c in "\x00\r\n"), f"Unsafe transfer member: {value!r}")
    return str(path)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class SshTransport:
    """Remote file operations over the controller's IAP SSH configuration (BatchMode, no retries)."""

    def __init__(self, host: str, ssh_config: Path):
        self.host = host
        self.ssh = ["ssh", "-F", str(ssh_config), "-o", "BatchMode=yes"]
        self.scp = ["scp", "-F", str(ssh_config), "-o", "BatchMode=yes", "-q"]
        self.env = dict(os.environ, CLOUDSDK_PYTHON=CONDA_PYTHON, CLOUDSDK_PYTHON_SITEPACKAGES="1")

    def _run(self, command: list[str], **kwargs) -> subprocess.CompletedProcess:
        return subprocess.run(command, env=self.env, stdin=subprocess.DEVNULL, **kwargs)

    def read_bytes(self, path: str) -> bytes | None:
        quoted = shlex.quote(path)
        result = self._run(self.ssh + [self.host, f"if [ -f {quoted} ]; then cat -- {quoted}; else exit 3; fi"],
                           capture_output=True)
        if result.returncode == 3:
            return None
        require(result.returncode == 0, f"ssh read of {path} failed: {result.stderr.decode(errors='replace')}")
        return result.stdout

    def fetch(self, remote_root: str, members: list[str], local_root: Path) -> None:
        with tempfile.NamedTemporaryFile("wb", delete=False) as handle:
            handle.write(b"".join(name.encode() + b"\x00" for name in members))
            listing = handle.name
        try:
            result = self._run(["rsync", "-a", "--no-links", "--protect-args", "--from0", f"--files-from={listing}",
                                "-e", shlex.join(self.ssh), f"{self.host}:{remote_root}/", f"{local_root}/"],
                               capture_output=True)
        finally:
            os.unlink(listing)
        require(result.returncode == 0, f"rsync failed: {result.stderr.decode(errors='replace')}")

    def put(self, local: Path, remote: str) -> None:
        result = self._run(self.scp + [str(local), f"{self.host}:{remote}"], capture_output=True)
        require(result.returncode == 0, f"scp upload failed: {result.stderr.decode(errors='replace')}")

    def move(self, source: str, target: str) -> None:
        result = self._run(self.ssh + [self.host, f"mv -f -- {shlex.quote(source)} {shlex.quote(target)}"],
                           capture_output=True)
        require(result.returncode == 0, f"remote rename failed: {result.stderr.decode(errors='replace')}")


class LocalTransport:
    """The same operations on a local directory tree (tests and dry rehearsals)."""

    def read_bytes(self, path: str) -> bytes | None:
        return Path(path).read_bytes() if Path(path).is_file() else None

    def fetch(self, remote_root: str, members: list[str], local_root: Path) -> None:
        for name in members:
            source = Path(remote_root) / name
            require(source.is_file() and not source.is_symlink(), f"missing remote member {name}")
            target = Path(local_root) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    def put(self, local: Path, remote: str) -> None:
        shutil.copy2(local, remote)

    def move(self, source: str, target: str) -> None:
        os.replace(source, target)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def set_aside_superseded(local_lane: Path, entries: list[dict[str, Any]], token: str) -> list[str]:
    """Move earlier local copies that a new transfer would overwrite with different bytes (a rerun of the
    same run name after archive-failed) to ``superseded/<token>/``, whole run directories at a time, so the
    acknowledged first attempt stays intact and verifiable."""
    roots = set()
    for entry in entries:
        relative = PurePosixPath(entry["path"])
        target = local_lane / relative
        if target.is_file() and _file_sha256(target) != entry["sha256"]:
            roots.add(PurePosixPath(*relative.parts[:2]) if relative.parts[0] == "runs" and len(relative.parts) > 2
                      else relative)
    moved = []
    for root in sorted(roots, key=str):
        source, destination = local_lane / root, local_lane / "superseded" / token / root
        require(not destination.exists(), f"{destination} already exists")
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
        moved.append(str(root))
    # Files of the earlier attempt that the new transfer would not overwrite stay with their run directory.
    for entry in entries:
        relative = PurePosixPath(entry["path"])
        if relative.parts[0] == "runs" and len(relative.parts) > 2:
            continue
        target = local_lane / relative
        if target.is_file() and _file_sha256(target) != entry["sha256"]:
            raise HostPullError(f"could not set aside {relative}")
    return moved


def pull_lane(transport, remote_lane: str, local_lane: Path, verify: Callable[..., dict[str, Any]]) -> dict[str, Any]:
    """Download, verify and acknowledge the pending transfer of one lane (idempotent)."""
    remote = PurePosixPath(remote_lane)
    require(remote.is_absolute() and ".." not in remote.parts, "--remote-root must be an absolute VM path")
    local_lane = Path(local_lane)
    local_lane.mkdir(parents=True, exist_ok=True)
    require(str(local_lane.resolve()) == str(local_lane.absolute()),
            f"{local_lane} must be an absolute path without symlinks (the worker compares it as a string)")
    local_lane = local_lane.resolve()
    with (local_lane / ".host_pull.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raw = transport.read_bytes(str(remote / "execution_state.json"))
        if raw is None:
            return {"status": "no_lane_state", "remote": str(remote)}
        state = json.loads(raw)
        transfer = state.get("pending_transfer")
        if not transfer:
            return {"status": "no_pending_transfer", "remote": str(remote)}
        require((state.get("pending_unit") or {}).get("status") != "running", "Refusing to acknowledge a running unit")
        require(transfer.get("destination_root") == str(local_lane),
                f"Recorded destination {transfer.get('destination_root')!r} differs from {str(local_lane)!r}")
        manifest_rel, ack_rel = relative_member(transfer["manifest_path"]), relative_member(transfer["ack_path"])
        require(manifest_rel.startswith("persistence_receipts/") and ack_rel.startswith("persistence_receipts/"),
                "Manifest and acknowledgment must be inside persistence_receipts/")
        remote_ack, local_ack = str(remote / ack_rel), local_lane / ack_rel
        manifest_bytes = transport.read_bytes(str(remote / manifest_rel))
        require(manifest_bytes is not None and sha256_bytes(manifest_bytes) == transfer["manifest_sha256"],
                "Remote manifest is missing or differs from the worker receipt")
        manifest = json.loads(manifest_bytes)
        require(manifest.get("destination_root") == str(local_lane) and manifest.get("ack_path") == ack_rel
                and len(manifest["files"]) == transfer["file_count"],
                "Manifest destination, acknowledgment or count differs from the worker receipt")
        members = [relative_member(entry["path"]) for entry in manifest["files"]]
        require(len(members) == len(set(members)), "Duplicate artifact paths in manifest")
        for relative in [manifest_rel, ack_rel, *members]:
            target = local_lane / relative
            require(local_lane in target.resolve().parents and not target.is_symlink(),
                    f"Destination escapes the lane root: {relative}")
        existing = json.loads(local_ack.read_text()) if local_ack.is_file() else None
        remote_existing = transport.read_bytes(remote_ack)
        if existing is not None and existing.get("manifest_sha256") == transfer["manifest_sha256"]:
            if remote_existing is not None:
                require(remote_existing == local_ack.read_bytes(),
                        "The lane holds a different acknowledgment for this transfer; inspect before continuing")
                return {"status": "already_acknowledged", "remote": str(remote), "ack": str(local_ack),
                        "verified_unix": existing.get("verified_unix")}
            # Earlier upload failed: prove the bytes again (throwaway receipt), then upload the original.
            with tempfile.TemporaryDirectory(prefix="v3-host-pull-") as scratch:
                verify(local_lane / manifest_rel, local_lane, Path(scratch) / "recheck.ack.json")
            uploaded, superseded = "re-uploaded unchanged", []
        else:
            require(remote_existing is None, "The lane already holds an acknowledgment that this host did not write")
            superseded = set_aside_superseded(local_lane, manifest["files"], PurePosixPath(manifest_rel).stem)
            transport.fetch(str(remote), members, local_lane)
            _atomic_write(local_lane / manifest_rel, manifest_bytes)
            verify(local_lane / manifest_rel, local_lane, local_ack)
            uploaded = "verified and uploaded"
        again = transport.read_bytes(str(remote / "execution_state.json"))
        require(again is not None and json.loads(again).get("pending_transfer") == transfer,
                "Worker transfer changed during download; acknowledgment was not uploaded")
        transport.put(local_ack, remote_ack + ".partial")
        transport.move(remote_ack + ".partial", remote_ack)
        ack = json.loads(local_ack.read_text())
        return {"status": "acknowledged", "how": uploaded, "remote": str(remote), "manifest": manifest_rel,
                "file_count": ack["file_count"], "verified_unix": ack["verified_unix"],
                "superseded_local_copies": superseded}


def pull(transport, remote_root: str, durable_root: Path, lanes: list[int], *, source_root: Path = ROOT
         ) -> dict[str, Any]:
    require(all(type(k) is int and 0 <= k <= 7 for k in lanes) and len(set(lanes)) == len(lanes), "lanes are 0..7")
    verify = load_verifier(source_root)

    def one(lane: int) -> dict[str, Any]:
        try:
            return pull_lane(transport, f"{remote_root}/lanes/lane{lane}", Path(durable_root) / f"lane{lane}", verify)
        except (HostPullError, OSError, ValueError, KeyError, RuntimeError) as exc:
            return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    with ThreadPoolExecutor(max_workers=max(1, len(lanes))) as pool:
        results = dict(zip((f"lane{k}" for k in lanes), pool.map(one, lanes)))
    return {"lanes": results, "ok": all(r["status"] != "error" for r in results.values())}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--remote-root", required=True, help="VM campaign root (lanes/laneK below it)")
    parser.add_argument("--durable-root", type=Path, required=True,
                        help="workstation destination, exactly the --durable-root given to the VM runner")
    parser.add_argument("--lanes", type=int, nargs="+", default=[0])
    parser.add_argument("--host", default="deepmzyme-vm")
    parser.add_argument("--ssh-config", type=Path, default=DEFAULT_SSH_CONFIG)
    parser.add_argument("--source-root", type=Path, default=ROOT, help="checkout holding verify_host_pull")
    args = parser.parse_args(argv)
    result = pull(SshTransport(args.host, args.ssh_config), args.remote_root, args.durable_root, args.lanes,
                  source_root=args.source_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
