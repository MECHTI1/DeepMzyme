"""Workstation transport and the deliberately started, bounded G2 measurement queue.

Provider actions remain in the installed controller. The default CLI only checks
local readiness; --run starts once or recovers this driver's still-active session.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from typing import Any, Mapping

from ec_clean30 import phase3

GIB = 1024 ** 3
MAX_G2_SECONDS = 117 * 60
GROSS_CEILING_USD = 3.0
PAYLOAD_CEILING_BYTES = 3 * GIB
EGRESS_RESERVE_GIB = 4
RESERVE_SECONDS = 900
OFFICIAL_ESM1B_WEIGHTS = {
    "https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt": {
        "field": "weights", "sha256": "0569754efaff7dcb7e068c27367bc73f10afb4b450ea30aac30d9bc60783a8b1", "size_bytes": 7828576466},
    "https://dl.fbaipublicfiles.com/fair-esm/regression/esm1b_t33_650M_UR50S-contact-regression.pt": {
        "field": "contact_weights", "sha256": "77193a8814f0db0b36a03aebb1a311adc6b4745f463c04839defc15407bbb28a", "size_bytes": 3174},
}
QUEUE = tuple([("timing-unit", name) for name in ("F1", "F2", "F3", "F4", "F5", "C-G", "F4_L3",
                                                        "lane_serial_1", "lane_serial_2")]
              + [("lane-gate", "lane_batch"), ("embed-timing", "esmc"), ("embed-timing", "esm1b")])


class WorkstationError(RuntimeError):
    """Readiness, ownership, persistence or a fixed operational ceiling refused work."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise WorkstationError(message)


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _relative(value: str) -> str:
    path = PurePosixPath(value)
    _require(bool(value) and not path.is_absolute() and ".." not in path.parts
             and str(path) == value and not any(c in value for c in "\n\r\0"), "Unsafe transfer member.")
    return value


def _absolute(value: str) -> str:
    path = PurePosixPath(value)
    _require(path.is_absolute() and ".." not in path.parts and not any(c in value for c in "\n\r\0"),
             "An explicit absolute path is required.")
    phase3._no_restricted_path(Path(value))
    return value


def _number(value: Any, name: str) -> float:
    _require(not isinstance(value, bool) and isinstance(value, (int, float))
             and math.isfinite(value) and value > 0, f"{name} must be finite and positive.")
    return float(value)


def _epoch(value: str) -> float:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _require(parsed.tzinfo is not None, "Controller timestamps require a timezone.")
    return parsed.timestamp()


class ControllerTransport:
    """Only controller lifecycle tools and its generated SSH configuration."""

    def __init__(self, controller_root: Path, ssh_alias: str, *, bootstrap_python: str = "/usr/bin/python3"):
        self.controller_root = Path(controller_root).resolve()
        self.ssh_config = self.controller_root / "state/ssh_config"
        _require(re.fullmatch(r"[A-Za-z0-9_.-]+", ssh_alias), "Invalid controller SSH alias.")
        self.ssh_alias, self.bootstrap_python = ssh_alias, _absolute(bootstrap_python)
        self.deadline_unix: float | None = None
        self.reserve_seconds = 30

    def _bounded(self, timeout: float) -> float:
        if self.deadline_unix is not None:
            timeout = min(timeout, self.deadline_unix - time.time() - self.reserve_seconds)
        _require(timeout > 0, "No allocation time remains for this transport operation and its reserve.")
        return timeout

    def controller(self, name: str, *args: str, timeout: float = 120) -> str:
        _require(name in {"status", "report", "start", "stop", "pycharm-info"}, "Unknown controller action.")
        result = subprocess.run([str(self.controller_root / "bin" / f"vm-{name}"), *args],
                                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)
        _require(result.returncode == 0, f"Controller {name} failed: {(result.stderr or result.stdout)[-2000:]}")
        return result.stdout

    def remote(self, argv: list[str], *, cwd: str | None = None, timeout: float = 60) -> str:
        command = shlex.join([str(value) for value in argv])
        if cwd:
            command = f"cd {shlex.quote(_absolute(cwd))} && {command}"
        result = subprocess.run(["ssh", "-F", str(self.ssh_config), "-o", "BatchMode=yes", "-o",
                                 "ConnectTimeout=20", "-o", "ServerAliveInterval=10", "-o",
                                 "ServerAliveCountMax=2", self.ssh_alias, command],
                                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=self._bounded(timeout))
        _require(result.returncode == 0, f"Remote command failed ({result.returncode}): {result.stderr[-2000:]}")
        _require(len(result.stdout) <= 4 * 1024 * 1024, "Unexpectedly large remote control output.")
        return result.stdout

    def read_json(self, path: str, *, optional: bool = False) -> Any:
        program = ("import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); "
                   "print(p.read_text() if p.is_file() else ('null' if sys.argv[2]=='1' else '{}')); "
                   "sys.exit(0 if p.is_file() or sys.argv[2]=='1' else 2)")
        return json.loads(self.remote([self.bootstrap_python, "-c", program, _absolute(path), str(int(optional))]))

    def stat_file(self, path: str) -> dict[str, Any] | None:
        program = ("import hashlib,json,pathlib,sys; p=pathlib.Path(sys.argv[1]); "
                   "print(json.dumps({'size_bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()} "
                   "if p.is_file() and not p.is_symlink() else None))")
        return json.loads(self.remote([self.bootstrap_python, "-c", program, _absolute(path)], timeout=600))

    def upload(self, local: Path, remote: str, *, timeout: float = 600) -> None:
        remote = _absolute(remote)
        self.remote(["mkdir", "-p", str(PurePosixPath(remote).parent)])
        subprocess.run(["scp", "-q", "-F", str(self.ssh_config), "-o", "BatchMode=yes",
                       str(local), f"{self.ssh_alias}:{remote}"], check=True, timeout=self._bounded(timeout),
                       stdin=subprocess.DEVNULL)

    def fetch_official_weight(self, url: str, partial: str) -> dict[str, Any]:
        """A terminal result permits upload fallback; an ambiguous SSH failure does not."""
        _require(url in OFFICIAL_ESM1B_WEIGHTS, "Remote weight URL is not approved.")
        expected = OFFICIAL_ESM1B_WEIGHTS[url]
        timeout = self._bounded(600)
        _require(timeout > 5, "No allocation time remains for a bounded public weight download.")
        program = """import hashlib,json,pathlib,signal,sys,urllib.request
url,target,digest,size,bound=sys.argv[1:]
target=pathlib.Path(target); size=int(size); bound=float(bound)
def expired(signum,frame):
    raise TimeoutError('Public weight download exceeded its wall-clock bound')
signal.signal(signal.SIGALRM,expired)
signal.setitimer(signal.ITIMER_REAL,bound)
try:
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.is_symlink(): raise ValueError('Weight partial cannot be a symlink')
    hashed=hashlib.sha256(); count=0
    with urllib.request.urlopen(url,timeout=min(30,bound)) as response:
        if response.geturl()!=url: raise ValueError('Unexpected public weight redirect')
        with target.open('wb') as output:
            while chunk:=response.read(8*1024*1024):
                count+=len(chunk)
                if count>size: raise ValueError('Public weight exceeds its pinned size')
                output.write(chunk); hashed.update(chunk)
    if count!=size or hashed.hexdigest()!=digest: raise ValueError('Public weight checksum or size differs')
    result={'terminal':True,'status':'verified','sha256':digest,'size_bytes':count}
except Exception as exc:
    result={'terminal':True,'status':'failed','error':type(exc).__name__+': '+str(exc)}
finally:
    signal.setitimer(signal.ITIMER_REAL,0)
print(json.dumps(result))
"""
        result = json.loads(self.remote([self.bootstrap_python, "-c", program, url, _absolute(partial),
                            expected["sha256"], str(expected["size_bytes"]), str(timeout - 3)], timeout=timeout))
        _require(result.get("terminal") is True and result.get("status") in {"verified", "failed"},
                 "Public weight download has no confirmed terminal result; upload fallback is unsafe.")
        return result

    def download(self, remote: str, local: Path, *, timeout: float = 120) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["scp", "-q", "-F", str(self.ssh_config), "-o", "BatchMode=yes",
                        f"{self.ssh_alias}:{_absolute(remote)}", str(local)], check=True, timeout=self._bounded(timeout),
                       stdin=subprocess.DEVNULL)

    def download_files(self, remote_root: str, local_root: Path, files: list[str], *, timeout: float) -> None:
        listing = local_root / ".pending_files"
        listing.write_text("".join(_relative(name) + "\n" for name in files))
        ssh = shlex.join(["ssh", "-F", str(self.ssh_config), "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"])
        subprocess.run(["rsync", "-a", "-s", "--timeout=45", f"--files-from={listing}", "-e", ssh,
                        f"{self.ssh_alias}:{_absolute(remote_root)}/", str(local_root) + "/"],
                       check=True, timeout=self._bounded(timeout), stdin=subprocess.DEVNULL)

    def launch(self, argv: list[str], *, cwd: str, job_dir: str) -> None:
        # An immutable job directory makes loss of the launch SSH response safe to reconcile.
        worker = ("import json,os,pathlib,subprocess,sys,time; d=pathlib.Path(sys.argv[1]); "
                  "a=json.loads(sys.argv[2]); c=sys.argv[3]; "
                  "p=subprocess.run(a,cwd=c); q=d/'result.partial'; "
                  "q.write_text(json.dumps({'returncode':p.returncode,'finished_unix':time.time()})); "
                  "q.replace(d/'result.json')")
        launcher = ("import json,os,pathlib,subprocess,sys; d=pathlib.Path(sys.argv[1]); "
                    "d.parent.mkdir(parents=True,exist_ok=True); "
                    "\ntry: d.mkdir()\nexcept FileExistsError: sys.exit(0)\n"
                    "a=json.loads(sys.argv[2]); c=sys.argv[3]; "
                    "(d/'command.json').write_text(json.dumps({'argv':a,'cwd':c})); "
                    "f=(d/'stdout.log').open('xb'); "
                    "p=subprocess.Popen([sys.executable,'-c',sys.argv[4],str(d),json.dumps(a),c], "
                    "stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True); "
                    "(d/'pid').write_text(str(p.pid))")
        self.remote([self.bootstrap_python, "-c", launcher, _absolute(job_dir), json.dumps(argv),
                     _absolute(cwd), worker])


class HostPull:
    """Independent readback; conservative transfer reservation survives retries and restarts."""

    def __init__(self, transport: ControllerTransport, durable_root: Path, state_root: Path, *,
                 max_download_bytes: int, deadline_unix: float):
        self.transport, self.root, self.state_root = transport, Path(durable_root).resolve(), Path(state_root).resolve()
        self.max_bytes, self.deadline = max_download_bytes, deadline_unix
        _require(max_download_bytes > 0, "A positive host-download ceiling is required.")
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.budget_path = self.state_root / "transfer_budget.json"

    def _reserve(self, count: int) -> None:
        state = _json(self.budget_path) if self.budget_path.exists() else {"reserved_payload_bytes": 0}
        _require(type(count) is int and count >= 0 and state["reserved_payload_bytes"] + count <= self.max_bytes,
                 "Independent transfer would exceed the recorded payload ceiling.")
        state["reserved_payload_bytes"] += count
        _write(self.budget_path, state)

    def _timeout(self) -> float:
        remaining = self.deadline - time.time()
        _require(remaining > 30, "No provider time remains for verified transfer.")
        return min(600, remaining - 15)

    def sync_pending(self, execution_root: str) -> bool:
        state = self.transport.read_json(f"{execution_root}/execution_state.json", optional=True)
        pending = (state or {}).get("pending_transfer")
        if not pending:
            return False
        relative, ack_relative = _relative(pending["manifest_path"]), _relative(pending["ack_path"])
        manifest_path, ack_path = self.root / relative, self.root / ack_relative
        started = time.monotonic()
        self.transport.download(f"{execution_root}/{relative}", manifest_path, timeout=self._timeout())
        manifest = _json(manifest_path)
        _require(_sha(manifest_path) == pending["manifest_sha256"]
                 and manifest.get("destination_root") == str(self.root)
                 and manifest.get("ack_path") == ack_relative, "Pending transfer identity differs.")
        files, count = [], 0
        for entry in manifest["files"]:
            name = _relative(entry["path"])
            path = self.root / name
            _require(self.root in path.resolve().parents and not path.is_symlink(), "Host file escapes its destination.")
            _require(type(entry["size_bytes"]) is int and entry["size_bytes"] >= 0, "Invalid transfer size.")
            if not path.is_file() or path.stat().st_size != entry["size_bytes"] or _sha(path) != entry["sha256"]:
                files.append(name)
                count += entry["size_bytes"]
        if files:
            for attempt in range(2):
                self._reserve(count)  # Charge the full attempt, including a failed/partial rsync.
                try:
                    self.transport.download_files(execution_root, self.root, files, timeout=self._timeout())
                    break
                except (OSError, subprocess.SubprocessError, WorkstationError):
                    if attempt == 1:
                        raise
        if ack_path.exists():
            previous = _json(ack_path)
            _require(previous.get("manifest_sha256") == pending["manifest_sha256"], "Existing host acknowledgment is foreign.")
            # Read every byte again without replacing the already-uploaded acknowledgment.
            from benchmarking.pmm_execution import verify_host_pull
            temporary = self.state_root / "rechecked_ack.json"
            verify_host_pull(manifest_path, self.root, temporary)
            temporary.unlink()
        else:
            phase3.host_ack(manifest_path, self.root, ack_path, transfer_seconds=time.monotonic() - started)
        self.transport.upload(ack_path, f"{execution_root}/{ack_relative}", timeout=self._timeout())
        return True

    def receipt(self, execution_root: str, relative: str) -> dict[str, Any]:
        relative = _relative(relative)
        remote = f"{execution_root}/{relative}"
        expected = self.transport.stat_file(remote)
        _require(expected is not None, "Completed receipt is absent.")
        self._reserve(int(expected["size_bytes"]))
        local = self.root / relative
        self.transport.download(remote, local, timeout=self._timeout())
        _require(_sha(local) == expected["sha256"], "Completed receipt changed in transit.")
        return _json(local)


def complete_remote_unit(transport: ControllerTransport, host: HostPull, request: Mapping[str, Any], *,
                         runner: str, action: str, remote_request: str, completed_relative: str,
                         timeout: float) -> dict[str, Any]:
    """Run/recover one declared unit, acknowledge its bytes, then finalize and copy its receipt."""
    root = _absolute(request["execution_root"])
    _require(str(host.root) == request["durable_root"], "Request and independent destination differ.")
    host.sync_pending(root)
    local_request = host.state_root / "requests" / (hashlib.sha256(remote_request.encode()).hexdigest() + ".json")
    if local_request.exists():
        _require(_json(local_request) == dict(request), "An existing unit request cannot change during recovery.")
    else:
        _write(local_request, dict(request))
    transport.upload(local_request, remote_request)
    argv = [request["python"], _absolute(runner), action, "--request", _absolute(remote_request)]
    cwd = str(PurePosixPath(runner).parent.parent)
    for suffix in (".job", ".finalize", ".seal"):
        job = remote_request + suffix
        # Launch is idempotent even if its SSH response was lost; never invent another job path.
        try:
            transport.launch(argv, cwd=cwd, job_dir=job)
        except (OSError, subprocess.SubprocessError, WorkstationError):
            saved = transport.read_json(job + "/command.json", optional=True)
            _require(saved == {"argv": argv, "cwd": cwd}, "Ambiguous remote launch; no duplicate submitted.")
        _require(transport.read_json(job + "/command.json") == {"argv": argv, "cwd": cwd},
                 "An existing remote job has a different command; no duplicate submitted.")
        until = min(time.monotonic() + timeout, time.monotonic() + host.deadline - time.time() - 30)
        result = None
        while time.monotonic() < until:
            try:
                result = transport.read_json(job + "/result.json", optional=True)
            except (OSError, subprocess.SubprocessError, WorkstationError):
                # One bounded read-only reconnect; it cannot launch or restart a worker.
                result = transport.read_json(job + "/result.json", optional=True)
            if result is not None:
                break
            time.sleep(min(5, max(.01, until - time.monotonic())))
        acknowledged = host.sync_pending(root)  # Preserve failures before refusing.
        _require(result is not None, "Remote unit did not finish before its bound; reconcile its fixed job path.")
        _require(result.get("returncode") == 0, "Remote unit failed; its terminal artifacts were preserved.")
        if suffix != ".job" and not acknowledged:
            break
        _require(suffix != ".seal" or not acknowledged, "Unexpected additional persistence cycle; no next unit admitted.")
    return host.receipt(root, completed_relative)


def _controller_config(root: Path) -> dict[str, str]:
    result = {}
    for line in (root / "config.env").read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip().strip('"').strip("'")
    return result


def _repository_key(url: str) -> str:
    normalized = url.removeprefix("git@github.com:").removeprefix("https://github.com/").removesuffix(".git")
    _require(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", normalized), "Unexpected repository origin.")
    return normalized.lower()


def _identity(config: Mapping[str, Any], controller: Mapping[str, str]) -> None:
    names = {"project": "PROJECT_ID", "zone": "ZONE", "vm_name": "VM_NAME", "instance_id": "SELECTED_INSTANCE_ID"}
    _require(set(config["expected_identity"]) == set(names)
             and all(str(config["expected_identity"][key]) == controller.get(name) for key, name in names.items()),
             "Controller endpoint differs from the reviewed immutable instance identity.")
    _require(controller.get("MACHINE_TYPE") == "g2-standard-8" and controller.get("ZONE") == "us-central1-a"
             and controller.get("SSH_HOST_ALIAS") == config["ssh_alias"], "G2 hardware, region or alias differs.")


def _status(text: str, expected: Mapping[str, Any]) -> str:
    found = re.search(r"^\s*State:\s+([A-Z_]+)\s*$", text, re.MULTILINE)
    _require(found is not None and f"project={expected['project']}" in text and f"zone={expected['zone']}" in text
             and re.search(r"VM:\s+" + re.escape(expected["vm_name"]) + r"\s", text), "Unknown controller status/endpoint.")
    _require("PROTECTION PROBLEM" not in text and "duplicate scan failed" not in text
             and "Controller lock: HELD" not in text, "Controller protection or ownership is uncertain.")
    other = [line for line in text.splitlines() if "OTHER GPU/MANAGED VM:" in line]
    _require(all(line.rstrip().endswith(" TERMINATED") for line in other), "Another GPU is active or uncertain.")
    if found[1] == "RUNNING":
        _require("Google-side auto-STOP: VERIFIED" in text, "Provider STOP was not verified.")
    return found[1]


def _cost(config: Mapping[str, Any]) -> dict[str, float]:
    price = _json(Path(config["controller_root"]) / "state/price_cache.json")
    quote = config["pricing"]
    for record in (price, quote):
        _require(0 <= time.time() - _epoch(record["verified_at"]) <= 86400, "Cost evidence is stale or future-dated.")
    _require(price.get("machine_type") == "g2-standard-8" and price.get("region") == "us-central1",
             "Controller price is for different hardware or region.")
    gross = _number(price["gross_usd_per_hour"], "gross rate")
    snapshots = _number(quote["snapshot_usd_per_hour"], "retained snapshot rate")
    egress = _number(quote["egress_usd_per_gib"], "egress rate")
    _require(egress >= .15, "The recorded egress rate cannot omit the destination-region charge.")
    ceiling = gross * 2 * 1.15 + snapshots * 2 + egress * EGRESS_RESERVE_GIB
    _require(ceiling <= GROSS_CEILING_USD, "Compute, retained snapshots and egress exceed the $3 gross ceiling.")
    return {"gross_usd_per_hour": gross, "snapshot_usd_per_hour": snapshots,
            "egress_usd_per_gib": egress, "conservative_max_usd": ceiling}


def _view_archive_bytes(path: Path) -> int:
    """The separately extracted view also occupies disk alongside its staged tar."""
    try:
        with tarfile.open(path) as archive:
            members = archive.getmembers()
            names = [phase3._relative(member.name) for member in members]
            _require(len(names) == len(set(names)) and all(member.isreg() and not member.linkname for member in members),
                     "Fold-0 transfer requires unique regular archive members.")
            _require(phase3.TRANSFER_MANIFEST in names, "Fold-0 transfer manifest is absent.")
            descriptor = archive.getmember(phase3.TRANSFER_MANIFEST)
            _require(descriptor.size <= 16 * 1024 * 1024, "Fold-0 transfer manifest is oversized.")
            manifest = json.load(archive.extractfile(descriptor))
            _require(manifest.get("format") == phase3.TRANSFER_FORMAT and manifest.get("campaign_id") == phase3.CAMPAIGN_ID
                     and manifest.get("fold") == 0 and manifest.get("smoke") is False, "Invalid Fold-0 transfer header.")
            records = manifest.get("files", [])
            expected = {row["path"]: row for row in records}
            _require(len(expected) == len(records) and set(names) == {phase3.TRANSFER_MANIFEST, *expected}
                     and all(archive.getmember(name).size == row["size_bytes"] for name, row in expected.items()),
                     "Fold-0 transfer membership or uncompressed sizes differ from its manifest.")
            return sum(member.size for member in members)
    except (tarfile.TarError, ValueError, KeyError, TypeError) as exc:
        raise WorkstationError(f"Invalid Fold-0 transfer archive: {exc}") from exc


def _weight_url(config: Mapping[str, Any], item: Mapping[str, Any]) -> str | None:
    if "remote_url" not in item:
        return None
    url = item["remote_url"]
    expected = OFFICIAL_ESM1B_WEIGHTS.get(url) if isinstance(url, str) else None
    _require(expected is not None and item["kind"] == "file", "Remote weight URL is not approved.")
    clean = config["embedding"]["esm1b"]
    destination = clean["weights"] if expected["field"] == "weights" else str(PurePosixPath(clean["weights"]).with_suffix("")) + "-contact-regression.pt"
    _require(item["remote"] == destination and item["sha256"] == expected["sha256"]
             and clean[expected["field"] + "_sha256"] == expected["sha256"]
             and Path(item["local"]).stat().st_size == expected["size_bytes"],
             "Remote weight URL differs from its exact pin, size or embedding destination.")
    return url


def preflight(config: Mapping[str, Any], config_path: Path, *, published: bool = False) -> dict[str, Any]:
    """Local-only readiness; no controller or SSH call and no cloud allocation."""
    controller = Path(config["controller_root"]).resolve()
    controller_config = _controller_config(controller)
    _identity(config, controller_config)
    _require(controller_config["REPO_URL"].startswith("https://github.com/"), "Controller checkout URL must use canonical HTTPS.")
    origin = subprocess.check_output(["git", "-C", str(phase3.REPO_ROOT), "config", "--get", "remote.origin.url"], text=True).strip()
    _require(_repository_key(origin) == _repository_key(controller_config["REPO_URL"]), "Local/controller repository identities differ.")
    for name in ("status", "report", "start", "stop", "pycharm-info"):
        _require(os.access(controller / "bin" / f"vm-{name}", os.X_OK), f"Controller vm-{name} is unavailable.")
    source = phase3.verify_execution_source(Path(config["execution_manifest"]), require_published=published)
    secondary = Path(config["secondary_drive"]).resolve()
    _require(secondary.is_mount(), "The independent secondary drive is not mounted.")
    for key in ("durable_root", "state_root"):
        path = Path(_absolute(config[key])).resolve()
        _require(secondary in path.parents, f"{key} must be on the independent secondary drive.")
    _require(shutil.disk_usage(secondary).free >= 5 * GIB + PAYLOAD_CEILING_BYTES,
             "The independent drive lacks the 5 GiB reserve plus the download allowance.")
    _require(shutil.disk_usage(phase3.REPO_ROOT).free >= 5 * GIB, "Internal ext4 is below its 5 GiB reserve.")
    for key in ("remote_repo", "remote_python", "remote_campaign", "view_archive", "timing_inputs", "bootstrap_python"):
        _absolute(config[key])
    for command in ("ssh", "scp", "rsync"):
        _require(shutil.which(command), f"Required local transport command is unavailable: {command}")
    staged, content, staging_bytes = set(), set(), 0
    for item in config["stage"]:
        local, remote = Path(item["local"]).resolve(), _absolute(item["remote"])
        _require(local.is_file() and _sha(local) == item["sha256"], "A staging file is missing or changed.")
        _weight_url(config, item)
        _require(remote not in staged and item["kind"] in {"file", "tar"}, "Duplicate or unsupported staging entry.")
        staged.add(remote)
        content.add(remote)
        staging_bytes += local.stat().st_size
        if item["kind"] == "tar":
            _absolute(item["extract_to"])
            _require(type(item.get("conda_unpack", False)) is bool, "conda_unpack must be a boolean.")
            with tarfile.open(local) as archive:
                for member in archive:
                    name = member.name.removeprefix("./").rstrip("/")
                    if not name:
                        continue
                    _relative(name)
                    if member.isfile() or member.issym() or member.islnk():
                        content.add(str(PurePosixPath(item["extract_to"]) / name))
                    if member.isfile():
                        staging_bytes += member.size
    _require(config["view_archive"] in staged and any(item["remote"] == config["view_archive"]
             and item["sha256"] == config["view_archive_sha256"] for item in config["stage"]), "Fold-0 archive lacks a staging pin.")
    view_item = next(item for item in config["stage"] if item["remote"] == config["view_archive"])
    view_bytes = _view_archive_bytes(Path(view_item["local"]))
    if view_item["kind"] == "file":
        staging_bytes += view_bytes
    _require(set(config["bounds"]) == {name for _, name in QUEUE}, "G2 must retain the exact twelve-event queue.")
    for name, bound in config["bounds"].items():
        forecast = _number(bound["forecast_seconds"], name + " forecast")
        timeout = _number(bound["unit_timeout_seconds"], name + " timeout")
        _require(timeout <= forecast and forecast * 1.25 + RESERVE_SECONDS < MAX_G2_SECONDS,
                 "A bounded unit cannot fit the fixed G2 admission window.")
    for key in ("fit", "esm1b"):
        profile = config["profiles"][key]
        _require(isinstance(profile["cap_prefix"], list) and profile["cap_prefix"]
                 and all(isinstance(value, str) and value for value in profile["cap_prefix"]), "Explicit resource-cap argv required.")
        _number(profile["memory_limit_bytes"], "memory cap")
        cores, gib = (2, 8) if key == "fit" else (4, 24)
        required_cap = {"--user", "--scope", f"CPUQuota={cores * 100}%", f"MemoryMax={gib}G", "MemorySwapMax=0"}
        _require(Path(profile["cap_prefix"][0]).name == "systemd-run"
                 and required_cap.issubset(set(profile["cap_prefix"])) and profile["memory_limit_bytes"] == gib * GIB,
                 "G2 requires its declared systemd CPU/RAM/no-swap cap.")
    for key in ("cpu_percent", "ram_bytes", "vram_bytes"):
        _number(config["lane_limits"][key], key)
    _require(config["lane_limits"] == {"cpu_percent": 600, "ram_bytes": 24 * GIB, "vram_bytes": 21 * GIB},
             "Lane measurement limits differ from the G2 scope.")
    esmc, clean = config["embedding"]["esmc"], config["embedding"]["esm1b"]
    for key in ("hf_home", "esmc_weights"):
        _absolute(esmc[key])
    for key in ("clean_python", "harness", "extract_script", "weights"):
        _absolute(clean[key])
    for key in ("weights_sha256", "contact_weights_sha256"):
        _require(re.fullmatch(r"[0-9a-f]{64}", clean[key]), "Embedding weight checksum missing.")
    _number(clean["threads"], "CLEAN CPU threads")
    contact = str(PurePosixPath(clean["weights"]).with_suffix("")) + "-contact-regression.pt"
    required = [esmc["esmc_weights"], clean["clean_python"], clean["harness"], clean["extract_script"], clean["weights"], contact]
    required += [str(PurePosixPath(config["timing_inputs"]) / name) for name in ("timing_inputs.json", "timing.fasta", "sequence_map.tsv")]
    _require(all(path in content for path in required), "The prepared staging closure omits an embedding input or interpreter.")
    cost = _cost(config)
    return {"status": "ready" if source.get("published") else "draft_unpublished", "source": source,
            "cost": cost, "queue": [{"action": action, "name": name} for action, name in QUEUE],
            "staging_peak_bytes": staging_bytes, "pending_runtime_checks": ["VM free space", "both CUDA interpreters", "source and view probe"],
            "max_provider_seconds": MAX_G2_SECONDS, "payload_ceiling_bytes": PAYLOAD_CEILING_BYTES,
            "launch_command": shlex.join([sys.executable, str(phase3.REPO_ROOT / "scripts/run_ec_clean30_workstation.py"),
                                          "--config", str(config_path.resolve()), "--run"])}


def _session(config: Mapping[str, Any], transport: ControllerTransport, previous: str | None = None) -> dict[str, Any]:
    _identity(config, _controller_config(transport.controller_root))
    status = transport.controller("status")
    _require(_status(status, config["expected_identity"]) == "RUNNING", "The owned allocation is not RUNNING.")
    current = _json(transport.controller_root / "state/current_session.json")
    _require(current.get("active") is True and current.get("session_id")
             and all(str(current.get(key)) == str(value) for key, value in config["expected_identity"].items()),
             "Controller session does not identify the reviewed instance.")
    if previous:
        _require(current["session_id"] == previous, "The controller now owns a different allocation; refusing recovery.")
    started, deadline = _epoch(current["session_start"]), _epoch(current["termination_ts"])
    _require(0 < deadline - started <= MAX_G2_SECONDS and current["max_run_duration_s"] <= MAX_G2_SECONDS
             and started <= time.time() < deadline, "Actual provider deadline exceeds 117 minutes or has elapsed.")
    _require(abs(float(current["gross_usd_per_hour"]) - _cost(config)["gross_usd_per_hour"]) < 1e-6,
             "Actual session pricing differs from the verified quote.")
    return {"session_id": current["session_id"], "allocation_started_unix": started, "deadline_unix": deadline,
            "max_total_seconds": min(MAX_G2_SECONDS, deadline - started)}


def _stage_file(config: Mapping[str, Any], transport: ControllerTransport, item: Mapping[str, Any]) -> None:
    url = _weight_url(config, item)
    expected = {"sha256": item["sha256"], "size_bytes": Path(item["local"]).stat().st_size}
    existing = transport.stat_file(item["remote"])
    if existing is None:
        partial = item["remote"] + ".partial"
        downloaded = None
        if url:
            result = transport.fetch_official_weight(url, partial)
            # The downloader has exited and closed its writer before returning this status.
            if result["status"] == "verified":
                downloaded = transport.stat_file(partial)
        if downloaded != expected:
            transport.upload(Path(item["local"]), partial, timeout=MAX_G2_SECONDS)
            existing = transport.stat_file(partial)
        else:
            existing = downloaded
        _require(existing == expected, "Staged upload/download checksum or size differs.")
        transport.remote(["mv", "--", partial, item["remote"]])
    _require(existing == expected, "Remote staging checksum or size differs.")


def _stage_runtime(config: Mapping[str, Any], transport: ControllerTransport, source: Mapping[str, Any]) -> None:
    origin = _controller_config(transport.controller_root)["REPO_URL"]
    checkout = ("import pathlib,subprocess,sys; p=pathlib.Path(sys.argv[1]); "
                "\nif not p.exists(): subprocess.run(['git','clone',sys.argv[2],str(p)],check=True)\n"
                "assert not subprocess.check_output(['git','-C',str(p),'status','--porcelain'],text=True).strip(), 'Remote checkout is dirty'; "
                "subprocess.run(['git','-C',str(p),'fetch','origin','main:refs/remotes/origin/main'],check=True); "
                "subprocess.run(['git','-C',str(p),'checkout','--detach',sys.argv[3]],check=True)")
    transport.remote([config["bootstrap_python"], "-c", checkout, config["remote_repo"], origin, source["git_head"]], timeout=300)
    for item in config["stage"]:
        _stage_file(config, transport, item)
        if item["kind"] == "tar":
            # Python's data filter rejects paths/links escaping the new extraction directory.
            unpack = ("import json,pathlib,subprocess,sys,tarfile; a,p,h,u=sys.argv[1:]; p=pathlib.Path(p); "
                      "m=p/'.g2_archive.json'; "
                      "\nif p.exists():\n assert m.is_file() and json.loads(m.read_text())['sha256']==h, 'Unrecognized existing extraction'\n"
                      "else:\n p.mkdir(parents=True)\n with tarfile.open(a) as t: t.extractall(p,filter='data')\n"
                      " if u=='1': subprocess.run([str(p/'bin/python'),str(p/'bin/conda-unpack')],check=True)\n"
                      " m.write_text(json.dumps({'sha256':h}))\n")
            transport.remote([config["bootstrap_python"], "-c", unpack, item["remote"], item["extract_to"],
                              item["sha256"], str(int(item.get("conda_unpack", False)))], timeout=600)


def _runtime_preflight(config: Mapping[str, Any], transport: ControllerTransport, source: Mapping[str, Any]) -> tuple[str, str]:
    repo, campaign = config["remote_repo"], config["remote_campaign"]
    manifest = str(PurePosixPath(repo) / Path(config["execution_manifest"]).resolve().relative_to(phase3.REPO_ROOT).as_posix())
    runner = f"{repo}/scripts/run_ec_clean30_phase3.py"
    transport.remote([config["remote_python"], runner, "verify-execution", "--manifest", manifest], cwd=repo)
    gpu = ("import json,torch; assert torch.cuda.is_available() and torch.version.cuda; "
           "assert torch.cuda.get_device_name(0)=='NVIDIA L4'; "
           "assert (torch.ones(4,device='cuda')*2).sum().item()==8; torch.cuda.synchronize(); "
           "print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0)}))")
    for key, python in (("fit", config["remote_python"]), ("esm1b", config["embedding"]["esm1b"]["clean_python"])):
        transport.remote([*config["profiles"][key]["cap_prefix"], python, "-c", gpu], cwd=repo, timeout=90)
    view = f"{campaign}/views/fold0"
    if transport.read_json(f"{view}/view_manifest.json", optional=True) is None:
        transport.remote([config["remote_python"], runner, "unpack-view", "--archive", config["view_archive"],
                          "--sha256", config["view_archive_sha256"], "--destination", view], cwd=repo, timeout=300)
    cap = f"{campaign}/g2_requests/cap.json"
    local_cap = Path(config["state_root"]) / "cap.json"
    _write(local_cap, config["profiles"]["fit"]["cap_prefix"])
    transport.upload(local_cap, cap)
    probe = f"{campaign}/g2_requests/probe_{source['execution_manifest_sha256']}.json"
    if transport.read_json(probe, optional=True) is None:
        transport.remote([config["remote_python"], runner, "vm-probe", "--view", view,
                          "--execution-manifest", manifest, "--out", probe, "--cap-json", cap,
                          "--python", config["remote_python"]], cwd=repo, timeout=300)
    return manifest, probe


def run_g2(config: Mapping[str, Any], config_path: Path, *, transport: ControllerTransport | None = None) -> dict[str, Any]:
    ready = preflight(config, config_path, published=True)
    transport = transport or ControllerTransport(Path(config["controller_root"]), config["ssh_alias"],
                                                 bootstrap_python=config["bootstrap_python"])
    state_root = Path(config["state_root"])
    state_root.mkdir(parents=True, exist_ok=True)
    state_path = state_root / "driver_state.json"
    with (state_root / "driver.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = _json(state_path) if state_path.exists() else {"config_sha256": _sha(config_path), "completed": []}
        _require(state["config_sha256"] == _sha(config_path), "A started driver's config cannot change.")
        _require(not state.get("closed"), "This allocation is closed; no automatic restart is permitted.")
        owned = bool(state.get("start_attempted"))
        try:
            if not owned:
                _require(_status(transport.controller("status"), config["expected_identity"]) == "TERMINATED",
                         "Start requires the existing instance to be positively stopped.")
                transport.controller("report", "--no-ssh")
                preview = transport.controller("start", "--dry-run", "--hours", "2", timeout=180)
                _require("maxRunDuration=7020s" in preview, "Controller preview does not give the reviewed 117-minute STOP.")
                ready["cost"] = _cost(config)  # Start preview refreshes the controller's quote.
                state.update(start_attempted=True, start_preview=preview, cost=ready["cost"], started_attempt_unix=time.time())
                _write(state_path, state)
                owned = True
                transport.controller("start", "--confirm", "--hours", "2", timeout=1200)
                candidate = _json(transport.controller_root / "state/current_session.json")
                _require(candidate.get("active") is True and candidate.get("session_id")
                         and all(str(candidate.get(key)) == str(value)
                                 for key, value in config["expected_identity"].items())
                         and math.floor(state["started_attempt_unix"]) <= _epoch(candidate["session_start"]) <= time.time(),
                         "Successful start has no matching fresh controller session; ownership is unbound.")
                # Bind successful allocation ownership before later pricing/runtime checks can fail.
                state.update(session_id=candidate["session_id"], controller_session_start=candidate["session_start"])
                _write(state_path, state)
            session = _session(config, transport, state.get("session_id"))
            state.update(session)
            _write(state_path, state)
            transport.deadline_unix = session["deadline_unix"]
            transport.reserve_seconds = RESERVE_SECONDS
            transport.controller("pycharm-info")  # Generates only the controller's own SSH configuration.
            free = transport.remote([config["bootstrap_python"], "-c",
                "import pathlib,shutil,sys; p=pathlib.Path(sys.argv[1]); "
                "\nwhile not p.exists(): p=p.parent\nprint(shutil.disk_usage(p).free)", config["remote_campaign"]])
            _require(int(free.strip()) >= ready["staging_peak_bytes"] + 5 * GIB,
                     "VM disk cannot stage the prepared closure while keeping 5 GiB free.")
            _stage_runtime(config, transport, ready["source"])
            manifest, probe = _runtime_preflight(config, transport, ready["source"])
            transport.reserve_seconds = 30
            host = HostPull(transport, Path(config["durable_root"]), state_root,
                            max_download_bytes=PAYLOAD_CEILING_BYTES, deadline_unix=session["deadline_unix"])
            root = f"{config['remote_campaign']}/runs/fold0/g2"
            common = {**session, "execution_manifest": manifest, "view": f"{config['remote_campaign']}/views/fold0",
                      "vm_probe": probe, "execution_root": root, "durable_root": str(Path(config["durable_root"]).resolve()),
                      "persistence_mode": "host_pull", "python": config["remote_python"], "device": "cuda", "smoke": False,
                      "attempt": 1}
            for action, name in QUEUE:
                if name in state["completed"]:
                    continue
                _session(config, transport, session["session_id"])
                request = {**common, **config["profiles"]["esm1b" if name == "esm1b" else "fit"], **config["bounds"][name]}
                if action == "timing-unit":
                    request["recipe"] = name
                    complete = f"runs/ecc30timing_{name}_attempt1/g2_complete.json"
                elif action == "lane-gate":
                    request.update(serial_receipts=[f"{root}/runs/ecc30timing_lane_serial_{i}_attempt1/g2_complete.json" for i in (1, 2)],
                                   limits=config["lane_limits"])
                    complete = "runs/ecc30timing_lane_batch_attempt1/lane_gate.json"
                    measured = [transport.read_json(path)["wall_seconds"] for path in request["serial_receipts"]]
                    request["forecast_seconds"] = max(request["forecast_seconds"], 3 * max(measured))
                else:
                    request.update(kind=name, timing_inputs=config["timing_inputs"], **config["embedding"][name])
                    complete = f"runs/ecc30timing_embed_{name}_attempt1/embedding_complete.json"
                remaining = session["deadline_unix"] - time.time()
                _require(remaining >= request["forecast_seconds"] * 1.25 + RESERVE_SECONDS,
                         "Next complete unit does not fit the approved allocation; retain evidence and request a new decision.")
                completed = complete_remote_unit(transport, host, request, runner=f"{config['remote_repo']}/scripts/run_ec_clean30_phase3.py",
                    action=action, remote_request=f"{config['remote_campaign']}/g2_requests/{name}.json",
                    completed_relative=complete, timeout=request["unit_timeout_seconds"] + 60)
                _require(completed.get("execution_manifest_sha256") == ready["source"]["execution_manifest_sha256"]
                         and completed.get("status") in ({"passed", "failed"} if action == "lane-gate" else {"passed"}),
                         "The completed unit receipt is failed or foreign.")
                state["completed"].append(name)
                _write(state_path, state)
            state["status"] = "measurements_complete"
        except BaseException as exc:
            state.update(status="interrupted", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if owned:
                try:
                    _identity(config, _controller_config(transport.controller_root))
                    current = _json(transport.controller_root / "state/current_session.json")
                    _require(current.get("active") is False
                             or (current.get("active") is True and bool(state.get("session_id"))
                                 and current.get("session_id") == state["session_id"]
                                 and all(str(current.get(key)) == str(value)
                                         for key, value in config["expected_identity"].items())),
                             "Another allocation or unbound active session owns this endpoint; no blind stop.")
                    transport.controller("stop", "--now", timeout=600)
                    status = transport.controller("status")
                    _require(_status(status, config["expected_identity"]) == "TERMINATED", "Provider STOP is unconfirmed.")
                    state.update(closed=True, provider_stop_verified=True, stop_status=status)
                finally:
                    _write(state_path, state)
        return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", action="store_true", help="Deliberately start once, or recover this still-active owned allocation.")
    args = parser.parse_args(argv)
    try:
        config = _json(args.config)
        if args.run:
            def interrupted(signum, frame):
                raise KeyboardInterrupt(f"signal {signum}")
            signal.signal(signal.SIGTERM, interrupted)
            result = run_g2(config, args.config)
        else:
            result = preflight(config, args.config)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (WorkstationError, OSError, ValueError, KeyError, subprocess.SubprocessError, KeyboardInterrupt) as exc:
        print(f"Workstation refused: {exc}", file=sys.stderr)
        return 2
