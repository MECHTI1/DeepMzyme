"""Workstation transport and the deliberately started, bounded G2 measurement queue.

Provider actions remain in the installed controller. The default CLI only checks
local readiness; --run starts once or recovers this driver's still-active session.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time
from typing import Any, Mapping

from ec_clean30 import phase3

GIB = 1024 ** 3
MAX_G2_SECONDS = 117 * 60
GROSS_CEILING_USD = 3.0
CROSS_REGION_CEILING_USD = 19.0
CROSS_REGION_ZONES = ("us-west1-a", "us-west1-b", "us-west1-c", "us-east4-a", "us-east4-c")
PAYLOAD_CEILING_BYTES = 3 * GIB
EGRESS_RESERVE_GIB = 4
RESERVE_SECONDS = 900
# The closeout inventory is a verified control file, never SSH control output (bounded at 4 MiB).
INVENTORY_LIMIT_BYTES = 512 * 1024 ** 2
INVENTORY_SCAN_SECONDS = 1800
INVENTORY_MIN_SECONDS = 120
INVENTORY_FILE_SECONDS = 0.02
INVENTORY_HASH_BYTES_PER_SECOND = 32 * 1024 ** 2
INVENTORY_TRANSFER_BYTES_PER_SECOND = 1024 ** 2
PUBLISHED_RUNTIME_LOCKS_PATH = "docs/campaigns/ec_class_vs_clean30_2026-10/execution_v3/runtime_locks_v2.json"
PUBLISHED_RUNTIME_LOCKS_SHA256 = "dbbf3cd47be2062f444d4648a57b54c4aa83e717437e139dd3e20a2e261c0483"
OFFICIAL_ESM1B_WEIGHTS = {
    "https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt": {
        "field": "weights", "sha256": "0569754efaff7dcb7e068c27367bc73f10afb4b450ea30aac30d9bc60783a8b1", "size_bytes": 7828576466},
    "https://dl.fbaipublicfiles.com/fair-esm/regression/esm1b_t33_650M_UR50S-contact-regression.pt": {
        "field": "contact_weights", "sha256": "77193a8814f0db0b36a03aebb1a311adc6b4745f463c04839defc15407bbb28a", "size_bytes": 3174},
}
OFFICIAL_WEIGHT_PROGRAM = """import hashlib,json,pathlib,signal,ssl,sys,urllib.error,urllib.request
url,target,digest,size,bound=sys.argv[1:]
target=pathlib.Path(target); size=int(size); bound=float(bound)
def expired(signum,frame):
    raise TimeoutError('Public weight download exceeded its wall-clock bound')
def fetch(context):
    hashed=hashlib.sha256(); count=0
    with urllib.request.urlopen(url,timeout=min(30,bound),context=context) as response:
        if response.geturl()!=url: raise ValueError('Unexpected public weight redirect')
        with target.open('wb') as output:
            while chunk:=response.read(8*1024*1024):
                count+=len(chunk)
                if count>size: raise ValueError('Public weight exceeds its pinned size')
                output.write(chunk); hashed.update(chunk)
    if count!=size or hashed.hexdigest()!=digest: raise ValueError('Public weight checksum or size differs')
    return count
signal.signal(signal.SIGALRM,expired)
signal.setitimer(signal.ITIMER_REAL,bound)
try:
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.is_symlink(): raise ValueError('Weight partial cannot be a symlink')
    tls='verified'
    try:
        count=fetch(None)
    except (urllib.error.URLError,ssl.SSLCertVerificationError) as exc:
        reason=getattr(exc,'reason',exc)
        # Only an expired certificate: the pinned size and SHA-256 still decide whether the bytes are accepted.
        if not (isinstance(reason,ssl.SSLCertVerificationError) and reason.verify_code==10): raise
        context=ssl.create_default_context(); context.check_hostname=False; context.verify_mode=ssl.CERT_NONE
        tls='expired_certificate_content_pinned'
        count=fetch(context)
    result={'terminal':True,'status':'verified','sha256':digest,'size_bytes':count,'tls':tls}
except Exception as exc:
    result={'terminal':True,'status':'failed','error':type(exc).__name__+': '+str(exc)}
    try:
        if not target.is_symlink(): target.unlink(missing_ok=True)
    except OSError:
        pass
finally:
    signal.setitimer(signal.ITIMER_REAL,0)
print(json.dumps(result))
"""
QUEUE = tuple([("timing-unit", name) for name in ("F1", "F2", "F3", "F4", "F5", "C-G", "F4_L3",
                                                        "lane_serial_1", "lane_serial_2")]
              + [("lane-gate", "lane_batch"), ("embed-timing", "esmc"), ("embed-timing", "esm1b")])


def measurement_queue(source: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """The canonical science-owned timing recipes, followed by the frozen lane and embedding checks."""
    if source.get("format") != phase3.EXECUTION_V5_FORMAT:
        return QUEUE
    from ec_clean30 import campaign

    recipes = campaign.timing_recipes_v2()
    return tuple([("timing-unit", name) for name, _recipe in recipes]
                 + [("timing-unit", "lane_serial_1"), ("timing-unit", "lane_serial_2"),
                    ("lane-gate", "lane_batch"), ("embed-timing", "esmc"), ("embed-timing", "esm1b")])


def _g2_download_limit(config: Mapping[str, Any], source: Mapping[str, Any]) -> int:
    if source.get("format") != phase3.EXECUTION_V5_FORMAT:
        return PAYLOAD_CEILING_BYTES
    limit = config.get("g2_max_download_bytes")
    _require(type(limit) is int and limit > 0, "Execution v5 requires its positive G2 preservation-byte ceiling.")
    return limit


class WorkstationError(RuntimeError):
    """Readiness, ownership, persistence or a fixed operational ceiling refused work."""


class HostRejected(WorkstationError):
    """The allocated host refused kernel delegation before any staging; the inputs are not at fault."""


HOST_REJECTION_PREFIX = "Remote command failed (125): container caps refused:"
VM_PROBE_SECONDS = 1800
# SSH reports a failed connect in this form before it delivers any command, so the command did not run.
SSH_CONNECT_FAILURE = re.compile(r"^ssh: connect to host \S+ port \d+: ", re.M)
SSH_RECONNECT_DELAYS = (5, 10, 15)
SSH_RECONNECT_SECONDS = 240
# A new pass (recovery, retry or resume) re-proves its own roots locally; the staged inputs do not move.
PASS_LOCAL_FIELDS = ("state_root", "durable_root", "mirror_root")
PASS_LOCAL_RUNPOD_FIELDS = ("state_root", "pod_name")


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


def session_chain_receipt_path(config: Mapping[str, Any]) -> Path:
    """The write-once receipt of this workstation session."""
    path = Path(config.get("amendment_chain_receipt", Path(config["state_root"]) / "amendment_chain_receipt.json"))
    _require(path.is_absolute() and path.resolve() == path and not path.is_symlink(),
             "The session amendment-chain receipt needs a canonical absolute path.")
    return path


def _chain_manifest(config: Mapping[str, Any]) -> dict[str, Any] | None:
    document = _json(Path(config["execution_manifest"]))
    return document if document.get("format") == phase3.EXECUTION_V5_FORMAT else None


def prepare_session_chain_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    """Verify once on the full clone before allocation; recovery checks the existing receipt.

    This published-source dispatch hook is never called by draft preflight or the historical launcher.
    It cannot create a draft receipt or replace one when the source changes.
    """
    from ec_clean30 import amendment2, amendment_chain

    document = _chain_manifest(config)
    _require(document is not None, "A session chain receipt requires execution manifest v5.")
    chain = document.get("amendment_chain")
    _require(isinstance(chain, list) and len(chain) >= 2
             and all(isinstance(item, str) and re.fullmatch(r"[0-9a-f]{64}", item) for item in chain),
             "Execution v5 requires the manifest's complete amendment-2 chain.")
    tip = phase3.REPO_ROOT / _relative(document["amendment_path"])
    receipt = session_chain_receipt_path(config)
    if not receipt.exists():
        receipt.parent.mkdir(parents=True, exist_ok=True)
        amendment2.write_chain_receipt(tip, document["freeze1_sha256"], receipt,
                                      draft=False, expected_chain=chain, repo=phase3.REPO_ROOT)
    verified = amendment_chain.check_chain_receipt(receipt, repo=phase3.REPO_ROOT, expected_chain=chain)
    _require(verified.get("published") is True and verified.get("tip_sequence", 0) >= 2
             and verified.get("scientific_identity_sha256") == document.get("scientific_identity_sha256")
             and verified.get("fit_chain") == document.get("fit_chain"),
             "The session receipt differs from the manifest's published scientific identity.")
    source = phase3.verify_execution_source(Path(config["execution_manifest"]), require_published=True,
                                            chain_receipt_path=receipt, expected_chain=chain)
    return {"path": str(receipt), "sha256": _sha(receipt), "expected_chain": chain,
            "scientific_identity_sha256": source["scientific_identity_sha256"], "fit_chain": source["fit_chain"]}


def _execution_source(config: Mapping[str, Any], *, published: bool) -> dict[str, Any]:
    document = _chain_manifest(config) if config.get("execution_scope") in {"g2_then_fold0_v5", "fold_execution_v5"} else None
    _require(config.get("execution_scope") not in {"g2_then_fold0_v5", "fold_execution_v5"} or document is not None,
             "The v5 session requires an execution-v5 manifest.")
    options = {} if document is None else {"chain_receipt_path": session_chain_receipt_path(config),
                                          "expected_chain": document["amendment_chain"]}
    return phase3.verify_execution_source(Path(config["execution_manifest"]), require_published=published, **options)


def _remote_chain_request(config: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    if source.get("format") != phase3.EXECUTION_V5_FORMAT:
        return {}
    return {"amendment_chain_receipt": f"{config['remote_campaign']}/g2_requests/amendment_chain_receipt.json",
            "expected_chain": list(source["amendment_chain"])}


def _chain_cli_options(fields: Mapping[str, Any]) -> list[str]:
    if not fields:
        return []
    result = ["--amendment-chain-receipt", fields["amendment_chain_receipt"]]
    for digest in fields["expected_chain"]:
        result += ["--expected-chain", digest]
    return result


class ControllerTransport:
    """Only controller lifecycle tools and its generated SSH configuration."""

    def __init__(self, controller_root: Path, ssh_alias: str, *, bootstrap_python: str = "/usr/bin/python3"):
        self.controller_root = Path(controller_root).resolve()
        self.ssh_config = self.controller_root / "state/ssh_config"
        _require(re.fullmatch(r"[A-Za-z0-9_.-]+", ssh_alias), "Invalid controller SSH alias.")
        self.ssh_alias, self.bootstrap_python = ssh_alias, _absolute(bootstrap_python)
        self.deadline_unix: float | None = None
        self.reserve_seconds = 30

    def refresh(self) -> None:
        """Renew the SSH endpoint before a reconnect; this controller's endpoint is fixed."""

    def _reconnect(self, attempt: int, started: float) -> bool:
        """Wait before repeating an undelivered SSH command; False once the reconnect bound is spent.

        The waits are 5, 10 and then 15 seconds, for at most ``SSH_RECONNECT_SECONDS`` after the first
        failure and never into the reserve before the session deadline. The endpoint is renewed before
        each repeat, since a restarted SSH service can return on another port.
        """
        delay = SSH_RECONNECT_DELAYS[min(attempt, len(SSH_RECONNECT_DELAYS) - 1)]
        if time.monotonic() + delay > started + SSH_RECONNECT_SECONDS:
            return False
        if self.deadline_unix is not None and time.time() + delay + self.reserve_seconds >= self.deadline_unix:
            return False
        time.sleep(delay)
        try:
            self.refresh()
        except (OSError, subprocess.SubprocessError, RuntimeError):
            pass  # The provider may not report an endpoint yet; the previous one is tried again.
        return True

    def _bounded(self, timeout: float) -> float:
        if self.deadline_unix is not None:
            timeout = min(timeout, self.deadline_unix - time.time() - self.reserve_seconds)
        _require(timeout > 0, "No allocation time remains for this transport operation and its reserve.")
        return timeout

    def controller(self, name: str, *args: str, timeout: float = 120) -> str:
        _require(name in {"status", "report", "start", "stop", "pycharm-info", "fallback"}, "Unknown controller action.")
        result = subprocess.run([str(self.controller_root / "bin" / f"vm-{name}"), *args],
                                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout,
                                env=phase3.clean_repository_environment())
        _require(result.returncode == 0, f"Controller {name} failed: {(result.stderr or result.stdout)[-2000:]}")
        return result.stdout

    def remote(self, argv: list[str], *, cwd: str | None = None, timeout: float = 60) -> str:
        clean_exec = ("import os,sys; env={k:v for k,v in os.environ.items() if not k.startswith('GIT_')}; "
                      "os.execvpe(sys.argv[1],sys.argv[1:],env)")
        command = shlex.join([self.bootstrap_python, "-c", clean_exec, *[str(value) for value in argv]])
        if cwd:
            command = f"cd {shlex.quote(_absolute(cwd))} && {command}"
        attempt, started = 0, time.monotonic()
        while True:
            result = subprocess.run(["ssh", "-F", str(self.ssh_config), "-o", "BatchMode=yes", "-o",
                                     "ConnectTimeout=20", "-o", "ServerAliveInterval=10", "-o",
                                     "ServerAliveCountMax=2", self.ssh_alias, command],
                                    stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                    timeout=self._bounded(timeout), env=phase3.clean_repository_environment())
            # A refused or timed-out connect delivered nothing, so the repeat cannot start remote work twice.
            # A connection lost after the command was sent is never repeated here.
            if (result.returncode != 255 or not SSH_CONNECT_FAILURE.search(result.stderr)
                    or not self._reconnect(attempt, started)):
                break
            attempt += 1
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

    def _copy(self, argv: list[str], timeout: float) -> None:
        """One scp of a single file; after an SSH transport failure (exit 255) it is repeated once.

        The read between the two attempts waits for the endpoint through the bounded reconnect of
        ``remote``. The repeat writes the same bytes to the same path.
        """
        for attempt in range(2):
            try:
                subprocess.run(["scp", "-q", "-F", str(self.ssh_config), "-o", "BatchMode=yes", *argv], check=True,
                               timeout=self._bounded(timeout), stdin=subprocess.DEVNULL,
                               env=phase3.clean_repository_environment())
                return
            except subprocess.CalledProcessError as exc:
                if attempt or exc.returncode != 255:
                    raise
                self.remote([self.bootstrap_python, "-c", "print('ready')"], timeout=15)

    def upload(self, local: Path, remote: str, *, timeout: float = 600) -> None:
        remote = _absolute(remote)
        self.remote(["mkdir", "-p", str(PurePosixPath(remote).parent)])
        self._copy([str(local), f"{self.ssh_alias}:{remote}"], timeout)

    def fetch_official_weight(self, url: str, partial: str) -> dict[str, Any]:
        """A terminal result permits upload fallback; an ambiguous SSH failure does not."""
        _require(url in OFFICIAL_ESM1B_WEIGHTS, "Remote weight URL is not approved.")
        expected = OFFICIAL_ESM1B_WEIGHTS[url]
        timeout = self._bounded(600)
        _require(timeout > 5, "No allocation time remains for a bounded public weight download.")
        program = OFFICIAL_WEIGHT_PROGRAM
        result = json.loads(self.remote([self.bootstrap_python, "-c", program, url, _absolute(partial),
                            expected["sha256"], str(expected["size_bytes"]), str(timeout - 3)], timeout=timeout))
        _require(result.get("terminal") is True and result.get("status") in {"verified", "failed"},
                 "Public weight download has no confirmed terminal result; upload fallback is unsafe.")
        return result

    def download(self, remote: str, local: Path, *, timeout: float = 120) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        self._copy([f"{self.ssh_alias}:{_absolute(remote)}", str(local)], timeout)

    def download_files(self, remote_root: str, local_root: Path, files: list[str], *, timeout: float) -> None:
        listing = local_root / ".pending_files"
        listing.write_text("".join(_relative(name) + "\n" for name in files))
        ssh = shlex.join(["ssh", "-F", str(self.ssh_config), "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"])
        subprocess.run(["rsync", "-a", "-s", "--timeout=45", f"--files-from={listing}", "-e", ssh,
                        f"{self.ssh_alias}:{_absolute(remote_root)}/", str(local_root) + "/"],
                       check=True, timeout=self._bounded(timeout), stdin=subprocess.DEVNULL,
                       env=phase3.clean_repository_environment())

    def launch(self, argv: list[str], *, cwd: str, job_dir: str) -> None:
        # An immutable job directory makes loss of the launch SSH response safe to reconcile.
        worker = ("import json,os,pathlib,subprocess,sys,time; d=pathlib.Path(sys.argv[1]); "
                  "a=json.loads(sys.argv[2]); c=sys.argv[3]; "
                  "env={k:v for k,v in os.environ.items() if not k.startswith('GIT_')}; "
                  "p=subprocess.run(a,cwd=c,env=env); q=d/'result.partial'; "
                  "q.write_text(json.dumps({'returncode':p.returncode,'finished_unix':time.time()})); "
                  "q.replace(d/'result.json')")
        launcher = ("import json,os,pathlib,subprocess,sys; d=pathlib.Path(sys.argv[1]); "
                    "d.parent.mkdir(parents=True,exist_ok=True); "
                    "\ntry: d.mkdir()\nexcept FileExistsError: sys.exit(0)\n"
                    "a=json.loads(sys.argv[2]); c=sys.argv[3]; "
                    "env={k:v for k,v in os.environ.items() if not k.startswith('GIT_')}; "
                    "(d/'command.json').write_text(json.dumps({'argv':a,'cwd':c})); "
                    "f=(d/'stdout.log').open('xb'); "
                    "p=subprocess.Popen([sys.executable,'-c',sys.argv[4],str(d),json.dumps(a),c], "
                    "stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env); "
                    "(d/'pid').write_text(str(p.pid))")
        self.remote([self.bootstrap_python, "-c", launcher, _absolute(job_dir), json.dumps(argv),
                     _absolute(cwd), worker])


class HostPull:
    """Independent readback; conservative transfer reservation survives retries and restarts."""

    def __init__(self, transport: ControllerTransport, durable_root: Path, state_root: Path, *,
                 max_download_bytes: int, deadline_unix: float, mirror_root: Path | None = None):
        self.transport, self.root, self.state_root = transport, Path(durable_root).resolve(), Path(state_root).resolve()
        self.max_bytes, self.deadline = max_download_bytes, deadline_unix
        _require(max_download_bytes > 0, "A positive host-download ceiling is required.")
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.budget_path = self.state_root / "transfer_budget.json"
        self.mirror_root = Path(mirror_root).resolve() if mirror_root is not None else None
        if self.mirror_root is not None:
            _require(self.mirror_root != self.root and self.root not in self.mirror_root.parents
                     and self.mirror_root not in self.root.parents, "Persistence destinations must be separate.")
            self.mirror_root.mkdir(parents=True, exist_ok=True)

    def _mirror(self, relative: str) -> None:
        if self.mirror_root is None:
            return
        relative = _relative(relative)
        source, target = self.root / relative, self.mirror_root / relative
        _require(self.mirror_root in target.resolve().parents and not target.is_symlink(), "Mirror path escapes its destination.")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".partial")
        with source.open("rb") as reader, temporary.open("wb") as writer:
            shutil.copyfileobj(reader, writer)
            writer.flush()
            os.fsync(writer.fileno())
        _require(_sha(temporary) == _sha(source), "NVMe mirror readback differs from independent HostPull.")
        temporary.replace(target)

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
                    # A read-only read waits for the endpoint (bounded reconnect) before the charged repeat.
                    self.transport.read_json(f"{execution_root}/execution_state.json", optional=True)
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
        # A remote acknowledgment is released only after both local copies read back.
        for entry in manifest["files"]:
            self._mirror(entry["path"])
        self._mirror(relative)
        self._mirror(ack_relative)
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
        self._mirror(relative)
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
                # One further read-only read; it cannot launch or restart a worker. A refused connect has
                # already been repeated inside the transport for its bounded reconnect time.
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
    zones = {"us-central1-a", *_fallback_policy(config).get("allowed_zones", [])}
    region = controller.get("ZONE", "").rsplit("-", 1)[0]
    _require(controller.get("MACHINE_TYPE") == "g2-standard-8" and controller.get("ZONE") in zones
             and controller.get("REGION", "us-central1") == region
             and controller.get("PROVISIONING_MODEL", "STANDARD") == "STANDARD"
             and controller.get("SSH_HOST_ALIAS") == config["ssh_alias"], "G2 hardware, region or alias differs.")
    budget = _cross_region_budget(config)
    if budget:
        profile = budget["region_profiles"][region]
        expected_rate = "PRICE_EXPECTED_COMPUTE_USD_PER_HOUR"
        _require(all(controller.get(key) == str(value) for key, value in profile.items() if key != expected_rate)
                 and math.isclose(float(controller.get(expected_rate, "nan")), float(profile[expected_rate]),
                                  rel_tol=1e-6, abs_tol=1e-9),
                 "Controller region/subnet/pricing profile differs from the reviewed recovery budget.")


def _fallback_policy(config: Mapping[str, Any]) -> Mapping[str, Any]:
    policy = config.get("fallback", {})
    _require(isinstance(policy, Mapping), "Fallback authority must be an explicit configuration object.")
    if policy:
        zones = policy.get("allowed_zones")
        _require(isinstance(zones, list) and zones and all(isinstance(zone, str) for zone in zones)
                 and len(zones) == len(set(zones)), "Fallback requires distinct explicit destination zones.")
        if policy.get("scope") == "cross-region-v1":
            _require(tuple(zones) == CROSS_REGION_ZONES, "Fallback zones/order exceed the authorized cross-region pass.")
        else:
            _require(policy.get("scope") in (None, "same-region-v1")
                     and set(zones) <= {"us-central1-b", "us-central1-c"},
                     "Fallback zones exceed the authorized same-region pass.")
    return policy


def _cross_region_budget(config: Mapping[str, Any]) -> Mapping[str, Any]:
    policy = _fallback_policy(config)
    if policy.get("scope") != "cross-region-v1":
        return {}
    reference = policy["recovery_budget"]
    path = Path(_absolute(reference["path"]))
    _require(path.is_file() and _sha(path) == reference["sha256"], "Recovery budget checksum differs.")
    budget = _json(path)
    _require(budget.get("schema_version") == 1 and budget.get("scope") == "cross-region-v1"
             and budget.get("allowed_zones") == policy["allowed_zones"], "Recovery budget scope differs.")
    _require(0 < _number(budget["gross_ceiling_usd"], "cross-region gross ceiling") <= CROSS_REGION_CEILING_USD,
             "Cross-region recovery exceeds the authorized $19 ceiling.")
    _require(0 <= time.time() - _epoch(budget["verified_at"]) <= 86400,
             "Cross-region recovery prices are stale or future-dated.")
    _require(budget["max_storage_seconds"] == policy["storage_window_seconds"] == 14400,
             "Cross-region recovery requires its bounded four-hour storage window.")
    _require(budget["host_egress_reserve_usd"] >= EGRESS_RESERVE_GIB * config["pricing"]["egress_usd_per_gib"]
             and budget["retained_snapshot_usd_per_hour"] >= config["pricing"]["snapshot_usd_per_hour"],
             "Recovery budget omits host egress or retained snapshots.")
    for key in ("host_egress_reserve_usd", "retained_snapshot_usd_per_hour", "snapshot_usd_per_hour", "recycle_usd_per_hour"):
        _number(budget[key], key)
    _require(_number(budget["transfer_usd_per_gib"], "snapshot restore transfer rate") >= .02
             and type(budget["transfer_bytes_per_restore"]) is int
             and budget["transfer_bytes_per_restore"] >= 150 * GIB
             and budget["transfer_source_url"] == "https://cloud.google.com/compute/disks-image-pricing",
             "Recovery budget omits the conservative cross-region snapshot restore charge.")
    regions = {"us-central1", "us-west1", "us-east4"}
    _require(set(budget["region_profiles"]) == set(budget["region_prices"]) == regions,
             "Recovery budget must cover every authorized region.")
    keys = {"REGION", "SUBNET", "SUBNET_RANGE", "PRICE_COMPUTE_SKUS", "PRICE_DISK_CAPACITY_SKU_ID",
            "PRICE_EXPECTED_COMPUTE_USD_PER_HOUR"}
    for region in regions:
        profile, price = budget["region_profiles"][region], budget["region_prices"][region]
        _require(set(profile) == keys and profile["REGION"] == region and profile["SUBNET"] == "deepmzyme-subnet",
                 "Recovery budget region profile is incomplete or foreign.")
        _require(0 <= time.time() - _epoch(price["verified_at"]) <= 86400,
                 "Regional price evidence is stale or future-dated.")
        for key in ("gross_usd_per_hour", "disk_usd_per_hour"):
            _number(price[key], key)
    return budget


def _verify_recovery_envelope(config: Mapping[str, Any], envelope: Mapping[str, Any]) -> None:
    policy, budget = _fallback_policy(config), _cross_region_budget(config)
    _require(budget and envelope.get("scope") == "cross-region-v1"
             and envelope.get("budget_sha256") == policy["recovery_budget"]["sha256"]
             and envelope.get("allowed") is True and envelope.get("allowed_zones") == policy["allowed_zones"]
             and envelope.get("max_attempts") == 5 and envelope.get("attempt_window_seconds") == 3600
             and envelope.get("max_storage_seconds") == budget["max_storage_seconds"],
             "Controller recovery envelope differs from the reviewed pass.")
    minimum_transfer = budget["transfer_bytes_per_restore"] / GIB * budget["transfer_usd_per_gib"] * 5
    _require(_number(envelope["transfer_reserve_usd"], "controller transfer reserve") >= minimum_transfer
             and envelope["gross_ceiling_usd"] == budget["gross_ceiling_usd"]
             and _number(envelope["worst_case_gross_usd"], "controller recovery envelope") <= budget["gross_ceiling_usd"],
             "Controller recovery envelope omits transfer charges or exceeds the authorized ceiling.")
    _require(set(envelope["region_prices"]) == set(budget["region_prices"]), "Controller recovery price regions differ.")
    for region, actual in envelope["region_prices"].items():
        for key in ("gross_usd_per_hour", "disk_usd_per_hour"):
            _require(0 < _number(actual[key], key) <= budget["region_prices"][region][key] + 1e-9,
                     "Controller regional prices exceed the reviewed recovery budget.")


def _fallback_cost(config: Mapping[str, Any]) -> float:
    policy = _fallback_policy(config)
    if not policy:
        return 0.0
    proof = policy["storage_cost_evidence"]
    _require(0 <= time.time() - _epoch(proof["verified_at"]) <= 86400, "Fallback storage prices are stale or future-dated.")
    window = _number(policy["storage_window_seconds"], "fallback storage window")
    _require(window == 14400, "Fallback reserves one hour of preparation, two hours running and one hour of cleanup.")
    # The base envelope charges one disk and old snapshots for two hours. Cover
    # every candidate/source disk throughout recovery and the additional storage time.
    hours = window / 3600
    disk = _number(proof["disk_usd_per_hour"], "fallback disk rate")
    budget = _cross_region_budget(config)
    if budget:
        _require(disk >= max(row["disk_usd_per_hour"] for row in budget["region_prices"].values()),
                 "Fallback storage reserve omits a destination region's disk rate.")
        _require(proof["snapshot_usd_per_hour"] >= budget["snapshot_usd_per_hour"]
                 and proof["recycle_minimum_usd"] >= budget["recycle_usd_per_hour"],
                 "Fallback storage evidence omits snapshot or recycle-bin charges.")
    required = ((len(policy["allowed_zones"]) + 1) * hours - 2) * disk
    required += hours * _number(proof["snapshot_usd_per_hour"], "fallback snapshot rate")
    retained = budget["retained_snapshot_usd_per_hour"] if budget else config["pricing"]["snapshot_usd_per_hour"]
    required += (hours - 2) * _number(retained, "retained snapshot rate")
    required += _number(proof["recycle_minimum_usd"], "recycled snapshot minimum")
    reserved = _number(policy["storage_reserve_usd"], "fallback storage reserve")
    _require(reserved >= required, "Fallback storage reserve omits overlapping disks or snapshot minimums.")
    return reserved


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
    region = config["expected_identity"]["zone"].rsplit("-", 1)[0]
    _require(price.get("machine_type") == "g2-standard-8" and price.get("region") == region,
             "Controller price is for different hardware or region.")
    gross = _number(price["gross_usd_per_hour"], "gross rate")
    snapshots = _number(quote["snapshot_usd_per_hour"], "retained snapshot rate")
    egress = _number(quote["egress_usd_per_gib"], "egress rate")
    _require(egress >= .15, "The recorded egress rate cannot omit the destination-region charge.")
    recovery = _fallback_cost(config)
    budget = _cross_region_budget(config)
    transfer, budget_gross, maximum = 0.0, gross, GROSS_CEILING_USD
    host_egress = egress * EGRESS_RESERVE_GIB
    if budget:
        _require(gross <= budget["region_prices"][region]["gross_usd_per_hour"] + 1e-9,
                 "Actual regional compute price exceeds the reviewed recovery budget.")
        budget_gross = max(row["gross_usd_per_hour"] for row in budget["region_prices"].values())
        transfer = budget["transfer_bytes_per_restore"] / GIB * budget["transfer_usd_per_gib"] * len(budget["allowed_zones"])
        snapshots = budget["retained_snapshot_usd_per_hour"]
        host_egress = budget["host_egress_reserve_usd"]
        maximum = budget["gross_ceiling_usd"]
    ceiling = budget_gross * 2 * 1.15 + snapshots * 2 + host_egress + recovery + transfer
    _require(ceiling <= maximum,
             f"Compute, storage, egress and recovery require ${ceiling:.6f}, exceeding the ${maximum:g} gross ceiling.")
    return {"gross_usd_per_hour": gross, "snapshot_usd_per_hour": snapshots,
            "egress_usd_per_gib": egress, "recovery_storage_reserve_usd": recovery,
            "recovery_transfer_reserve_usd": transfer, "gross_ceiling_usd": maximum, "conservative_max_usd": ceiling}


def _view_archive_bytes(path: Path, fold: int = 0) -> int:
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
            _require(manifest.get("format") == (phase3.TRANSFER_FORMAT if fold == 0 else phase3.FOLD_TRANSFER_FORMAT)
                     and manifest.get("campaign_id") == phase3.CAMPAIGN_ID and manifest.get("fold") == fold and manifest.get("smoke") is False, "Invalid Fold-0 transfer header.")
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


def _validate_remote_campaign(config: Mapping[str, Any]) -> None:
    root = config.get("remote_campaign")
    _require(isinstance(root, str), "An explicit remote_campaign path is required.")
    _absolute(root)
    _require(PurePosixPath(root).name == phase3.CAMPAIGN_ID,
             f"remote_campaign must end with {phase3.CAMPAIGN_ID!r} for the frozen view location rule.")


def _validate_runpod_structure(config: Mapping[str, Any]) -> None:
    """Reject incomplete runtime references before reading large archives."""
    runtimes = config.get("runtime_environments", {})
    _require(isinstance(runtimes, dict) and set(runtimes) == {"fit", "esm1b"}, "Both runtime locks are required.")
    stage = config.get("stage", [])
    _require(isinstance(stage, list) and stage, "A nonempty staging closure is required.")
    for key, runtime in runtimes.items():
        _require(isinstance(runtime, dict) and isinstance(runtime.get("archive"), str)
                 and isinstance(runtime.get("packages"), dict) and runtime["packages"]
                 and runtime.get("python_version") and runtime.get("torch_cuda")
                 and re.fullmatch(r"[0-9a-f]{64}", str(runtime.get("archive_sha256", "")))
                 and any(item.get("remote") == runtime["archive"]
                         and item.get("sha256") == runtime["archive_sha256"]
                         and item.get("conda_unpack") is True for item in stage),
                 f"The {key} runtime lacks a locked, relocatable archive.")
    _validate_remote_campaign(config)
    if config.get("artifact_cache"):
        from ec_clean30 import runpod_bootstrap
        runpod_bootstrap.validate_cache_config(config)
    keep_alive = config.get('runpod', {}).get('keep_alive_for_operator', False)
    _require(type(keep_alive) is bool, 'keep_alive_for_operator must be a boolean.')
    if keep_alive:
        _operator_root(config, prepare=True)


def _validate_published_runtime_locks(config: Mapping[str, Any], source: Mapping[str, Any]) -> None:
    """Bind execution environments to the complete manifest-authenticated runtime recipes."""
    if source.get("format") not in {phase3.BOOTSTRAP_FORMAT, phase3.EXECUTION_V5_FORMAT}:
        return
    binding = source.get("runtime_locks") if source.get("format") == phase3.EXECUTION_V5_FORMAT else {
        "path": PUBLISHED_RUNTIME_LOCKS_PATH, "sha256": PUBLISHED_RUNTIME_LOCKS_SHA256}
    _require(isinstance(binding, dict) and set(binding) == {"path", "sha256"}
             and re.fullmatch(r"[0-9a-f]{64}", str(binding.get("sha256", ""))),
             "The execution manifest must bind its complete published runtime lock document.")
    path = phase3.REPO_ROOT / _relative(binding["path"])
    _require(path.is_file() and not path.is_symlink() and path.resolve() == path
             and _sha(path) == binding["sha256"],
             "The published runtime lock document is missing or changed.")
    document = _json(path)
    _require(document.get("format") == "deepmzyme_ec_clean30_runtime_locks_v1"
             and set(document.get("runtimes", {})) == {"fit", "esm1b"},
             "The published runtime lock document has an unexpected format.")
    runtimes = config.get("runtime_environments", {})
    _require(isinstance(runtimes, dict) and set(runtimes) == {"fit", "esm1b"},
             "Both published runtime recipes are required.")
    for key, expected in document["runtimes"].items():
        runtime = runtimes[key]
        _require(isinstance(runtime, dict)
                 and all(runtime.get(field) == expected[field] for field in
                         ("packages", "python_version", "torch_cuda", "archive_sha256", "bytes")),
                 f"The {key} runtime differs from its complete published recipe.")
        staged = [item for item in config.get("stage", []) if item.get("remote") == runtime.get("archive")]
        _require(len(staged) == 1 and staged[0].get("sha256") == expected["archive_sha256"]
                 and staged[0].get("conda_unpack") is True,
                 f"The {key} staging archive differs from its published runtime recipe.")
        local = Path(staged[0]["local"])
        _require(local.is_file() and not local.is_symlink() and local.stat().st_size == expected["bytes"],
                 f"The {key} archive size differs from its published runtime recipe.")


def _local_storage(config: Mapping[str, Any]) -> None:
    payload_limit = (_g2_download_limit(config, {"format": phase3.EXECUTION_V5_FORMAT})
                     if config.get("execution_scope") in {"g2_then_fold0_v5", "fold_execution_v5"} else PAYLOAD_CEILING_BYTES)
    secondary = Path(config["secondary_drive"]).resolve()
    _require(secondary.is_mount(), "The independent secondary drive is not mounted.")
    for key in ("durable_root", "state_root"):
        _require(secondary in Path(_absolute(config[key])).resolve().parents,
                 f"{key} must be on the independent secondary drive.")
    _require(shutil.disk_usage(secondary).free >= 5 * GIB + payload_limit,
             "The independent drive lacks the 5 GiB reserve plus the download allowance.")
    _require(shutil.disk_usage(phase3.REPO_ROOT).free >= 5 * GIB, "Internal ext4 is below its 5 GiB reserve.")
    if config.get("provider") == "runpod":
        ancestor = Path(_absolute(config["mirror_root"])).resolve()
        while not ancestor.exists():
            ancestor = ancestor.parent
        _require(ancestor.stat().st_dev != secondary.stat().st_dev
                 and shutil.disk_usage(ancestor).free >= 5 * GIB + payload_limit,
                 "Runpod requires an independent NVMe mirror with the payload allowance and reserve.")


def _posix_storage_plan(config, unpacked_bytes, view_bytes):
    root_value = config.get("runpod", {}).get("posix_root")
    if root_value is None:
        return None
    from ec_clean30 import runpod_bootstrap
    _require(config.get("artifact_cache"), "Local POSIX preparation requires the authenticated cache bootstrap.")
    root = PurePosixPath(_absolute(root_value))
    campaign = PurePosixPath(_absolute(config["remote_campaign"]))
    prefixes, components = [], {"views": view_bytes}
    for key in ("fit", "esm1b"):
        archive = config["runtime_environments"][key]["archive"]
        item = next(row for row in config["stage"] if row["remote"] == archive)
        prefix = PurePosixPath(_absolute(item["extract_to"]))
        _require(prefix.parent == root, "Both runtime prefixes must be direct children of runpod.posix_root.")
        python = config["remote_python"] if key == "fit" else config["embedding"]["esm1b"]["clean_python"]
        _require(python == str(prefix / "bin/python"), "Runtime interpreter differs from its local extraction prefix.")
        prefixes.append(str(prefix))
        components[prefix.name] = unpacked_bytes[item["remote"]]
    for item in config["stage"]:
        _require(PurePosixPath(item["remote"]).is_relative_to(campaign),
                 "POSIX staging keeps bulk artifacts beneath the nonsymlink campaign root.")
        destinations = [item["remote"]] + ([item["extract_to"]] if item["kind"] == "tar" else [])
        _require(not any(PurePosixPath(path).is_relative_to(campaign / name)
                         for path in destinations for name in ("views", "g2_requests")),
                 "Bulk staging cannot use the protected view or request aliases.")
        if item["kind"] == "tar" and PurePosixPath(item["extract_to"]).is_relative_to(root):
            _require(item["extract_to"] in prefixes, "Unbudgeted extraction beneath the local POSIX root.")
    request = {"format": "deepmzyme_ec_clean30_posix_storage_v1", "campaign_root": str(campaign),
               "local_root": str(root), "runtime_prefixes": prefixes, "component_bytes": components,
               "required_free_bytes": math.ceil(sum(components.values()) * 1.25) + 5 * GIB}
    runpod_bootstrap.validate_posix_storage_request(request)
    return request


def _preparation_identity(config: Mapping[str, Any]) -> str:
    value = copy.deepcopy(config)
    if value.get("provider") == "runpod":
        settings = value["runpod"]
        settings.pop("data_center_id", None)
        for name in ("observed_at", "source"):
            settings["quote"].pop(name, None)
        for name in PASS_LOCAL_RUNPOD_FIELDS:
            settings.pop(name, None)
        for name in PASS_LOCAL_FIELDS:
            value.pop(name, None)
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _prepared_preflight(config: Mapping[str, Any], config_path: Path, *, published: bool = False,
                        fresh: bool = True) -> dict[str, Any]:
    """Reuse a completed input proof only for explicit quote/placement changes."""
    if config.get("provider") != "runpod" or not config.get("artifact_cache"):
        return preflight(config, config_path, published=published, fresh=fresh)
    _validate_runpod_structure(config)
    from ec_clean30 import runpod_controller
    cost = runpod_controller.validate_config(config, fresh=fresh)
    source = _execution_source(config, published=published)
    _validate_published_runtime_locks(config, source)
    root = Path(config["state_root"])
    proof_path, baseline_path = root / "prepared_inputs.json", root / "prepared_config.json"
    if proof_path.exists() and baseline_path.exists():
        original, baseline = _json(proof_path), _json(baseline_path)
        if (original.get("status") == "ready" and original.get("source") == source
                and original.get("config_sha256") == _sha(baseline_path)
                and _preparation_identity(config) == _preparation_identity(baseline)):
            _local_storage(config)
            result = {**original, "config_sha256": _sha(config_path), "cost": cost,
                      "input_proof_reused": True, "original_input_proof_sha256": _sha(proof_path),
                      "allowed_metadata_changes": ["runpod.data_center_id", "runpod.quote.observed_at", "runpod.quote.source",
                                                   *(f"runpod.{name}" for name in PASS_LOCAL_RUNPOD_FIELDS),
                                                   *PASS_LOCAL_FIELDS]}
            _write(root / "readiness.json", result)
            return result
    result = preflight(config, config_path, published=published, fresh=fresh)
    result.update(config_sha256=_sha(config_path), input_proof_reused=False)
    # Preserve the exact original configuration bytes rather than reserializing its hash binding.
    root.mkdir(parents=True, exist_ok=True)
    baseline_path.write_bytes(config_path.read_bytes())
    _write(proof_path, result)
    _write(root / "readiness.json", result)
    return result


def preflight(config: Mapping[str, Any], config_path: Path, *, published: bool = False,
              fresh: bool = True) -> dict[str, Any]:
    """Local-only readiness; no controller or SSH call and no cloud allocation."""
    runpod = config.get("provider", "gcp") == "runpod"
    _require(config.get("provider", "gcp") in {"gcp", "runpod"}, "Unsupported execution provider.")
    _require(fresh is True or runpod, "Only a recovering Runpod launch window may relax quote freshness.")
    if runpod:
        from ec_clean30 import runpod_controller
        _validate_runpod_structure(config)
        cost = runpod_controller.validate_config(config, fresh=fresh)
        repo_url = config["repo_url"]
        _require(config["hardware_profile"] in {"runpod_l4", "runpod_rtx4090"}, "Explicit Runpod hardware profile required.")
    else:
        _validate_remote_campaign(config)
        controller = Path(config["controller_root"]).resolve()
        controller_config = _controller_config(controller)
        _identity(config, controller_config)
        cost = _cost(config)  # Refuse an unaffordable recovery before hashing multi-gigabyte inputs.
        repo_url = controller_config["REPO_URL"]
        for name in ("status", "report", "start", "stop", "pycharm-info"):
            _require(os.access(controller / "bin" / f"vm-{name}", os.X_OK), f"Controller vm-{name} is unavailable.")
        if _fallback_policy(config):
            _require(os.access(controller / "bin/vm-fallback", os.X_OK), "Controller vm-fallback is unavailable.")
    _require(repo_url.startswith("https://github.com/"), "Checkout URL must use canonical HTTPS.")
    origin = subprocess.check_output(["git", "-C", str(phase3.REPO_ROOT), "config", "--get", "remote.origin.url"],
                                     text=True, env=phase3.clean_repository_environment()).strip()
    _require(_repository_key(origin) == _repository_key(repo_url), "Local/controller repository identities differ.")
    source = _execution_source(config, published=published)
    _validate_published_runtime_locks(config, source)
    queue = measurement_queue(source)
    v5 = source.get("format") == phase3.EXECUTION_V5_FORMAT
    _require(not v5 or runpod, "Execution v5 uses only the Runpod transport.")
    max_seconds = cost["max_total_seconds"] if v5 else MAX_G2_SECONDS
    payload_limit = _g2_download_limit(config, source)
    _local_storage(config)
    for key in ("remote_repo", "remote_python", "remote_campaign", "view_archive", "timing_inputs", "bootstrap_python"):
        _absolute(config[key])
    for command in ("ssh", "scp", "rsync"):
        _require(shutil.which(command), f"Required local transport command is unavailable: {command}")
    staged, content, staging_bytes, unpacked_bytes = set(), set(), 0, {}
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
            unpacked_bytes[remote] = 0
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
                        unpacked_bytes[remote] += member.size
    _require(config["view_archive"] in staged and any(item["remote"] == config["view_archive"]
             and item["sha256"] == config["view_archive_sha256"] for item in config["stage"]), "Fold-0 archive lacks a staging pin.")
    view_item = next(item for item in config["stage"] if item["remote"] == config["view_archive"])
    view_bytes = _view_archive_bytes(Path(view_item["local"]), config.get("execution_fold", 0))
    if view_item["kind"] == "file":
        staging_bytes += view_bytes
    _require(set(config["bounds"]) == {name for _, name in queue}, "G2 must retain its exact versioned measurement queue.")
    for name, bound in config["bounds"].items():
        _require(not v5 or set(bound) == {"forecast_seconds", "unit_timeout_seconds"},
                 "V5 timing bounds cannot override source, paths or runtime authority.")
        forecast = _number(bound["forecast_seconds"], name + " forecast")
        timeout = _number(bound["unit_timeout_seconds"], name + " timeout")
        _require(timeout <= forecast and forecast * 1.25 + RESERVE_SECONDS < max_seconds,
                 "A bounded unit cannot fit the authenticated session admission window.")
    for key in ("fit", "esm1b"):
        profile = config["profiles"][key]
        _require(not v5 or set(profile) == {"cap_prefix", "memory_limit_bytes"},
                 "V5 resource profiles may contain only their explicit cap and memory limit.")
        _require(isinstance(profile["cap_prefix"], list) and profile["cap_prefix"]
                 and all(isinstance(value, str) and value for value in profile["cap_prefix"]), "Explicit resource-cap argv required.")
        _number(profile["memory_limit_bytes"], "memory cap")
        cores, gib = (2, 8) if key == "fit" else (4, 24)
        required_cap = {"--user", "--scope", f"CPUQuota={cores * 100}%", f"MemoryMax={gib}G", "MemorySwapMax=0"}
        if runpod:
            expected = [config["remote_python"], f"{config['remote_repo']}/src/ec_clean30/container_caps.py", "run",
                        "--cgroup-root", _absolute(config["cgroup_root"]), "--cpu-cores", str(cores),
                        "--memory-bytes", str(gib * GIB), "--"]
            _require(profile["cap_prefix"] == expected and profile["memory_limit_bytes"] == gib * GIB,
                     "Runpod requires its exact cgroup CPU/RAM/no-swap cap.")
        else:
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
    if runpod:
        required.append(config["remote_python"])
        _require(set(config["runtime_environments"]) == {"fit", "esm1b"}, "Both runtime locks are required.")
        for key, runtime in config["runtime_environments"].items():
            _require(runtime["packages"] and runtime["python_version"] and runtime["torch_cuda"]
                     and any(item["remote"] == runtime["archive"] and item["sha256"] == runtime["archive_sha256"]
                             and item.get("conda_unpack") is True for item in config["stage"]),
                     f"The {key} runtime lacks a locked, relocatable archive.")
    _require(all(path in content for path in required), "The prepared staging closure omits an embedding input or interpreter.")
    return {"status": "ready" if source.get("published") else "draft_unpublished", "source": source,
            "posix_storage": _posix_storage_plan(config, unpacked_bytes, view_bytes) if runpod else None,
            "cost": cost, "queue": [{"action": action, "name": name} for action, name in queue],
            "staging_peak_bytes": staging_bytes, "pending_runtime_checks": ["VM free space", "both CUDA interpreters", "source and view probe"],
            "max_provider_seconds": max_seconds, "payload_ceiling_bytes": payload_limit,
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
    if config.get("fallback_started_unix"):
        _require(time.time() + RESERVE_SECONDS < config["fallback_started_unix"]
                 + config["fallback"]["storage_window_seconds"], "Fallback storage/cleanup window is exhausted.")
    return {"session_id": current["session_id"], "allocation_started_unix": started, "deadline_unix": deadline,
            "max_total_seconds": min(MAX_G2_SECONDS, deadline - started)}


def _stage_file(config: Mapping[str, Any], transport: ControllerTransport, item: Mapping[str, Any]) -> None:
    if config.get("provider") == "runpod" and config.get("artifact_cache"):
        from ec_clean30 import runpod_bootstrap
        receipt = runpod_bootstrap.hydrate_cached_artifact(config, transport, item, transport.deadline_unix)
        if receipt is not None:
            _require(transport.stat_file(item["remote"]) == {"sha256": item["sha256"],
                     "size_bytes": Path(item["local"]).stat().st_size}, "Cached artifact readback differs.")
            _write(Path(config["state_root"]) / "staging_receipts" / f"{item['sha256']}.json", receipt)
            return
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
                _write(Path(config["state_root"]) / "staging_receipts" / f"official_{item['sha256']}.json",
                       {"url": url, "sha256": item["sha256"], "size_bytes": expected["size_bytes"],
                        "tls": result.get("tls", "verified"), "content_pinned": True})
            elif config.get("provider") == "runpod" and config.get("artifact_cache"):
                raise WorkstationError("Official model download failed; a bulk workstation fallback is not enabled for the cached route.")
        if downloaded != expected:
            transport.upload(Path(item["local"]), partial, timeout=MAX_G2_SECONDS)
            existing = transport.stat_file(partial)
        else:
            existing = downloaded
        _require(existing == expected, "Staged upload/download checksum or size differs.")
        transport.remote(["mv", "--", partial, item["remote"]])
    _require(existing == expected, "Remote staging checksum or size differs.")


def _stage_checkout(config: Mapping[str, Any], transport: ControllerTransport, source: Mapping[str, Any]) -> None:
    origin = config["repo_url"] if config.get("provider") == "runpod" else _controller_config(transport.controller_root)["REPO_URL"]
    checkout = ("import os,pathlib,subprocess,sys; p=pathlib.Path(sys.argv[1]); "
                "env={k:v for k,v in os.environ.items() if not k.startswith('GIT_')}; "
                "\nif not p.exists(): subprocess.run(['git','clone','--filter=blob:none',sys.argv[2],str(p)],check=True,env=env)\n"
                "assert not subprocess.check_output(['git','-C',str(p),'status','--porcelain'],text=True,env=env).strip(), 'Remote checkout is dirty'; "
                "subprocess.run(['git','-C',str(p),'fetch','origin','main:refs/remotes/origin/main'],check=True,env=env); "
                "subprocess.run(['git','-C',str(p),'checkout','--detach',sys.argv[3]],check=True,env=env)")
    transport.remote([config["bootstrap_python"], "-c", checkout, config["remote_repo"], origin, source["git_head"]], timeout=300)


def _unpack_stage(config: Mapping[str, Any], transport: ControllerTransport, item: Mapping[str, Any]) -> None:
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


def _stage_runtime(config: Mapping[str, Any], transport: ControllerTransport, source: Mapping[str, Any]) -> None:
    _stage_checkout(config, transport, source)
    for item in config["stage"]:
        _stage_file(config, transport, item)
        _unpack_stage(config, transport, item)


def _bootstrap_runpod(config, transport, source, session):
    """Prove kernel authority using only the pinned helper before bulk transfer."""
    from ec_clean30 import container_caps
    root = f"/workspace/.deepmzyme-bootstrap/{session['session_id']}"
    local = phase3.REPO_ROOT / "src/ec_clean30/container_caps.py"
    _require(_sha(local) == source["source_sha256"]["src/ec_clean30/container_caps.py"],
             "Bootstrap helper differs from its published pin.")
    remote = root + "/container_caps.py"
    transport.upload(local, remote)
    _require(transport.stat_file(remote) == {"sha256": _sha(local), "size_bytes": local.stat().st_size},
             "Bootstrap helper transfer differs.")
    def host_step(argv, timeout):
        try:
            return json.loads(transport.remote(argv, timeout=timeout))
        except WorkstationError as exc:
            if str(exc).startswith(HOST_REJECTION_PREFIX):
                raise HostRejected(str(exc)) from None
            raise

    prepared = host_step([config["bootstrap_python"], remote, "prepare", "--cgroup-root",
                          config["cgroup_root"], "--owner-id", session["pod_id"]], 60)
    _require(prepared.get("parent_limits_unchanged") is True, "Bootstrap changed provider limits.")
    _write(Path(config["state_root"]) / "bootstrap_prepare.json", prepared)
    evidence = host_step([config["bootstrap_python"], remote, "probe", "--cgroup-root", config["cgroup_root"]], 90)
    try:
        container_caps.validate_probe_evidence(evidence)
    except container_caps.CapError as exc:
        raise HostRejected(f"container caps evidence refused: {exc}") from None
    _require(evidence["helper_sha256"] == source["source_sha256"]["src/ec_clean30/container_caps.py"],
             "Early kernel proof belongs to another helper.")
    _write(Path(config["state_root"]) / "bootstrap_kernel.json", evidence)
    tools = json.loads(transport.remote([config["bootstrap_python"], "-c",
        "import json,shutil,tarfile;print(json.dumps({'missing':[n for n in ('git','curl','rsync') if not shutil.which(n)],"
        "'safe_tar':hasattr(tarfile,'data_filter')}))"], timeout=20))
    _require(tools["safe_tar"], "Bootstrap Python lacks safe archive extraction.")
    if tools["missing"]:
        _require(set(tools["missing"]) <= {"git", "curl", "rsync"}, "Unexpected system prerequisite.")
        transport.remote(["env", "DEBIAN_FRONTEND=noninteractive", "apt-get", "update"], timeout=180)
        transport.remote(["env", "DEBIAN_FRONTEND=noninteractive", "apt-get", "install", "-y",
                          "--no-install-recommends", *tools["missing"]], timeout=180)
    _write(Path(config["state_root"]) / "bootstrap_tools.json", tools)
    return evidence


def _prepare_runpod_storage(config, transport, ready, session):
    request = ready.get("posix_storage")
    if request is None:
        _require(config.get("runpod", {}).get("posix_root") is None,
                 "Local POSIX storage lacks its prebilling capacity plan.")
        return
    from ec_clean30 import runpod_bootstrap
    runpod_bootstrap.validate_posix_storage_request(request)
    helper = phase3.REPO_ROOT / "src/ec_clean30/runpod_bootstrap.py"
    digest = _sha(helper)
    _require(digest == ready["source"]["source_sha256"]["src/ec_clean30/runpod_bootstrap.py"],
             "POSIX preparation helper differs from its published pin.")
    root = f"/workspace/.deepmzyme-bootstrap/{session['session_id']}"
    remote_helper = f"{root}/{digest}.py"
    transport.upload(helper, remote_helper)
    _require(transport.stat_file(remote_helper) == {"sha256": digest, "size_bytes": helper.stat().st_size},
             "POSIX preparation helper transfer differs.")
    local = Path(config["state_root"]) / "posix_storage_request.json"
    _write(local, request)
    remote_request = f"{root}/posix_storage_request.json"
    transport.upload(local, remote_request)
    result = json.loads(transport.remote([config["bootstrap_python"], remote_helper,
                        "prepare-posix-storage", "--request", remote_request], timeout=90))
    _require(result.get("status") == "passed" and result.get("helper_sha256") == digest
             and all(result.get(key) == value for key, value in request.items()),
             "POSIX preparation lacks a successful source-bound receipt.")
    _write(Path(config["state_root"]) / "posix_storage.json", result)


def _stage_runpod_runtime(config, transport, source, check_session):
    """Main runtime first, followed by at most two independent staging writers."""
    from ec_clean30 import runpod_bootstrap
    _stage_checkout(config, transport, source)
    main = next(item for item in config["stage"] if item["remote"] == config["runtime_environments"]["fit"]["archive"])

    def stage(item):
        check_session()
        if not runpod_bootstrap.hydrate_public_artifact(config, transport, item, source, transport.deadline_unix):
            _stage_file(config, transport, item)
        _unpack_stage(config, transport, item)
        check_session()

    begun = time.time()
    stage(main)
    _container_probe(config, transport)
    _verify_runtime_lock(config, transport, "fit", config["remote_python"])
    cuda = json.loads(transport.remote([*config["profiles"]["fit"]["cap_prefix"], config["remote_python"], "-c",
                      phase3.cuda_probe_program(config["hardware_profile"])], cwd=config["remote_repo"], timeout=90))
    phase3.validate_cuda_evidence(cuda, config["hardware_profile"])
    _write(Path(config["state_root"]) / "cuda_fit_early.json", cuda)
    check_session()
    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="runpod-staging")
    pending = {pool.submit(stage, item) for item in config["stage"] if item is not main}
    try:
        while pending:
            check_session()
            done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                future.result()
    finally:
        pool.shutdown(wait=config.get('runpod', {}).get('keep_alive_for_operator') is True,
                      cancel_futures=True)
    _write(Path(config["state_root"]) / "staging_complete.json", {"status": "passed", "seconds": time.time() - begun,
            "workers": 2, "source": source, "main_runtime_first": True})


def _container_probe(config: Mapping[str, Any], transport: ControllerTransport, *, early: bool = False) -> str:
    from ec_clean30 import container_caps
    python = config["bootstrap_python"] if early else config["remote_python"]
    local = Path(config["state_root"]) / ("container_probe_early.json" if early else "container_probe.json")
    if early:
        inspect = ("import json,os,pathlib,sys; p=pathlib.Path(sys.argv[1]); "
                   "print(json.dumps({'mountinfo':[s for s in pathlib.Path('/proc/self/mountinfo').read_text().splitlines() "
                   "if ' - cgroup' in s],'membership':pathlib.Path('/proc/self/cgroup').read_text(),"
                   "'cgroup_root':str(p),'writable':os.access(p,os.W_OK),"
                   "'controls':{n:(p/n).read_text() if (p/n).is_file() else None "
                   "for n in ('cgroup.controllers','cgroup.subtree_control','cpu.max','memory.max','memory.swap.max')}}))")
        _write(local.parent / "container_observation.json", json.loads(transport.remote(
            [python, "-c", inspect, config["cgroup_root"]], timeout=30)))
    try:
        result = json.loads(transport.remote([python, f"{config['remote_repo']}/src/ec_clean30/container_caps.py",
            "probe", "--cgroup-root", config["cgroup_root"]], cwd=config["remote_repo"], timeout=90))
        container_caps.validate_probe_evidence(result)
    except BaseException as exc:
        _write(local, {"status": "failed", "error": str(exc), "cgroup_root": config["cgroup_root"]})
        raise
    _write(local, result)
    remote = f"{config['remote_campaign']}/g2_requests/{local.name}"
    transport.upload(local, remote)
    return remote


def _verify_runtime_lock(config: Mapping[str, Any], transport: ControllerTransport, key: str, python: str) -> None:
    program = ("import importlib.metadata as m,json,platform,re,torch; "
               "print(json.dumps({'python_version':platform.python_version(),'torch_cuda':torch.version.cuda,"
               "'packages':{re.sub(r'[-_.]+','-',d.metadata['Name']).lower():d.version for d in m.distributions()}}))")
    measured = json.loads(transport.remote([*config["profiles"][key]["cap_prefix"], python, "-c", program],
                                          cwd=config["remote_repo"], timeout=90))
    expected = config["runtime_environments"][key]
    _write(Path(config["state_root"]) / f"runtime_{key}.json", measured)
    _require(all(measured[field] == expected[field] for field in ("python_version", "torch_cuda", "packages")),
             f"The {key} interpreter differs from its complete published runtime lock.")


def stage_session_chain_receipt(config: Mapping[str, Any], transport: ControllerTransport,
                                source: Mapping[str, Any]) -> dict[str, Any]:
    """Ship the workstation receipt, refusing a changed local proof or a foreign remote file."""
    fields = _remote_chain_request(config, source)
    if not fields:
        return fields
    local = session_chain_receipt_path(config)
    current = _execution_source(config, published=True)
    _require(current["scientific_identity_sha256"] == source["scientific_identity_sha256"]
             and current["fit_chain"] == source["fit_chain"]
             and current["amendment_chain"] == source["amendment_chain"]
             and current["execution_manifest_sha256"] == source["execution_manifest_sha256"],
             "The source or chain changed before the session receipt was shipped.")
    expected = {"sha256": _sha(local), "size_bytes": local.stat().st_size}
    remote = fields["amendment_chain_receipt"]
    existing = transport.stat_file(remote)
    if existing is None:
        partial = remote + ".partial"
        transport.upload(local, partial)
        _require(transport.stat_file(partial) == expected, "The uploaded amendment-chain receipt differs.")
        transport.remote(["mv", "--", partial, remote])
        existing = transport.stat_file(remote)
    _require(existing == expected, "The remote amendment-chain receipt is foreign or changed.")
    return fields


def _runtime_preflight(config: Mapping[str, Any], transport: ControllerTransport, source: Mapping[str, Any]) -> tuple[str, str]:
    repo, campaign = config["remote_repo"], config["remote_campaign"]
    manifest = str(PurePosixPath(repo) / Path(config["execution_manifest"]).resolve().relative_to(phase3.REPO_ROOT).as_posix())
    runner = f"{repo}/scripts/run_ec_clean30_phase3.py"
    chain_fields = stage_session_chain_receipt(config, transport, source)
    chain_args = _chain_cli_options(chain_fields)
    transport.remote([config["remote_python"], runner, "verify-execution", "--manifest", manifest, *chain_args], cwd=repo)
    runpod = config.get("provider") == "runpod"
    profile = config.get("hardware_profile", "gcp_l4")
    cap_probe = _container_probe(config, transport) if runpod else None
    gpu = phase3.cuda_probe_program(profile)
    for key, python in (("fit", config["remote_python"]), ("esm1b", config["embedding"]["esm1b"]["clean_python"])):
        if runpod:
            _verify_runtime_lock(config, transport, key, python)
        measured = json.loads(transport.remote([*config["profiles"][key]["cap_prefix"], python, "-c", gpu], cwd=repo, timeout=90))
        phase3.validate_cuda_evidence(measured, profile)
        _write(Path(config["state_root"]) / f"cuda_{key}.json", measured)
    if runpod:
        program = phase3.main_model_probe_program(profile, f"{campaign}/g2_requests/main_probe_payload")
        measured = json.loads(transport.remote([*config["profiles"]["fit"]["cap_prefix"],
            config["remote_python"], "-c", program], cwd=repo, timeout=180))
        phase3.validate_main_model_evidence(measured, profile)
        local = Path(config["state_root"]) / "main_model_probe.json"
        _write(local, measured)
        transport.upload(local, f"{campaign}/g2_requests/main_model_probe.json")
        host = HostPull(transport, Path(config["durable_root"]), Path(config["state_root"]),
                        max_download_bytes=PAYLOAD_CEILING_BYTES, deadline_unix=transport.deadline_unix,
                        mirror_root=Path(config["mirror_root"]))
        for name, key in (("synthetic_checkpoint.pt", "checkpoint_sha256"),
                          ("cpu_predictions.json", "cpu_predictions_sha256"),
                          ("cuda_predictions.json", "gpu_predictions_sha256")):
            remote = f"{campaign}/g2_requests/main_probe_payload/{name}"
            expected = transport.stat_file(remote)
            _require(expected and expected["sha256"] == measured[key], "Qualification output differs from its receipt.")
            host._reserve(expected["size_bytes"])
            target = host.root / "qualification" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            transport.download(remote, target)
            _require(_sha(target) == measured[key], "Qualification output changed during readback.")
            host._mirror(f"qualification/{name}")
        clean = config["embedding"]["esm1b"]
        program = phase3.esm1b_model_probe_program(profile, clean["weights"], clean["weights_sha256"],
                                                  clean["contact_weights_sha256"])
        measured = json.loads(transport.remote([*config["profiles"]["esm1b"]["cap_prefix"],
            clean["clean_python"], "-c", program], cwd=repo, timeout=300))
        phase3.validate_esm1b_model_evidence(measured, profile)
        local = Path(config["state_root"]) / "esm1b_model_probe.json"
        _write(local, measured)
        transport.upload(local, f"{campaign}/g2_requests/esm1b_model_probe.json")
    fold = config.get("execution_fold", 0)
    view = f"{campaign}/views/fold{fold}"
    if transport.read_json(f"{view}/view_manifest.json", optional=True) is None:
        transport.remote([config["remote_python"], runner, "unpack-view", "--archive", config["view_archive"],
                          "--sha256", config["view_archive_sha256"], "--destination", view, "--fold", str(fold)], cwd=repo, timeout=300)
    cap = f"{campaign}/g2_requests/cap.json"
    local_cap = Path(config["state_root"]) / "cap.json"
    _write(local_cap, config["profiles"]["fit"]["cap_prefix"])
    transport.upload(local_cap, cap)
    probe = f"{campaign}/g2_requests/probe_fold{fold}_{source['execution_manifest_sha256']}.json"
    if transport.read_json(probe, optional=True) is None:
        arguments = [config["remote_python"], runner, "vm-probe", "--view", view,
                          "--execution-manifest", manifest, "--out", probe, "--cap-json", cap,
                          "--python", config["remote_python"], *chain_args]
        if runpod:
            arguments += ["--hardware-profile", profile, "--cap-probe", cap_probe]
        # The probe verifies the view and runs two cold-cache preparation probes under the two-core cap;
        # the same dry run took about ten minutes on the workstation.
        transport.remote(arguments, cwd=repo, timeout=VM_PROBE_SECONDS)
    return manifest, probe


def _selected_fallback(config: dict[str, Any], state: dict[str, Any], state_path: Path) -> None:
    """Bind only this pass's selected replacement before subsequent checks can fail."""
    root = Path(config["controller_root"])
    receipt = _json(root / "state/fallback.json")
    current = _json(root / "state/current_session.json")
    policy = _fallback_policy(config)
    replacement = receipt.get("replacement", {})
    source = state.get("fallback_source_identity", config["expected_identity"])
    _require(policy and receipt.get("phase") == "selected" and receipt.get("source") == source
             and receipt.get("requested_hours") == 2 and receipt.get("pass_id")
             and set(receipt.get("allowed_zones", [])) == set(policy["allowed_zones"])
             and receipt.get("candidates")
             and set(receipt.get("candidates", [])) <= set(policy["allowed_zones"])
             and replacement.get("zone") in receipt["candidates"]
             and replacement.get("project") == source["project"], "Selected fallback is outside this authorized pass.")
    _require(not state.get("fallback_pass_id") or state["fallback_pass_id"] == receipt["pass_id"],
             "A different fallback pass cannot replace this allocation.")
    effective = {**config, "expected_identity": replacement}
    _identity(effective, _controller_config(root))
    _require(current.get("active") is True and current.get("session_id") == receipt.get("session_id")
             and bool(current.get("session_id"))
             and all(str(current.get(key)) == str(value) for key, value in replacement.items())
             and current.get("session_start") == receipt.get("session_start")
             and math.floor(state["fallback_started_unix"]) <= _epoch(current["session_start"]) <= time.time(),
             "Selected fallback has no matching fresh owned session.")
    _require(not state.get("session_id") or state["session_id"] == current["session_id"],
             "The fallback now owns a different allocation.")
    config.update(expected_identity=replacement, fallback_started_unix=state["fallback_started_unix"])
    state.update(effective_identity=replacement, fallback_pass_id=receipt["pass_id"],
                 session_id=current["session_id"], controller_session_start=current["session_start"],
                 fallback_source_identity=source)
    _write(state_path, state)
    _write(state_path.parent / "fallback_selected.json", receipt)
    _write(state_path.parent / "effective_config.json", config)
    budget = _cross_region_budget(config)
    if budget:
        _require(receipt.get("recovery_budget_sha256") == policy["recovery_budget"]["sha256"],
                 "Selected recovery is bound to a different budget.")
        _verify_recovery_envelope(config, receipt["recovery_envelope"])
        _require(budget["transfer_bytes_per_restore"] >= float(receipt["source_disk"]["sizeGb"]) * GIB,
                 "Selected source disk exceeds the reserved restore-transfer size.")
    _require(0 < receipt["limit"]["effective_seconds"] <= MAX_G2_SECONDS,
             "Selected fallback exceeds the approved provider duration.")
    proof = policy["storage_cost_evidence"]
    size = _number(float(receipt["source_disk"]["sizeGb"]), "fallback source disk size")
    actual = {"disk_usd_per_hour": receipt["price"]["disk_usd_per_hour"],
              "snapshot_usd_per_hour": size * receipt["snapshot_price"]["usd_per_gib_month"] / 730,
              "recycle_minimum_usd": size * receipt["recycle_price"]["usd_per_gib_month"] / 730}
    _require(all(0 < float(rate) <= proof[key] + 1e-9 for key, rate in actual.items()),
             "Actual fallback storage prices exceed the verified reserve basis.")
    _cost(config)


def _fallback_after_stockout(config: dict[str, Any], state: dict[str, Any], state_path: Path,
                             transport: ControllerTransport, failure: Exception) -> None:
    policy = _fallback_policy(config)
    _require(policy and "ZONE_RESOURCE_POOL_EXHAUSTED" in str(failure),
             "Fallback requires an authorized, confirmed source capacity failure.")
    _require(not state.get("fallback_attempted") and config["expected_identity"]["zone"] == "us-central1-a",
             "Only one fallback pass after the source start is authorized.")
    _require(_status(transport.controller("status"), config["expected_identity"]) == "TERMINATED"
             and _json(transport.controller_root / "state/current_session.json").get("active") is False,
             "Fallback requires a positively stopped source without an active session.")
    _require(not (transport.controller_root / "state/fallback.json").exists(),
             "An existing fallback pass must be reconciled; no new pass is admitted.")
    arguments = ["--hours", "2"]
    for zone in policy["allowed_zones"]:
        arguments += ["--allowed-zone", zone]
    budget = _cross_region_budget(config)
    if budget:
        arguments += ["--recovery-budget", policy["recovery_budget"]["path"]]
    preview = transport.controller("fallback", "--dry-run", *arguments, timeout=300)
    candidates = re.search(r"^Candidate offerings \(capacity not guaranteed\): (.+)$", preview, re.MULTILINE)
    _require(candidates is not None and set(candidates[1].split(", ")) <= set(policy["allowed_zones"])
             and ("maxRunDuration=7020s" in preview or re.search(r"hard STOP after:\s+117 min", preview)),
             "Fallback preview differs from the approved zones or 117-minute STOP.")
    rates_line = re.search(r"^Verified storage rates: (.+)$", preview, re.MULTILINE)
    _require(rates_line is not None, "Fallback preview lacks verified storage prices.")
    rates = json.loads(rates_line[1])
    proof = policy["storage_cost_evidence"]
    _require(0 <= time.time() - _epoch(rates["verified_at"]) <= 86400
             and rates.get("snapshot_minimum_seconds") == rates.get("recycle_minimum_seconds") == 3600
             and all(0 < float(rates[key]) <= proof[field] + 1e-9 for key, field in (
                 ("disk_usd_per_hour", "disk_usd_per_hour"), ("snapshot_usd_per_hour", "snapshot_usd_per_hour"),
                 ("recycle_usd_per_hour", "recycle_minimum_usd"))),
             "Fallback preview storage prices exceed the approved reserve basis.")
    if budget:
        envelope_line = re.search(r"^Verified recovery envelope: (.+)$", preview, re.MULTILINE)
        _require(envelope_line is not None, "Fallback preview lacks its cross-region recovery envelope.")
        _verify_recovery_envelope(config, json.loads(envelope_line[1]))
    state.update(fallback_attempted=True, fallback_started_unix=time.time(), fallback_preview=preview,
                 fallback_source_identity=config["expected_identity"], start_failure=str(failure), cost=_cost(config))
    _write(state_path, state)
    try:
        transport.controller("fallback", "--confirm", *arguments, timeout=5400 if budget else 3600)
    except (WorkstationError, OSError, subprocess.SubprocessError):
        # A lost reply is not permission for another allocation. A durable selected
        # receipt can establish this pass's ownership, allowing the finalizer to stop it.
        receipt_path = transport.controller_root / "state/fallback.json"
        if receipt_path.is_file() and _json(receipt_path).get("phase") == "selected":
            _selected_fallback(config, state, state_path)
        raise
    _selected_fallback(config, state, state_path)


def _g2_common_request(config, transport, session, source, manifest, probe):
    root = f"{config['remote_campaign']}/runs/fold0/g2"
    common = {**session, "execution_manifest": manifest, "view": f"{config['remote_campaign']}/views/fold0",
              "vm_probe": probe, "execution_root": root, "durable_root": str(Path(config["durable_root"]).resolve()),
              "persistence_mode": "host_pull", "python": config["remote_python"], "device": "cuda", "smoke": False,
              "attempt": 1}
    common.update(_remote_chain_request(config, source))
    if config.get("provider") == "runpod":
        common.update(hardware_profile=config["hardware_profile"],
                      main_model_probe=f"{config['remote_campaign']}/g2_requests/main_model_probe.json",
                      main_model_probe_sha256=_sha(Path(config["state_root"]) / "main_model_probe.json"),
                      esm1b_model_probe=f"{config['remote_campaign']}/g2_requests/esm1b_model_probe.json",
                      esm1b_model_probe_sha256=_sha(Path(config["state_root"]) / "esm1b_model_probe.json"))
        if source.get("format") == phase3.EXECUTION_V5_FORMAT:
            common.update(provider="runpod", runpod_config_sha256=transport.stop_controller.config_sha,
                          plan_version=2)
    return common


def _g2_host(config, transport, session, source):
    return HostPull(transport, Path(config["durable_root"]), Path(config["state_root"]),
                    max_download_bytes=_g2_download_limit(config, source), deadline_unix=session["deadline_unix"],
                    mirror_root=Path(config["mirror_root"]) if config.get("mirror_root") else None)


def _g2_evidence(config, relative, source):
    local = Path(config["durable_root"]) / _relative(relative)
    _require(local.is_file() and not local.is_symlink(), "The completed G2 evidence was not preserved locally.")
    record = _json(local)
    phase3.require_record_precedence(record, source)
    return {"local_path": str(local.resolve()), "remote_path": f"{config['remote_campaign']}/runs/fold0/g2/{relative}",
            "sha256": _sha(local)}


def _prepare_g2_v5(config, transport, state, state_path, session, ready, manifest, probe, check_session, prepare_g2):
    """Persist the cache/CPU parity/attachment unit before the first G2 timing admission."""
    source = ready["source"]
    _require(source.get("format") == phase3.EXECUTION_V5_FORMAT and callable(prepare_g2),
             "Execution v5 needs its published-source preparation hook.")
    specification = dict(prepare_g2(config, None, state, session, ready, manifest, probe, check_session))
    _require(set(specification) == {"pretrain_request", "max_download_bytes"}
             and specification["max_download_bytes"] == _g2_download_limit(config, source),
             "G2 preparation changes the approved preservation ceiling.")
    options = specification["pretrain_request"]
    allowed = {"union_fasta", "corpus_table", "holdout_receipt", "official_split_dir", "identity_work",
               "hf_home", "esmc_weights", "cpu_cap_prefix", "forecast_seconds", "unit_timeout_seconds"}
    _require(isinstance(options, dict) and set(options) == allowed,
             "G2 preparation accepts only its bound F6 inputs and explicit resource/time bounds.")
    common = _g2_common_request(config, transport, session, source, manifest, probe)
    request = {**common, **config["profiles"]["fit"], **options,
               "attempt": state.get("resume_g2_attempts", {}).get("pretrain", 1)}
    binding = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    _require(state.get("pretrain_request_sha256", binding) == binding,
             "The session's F6 preparation request changed during recovery.")
    state["pretrain_request_sha256"] = binding
    state["g2_max_download_bytes"] = specification["max_download_bytes"]
    _write(state_path, state)
    forecast = _number(request["forecast_seconds"], "F6 preparation forecast")
    timeout = _number(request["unit_timeout_seconds"], "F6 preparation timeout")
    _require(timeout <= forecast, "F6 preparation timeout exceeds its admission forecast.")
    check_session()
    _require(session["deadline_unix"] - time.time() >= forecast * 1.25 + RESERVE_SECONDS,
             "F6 preparation does not fit the owned session and closeout reserve.")
    relative = f"runs/ecc30_pretrain_fold0_attempt{request['attempt']}/pretrain_complete.json"
    host = _g2_host(config, transport, session, source)
    completed = complete_remote_unit(transport, host, request,
        runner=f"{config['remote_repo']}/scripts/run_ec_clean30_phase3.py", action="pretrain-prepare",
        remote_request=f"{config['remote_campaign']}/g2_requests/pretrain_prepare.json",
        completed_relative=relative, timeout=timeout + 60)
    _require(completed.get("status") == "passed", "F6 cache, parity and attachment must pass before G2 v2.")
    phase3.require_record_precedence(completed, source)
    evidence = _g2_evidence(config, relative, source)
    _require(_json(Path(evidence["local_path"])) == completed, "Preserved F6 preparation receipt differs.")
    state["pretrain_preparation"] = {**evidence, "record": completed}
    _write(state_path, state)


def _execute_queue(config, transport, state, state_path, session, ready, manifest, probe, check_session):
    """The exact versioned queue, with all receipts preserved before continuation."""
    source = ready["source"]
    transport.reserve_seconds = 30
    host = _g2_host(config, transport, session, source)
    common = _g2_common_request(config, transport, session, source, manifest, probe)
    root = common["execution_root"]
    v5 = source.get("format") == phase3.EXECUTION_V5_FORMAT
    if v5:
        pretrain = state.get("pretrain_preparation", {})
        _require(pretrain.get("record", {}).get("status") == "passed" and pretrain.get("remote_path"),
                 "G2 v2 requires the completed cache, parity and attachment unit.")
        common["pretrain_evidence"] = pretrain["remote_path"]
    for action, name in measurement_queue(source):
        if name in state["completed"]:
            continue
        check_session()
        attempt = state.get("resume_g2_attempts", {}).get(name, 1)
        request = {**common, **config["profiles"]["esm1b" if name == "esm1b" else "fit"], **config["bounds"][name], "attempt": attempt}
        if action == "timing-unit":
            request["recipe"] = name
            complete = f"runs/ecc30timing_{name}_attempt{attempt}/g2_complete.json"
        elif action == "lane-gate":
            request.update(serial_receipts=[f"{root}/runs/ecc30timing_lane_serial_{i}_attempt{state.get('resume_g2_attempts', {}).get(f'lane_serial_{i}', 1)}/g2_complete.json" for i in (1, 2)],
                           limits=config["lane_limits"])
            complete = f"runs/ecc30timing_lane_batch_attempt{attempt}/lane_gate.json"
            measured = [transport.read_json(path)["wall_seconds"] for path in request["serial_receipts"]]
            request["forecast_seconds"] = max(request["forecast_seconds"], 3 * max(measured))
        else:
            request.update(kind=name, timing_inputs=config["timing_inputs"], **config["embedding"][name])
            complete = f"runs/ecc30timing_embed_{name}_attempt{attempt}/embedding_complete.json"
        _require(not v5 or all(request.get(key) == value for key, value in common.items() if key != "attempt"),
                 "A v5 timing request cannot override session, source or preparation identity.")
        remaining = session["deadline_unix"] - time.time()
        _require(remaining >= request["forecast_seconds"] * 1.25 + RESERVE_SECONDS,
                 "Next complete unit does not fit the approved allocation; retain evidence and request a new decision.")
        completed = complete_remote_unit(transport, host, request, runner=f"{config['remote_repo']}/scripts/run_ec_clean30_phase3.py",
            action=action, remote_request=f"{config['remote_campaign']}/g2_requests/{name}.json",
            completed_relative=complete, timeout=request["unit_timeout_seconds"] + 60)
        expected_identity = phase3.source_identity(ready["source"])
        _require(all(completed.get(key) == value for key, value in expected_identity.items())
                 and completed.get("status") in ({"passed", "failed"} if action == "lane-gate" else {"passed"}),
                 "The completed unit receipt is failed or foreign.")
        if v5:
            phase3.require_record_precedence(completed, source)
            state.setdefault("g2_evidence", {})[name] = _g2_evidence(config, complete, source)
        state["completed"].append(name)
        _write(state_path, state)
    state["status"] = "measurements_complete"


def _complete_g2_v5(config, transport, state, state_path, session, ready, manifest, probe, check_session, complete_g2):
    """Materialize Freeze 1c locally from persisted evidence, then ship its immutable bytes."""
    source = ready["source"]
    _require(state.get("completed") == [name for _, name in measurement_queue(source)] and callable(complete_g2),
             "Freeze 1c v2 requires the complete authenticated G2 v2 queue.")
    specification = dict(complete_g2(config, None, state, session, ready, manifest, probe, check_session))
    _require(set(specification) == {"local", "remote", "sha256"}
             and specification["remote"] == f"{config['remote_campaign']}/g2_requests/freeze1c_v2.json",
             "Freeze 1c v2 must use its exact session staging destination.")
    local = Path(specification["local"])
    _require(local.is_absolute() and local.resolve() == local and local.is_file() and not local.is_symlink()
             and _sha(local) == specification["sha256"], "The Freeze 1c v2 file is missing or changed.")
    record = _json(local)
    _require(record.get("format") == "deepmzyme_ec_clean30_freeze1c_v2",
             "The continuation requires Freeze 1c v2.")
    phase3.require_record_precedence(record, source)
    _require(not state.get("freeze1c") or state["freeze1c"] == specification,
             "The session's Freeze 1c v2 binding cannot change during recovery.")
    check_session()
    _stage_file(config, transport, {**specification, "kind": "file"})
    state["freeze1c"] = specification
    _write(state_path, state)


class RunpodTransport(ControllerTransport):
    """The same authenticated file transport with an independent stop controller."""

    def __init__(self, config, controller):
        super().__init__(Path(config["state_root"]), config["runpod"]["ssh_alias"],
                         bootstrap_python=config["bootstrap_python"])
        self.stop_controller = controller

    def _bounded(self, timeout):
        self.stop_controller.heartbeat()
        return super()._bounded(timeout)

    def refresh(self):
        # The provider's current direct SSH address and port for the owned pod; ownership is checked first.
        self.ssh_config = self.stop_controller.write_ssh_config()

    def remote(self, argv, *, cwd=None, timeout=60):
        # SSH does not necessarily retain container-level environment variables.
        # The detached launcher and every descendant inherit this fixed policy.
        return super().remote(["env", "NVIDIA_TF32_OVERRIDE=0", "PYTHONNOUSERSITE=1", *argv], cwd=cwd, timeout=timeout)

    def controller(self, name, *args, **kwargs):
        raise WorkstationError("GCP lifecycle commands cannot operate a Runpod allocation.")


def _remote_inventory(config, transport, session, root, program, state_root):
    """The closeout inventory as a hashed control file outside its root, downloaded and verified before use.

    SSH output carries only a work estimate and the file's digest. Hashing time scales with the files and bytes
    found; the transport bounds every call by the remaining session time minus its reserve.
    """
    digest = hashlib.sha256(json.dumps([session["pod_id"], session["session_id"], root]).encode()).hexdigest()
    control = f"{config['remote_campaign']}/g2_requests/closeout/inventory_{digest}.json"
    _require(PurePosixPath(root) not in PurePosixPath(control).parents,
             "The inventory control file must lie outside the inventoried root.")
    scan = ("import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); n=b=0"
            "\nif p.exists():\n for f in p.rglob('*'):\n  if not f.is_symlink() and f.is_file(): n+=1; b+=f.stat().st_size\n"
            "print(json.dumps({'files':n,'bytes':b}))")
    work = json.loads(transport.remote([config["bootstrap_python"], "-c", scan, root], timeout=INVENTORY_SCAN_SECONDS))
    _require(isinstance(work, dict) and all(type(work.get(key)) is int and work[key] >= 0 for key in ("files", "bytes")),
             "The remote inventory work estimate is malformed.")
    timeout = (INVENTORY_MIN_SECONDS + work["files"] * INVENTORY_FILE_SECONDS
               + work["bytes"] / INVENTORY_HASH_BYTES_PER_SECOND)
    pointer = json.loads(transport.remote([config["bootstrap_python"], "-c", program, root, control,
                                           str(INVENTORY_LIMIT_BYTES)], timeout=timeout))
    _require(isinstance(pointer, dict) and pointer.get("path") == control and type(pointer.get("size_bytes")) is int
             and 0 < pointer["size_bytes"] <= INVENTORY_LIMIT_BYTES
             and re.fullmatch(r"[0-9a-f]{64}", str(pointer.get("sha256"))),
             "The remote inventory digest is malformed or exceeds its size bound.")
    local = Path(state_root) / "closeout_inventories" / f"inventory_{digest}.json"
    transport.download(control, local, timeout=INVENTORY_MIN_SECONDS
                       + pointer["size_bytes"] / INVENTORY_TRANSFER_BYTES_PER_SECOND)
    _require(local.is_file() and not local.is_symlink() and local.stat().st_size == pointer["size_bytes"]
             and _sha(local) == pointer["sha256"], "The downloaded remote inventory differs from its remote digest.")
    inventory = json.loads(local.read_bytes())
    _require(isinstance(inventory, dict) and isinstance(inventory.get("files"), list), "The remote inventory is malformed.")
    return inventory


def _preserve_runpod_outputs(config, transport, controller, session, *, output=None, receipt_name="preservation_receipt.json",
                            max_download_bytes=PAYLOAD_CEILING_BYTES, host_state_root=None):
    """Read every output back twice before authorizing deletion of persistent storage."""
    output = output or {"execution_root": f"{config['remote_campaign']}/runs/fold0/g2",
                        "primary_root": config["durable_root"], "mirror_root": config["mirror_root"]}
    root = output["execution_root"]
    program = ("import hashlib,json,os,pathlib,sys; p=pathlib.Path(sys.argv[1]); files=[]; "
        "\nif p.exists():\n for f in sorted(p.rglob('*')):\n"
        "  assert not f.is_symlink(), 'Output symlink is not eligible for cleanup'\n"
        "  if f.is_file():\n"
        "   if f.name == 'hot_storage.json':\n"
        "    marker=json.loads(f.read_text()); assert marker.get('status') == 'restored', 'Pending hot data blocks cleanup'\n"
        "   if f.name == 'pretrain_attachment.json':\n"
        "    marker=json.loads(f.read_text()); assert marker.get('status') == 'restored', 'Pending pretraining data blocks cleanup'\n"
        "   if f.name == 'scientific_outputs_pending.json':\n"
        "    marker=json.loads(f.read_text()); assert marker.get('status') == 'restored', 'Pending scientific data blocks cleanup'\n"
        "   with f.open('rb') as h: digest=hashlib.file_digest(h,'sha256').hexdigest()\n"
        "   files.append({'path':f.relative_to(p).as_posix(),'size_bytes':f.stat().st_size,'sha256':digest})\n"
        "s=json.loads((p/'execution_state.json').read_text()) if (p/'execution_state.json').exists() else {}; "
        "data=json.dumps({'files':files,'pending_transfer':s.get('pending_transfer'),'pending_unit':s.get('pending_unit')}).encode()"
        "\nassert len(data) <= int(sys.argv[3]), 'Remote inventory exceeds its size bound'\n"
        "o=pathlib.Path(sys.argv[2]); o.parent.mkdir(parents=True,exist_ok=True); q=o.with_name(o.name+'.partial'); "
        "q.write_bytes(data); os.replace(q,o); "
        "print(json.dumps({'path':str(o),'sha256':hashlib.sha256(data).hexdigest(),'size_bytes':len(data)}))")
    inventory = _remote_inventory(config, transport, session, root, program, host_state_root or config["state_root"])
    identity = {"pod_id": session["pod_id"], "session_id": session["session_id"], "config_sha256": controller.config_sha}
    inventory.update(format="deepmzyme_ec_clean30_runpod_remote_inventory_v1", **identity)
    host = HostPull(transport, Path(output["primary_root"]), Path(host_state_root or config["state_root"]),
                    max_download_bytes=max_download_bytes, deadline_unix=session["deadline_unix"],
                    mirror_root=Path(output["mirror_root"]))
    missing, count = [], 0
    for entry in inventory["files"]:
        name = _relative(entry["path"])
        local = host.root / name
        _require(host.root in local.resolve().parents and not local.is_symlink(), "Unsafe closeout member.")
        if not local.is_file() or local.stat().st_size != entry["size_bytes"] or _sha(local) != entry["sha256"]:
            missing.append(name)
            count += entry["size_bytes"]
    if missing:
        for attempt in range(2):
            host._reserve(count)  # Each attempt is charged in full, as in HostPull.sync_pending.
            try:
                transport.download_files(root, host.root, missing, timeout=host._timeout())
                break
            except (OSError, subprocess.SubprocessError, WorkstationError):
                if attempt == 1:
                    raise
                # A read-only read waits for the endpoint (bounded reconnect) before the charged repeat.
                transport.remote([config["bootstrap_python"], "-c", "print('ready')"], timeout=15)
    for entry in inventory["files"]:
        local = host.root / entry["path"]
        _require(local.stat().st_size == entry["size_bytes"] and _sha(local) == entry["sha256"],
                 "Closeout output readback differs from the remote inventory.")
        host._mirror(entry["path"])
    relative = "closeout/remote_inventory.json"
    _write(host.root / relative, inventory)
    host._mirror(relative)
    files = inventory["files"] + [{"path": relative, "size_bytes": (host.root / relative).stat().st_size,
                                   "sha256": _sha(host.root / relative)}]
    manifests = []
    for entry in inventory["files"]:
        name = entry["path"]
        if name.startswith("persistence_receipts/") and name.endswith(".json"):
            document = _json(host.root / name)
            if (isinstance(document, dict) and document.get("schema_version") == 1
                    and (not config.get("resume") or document.get("destination_root") == str(host.root))
                    and document.get("persistence_mode") == "host_pull"
                    and isinstance(document.get("ack_path"), str) and isinstance(document.get("files"), list)):
                manifests.append(name)
    receipt = {"format": "deepmzyme_ec_clean30_runpod_preservation_v1", **identity,
               "execution_root": root,
               "primary_root": str(host.root), "mirror_root": str(host.mirror_root), "files": files,
               "remote_inventory_path": relative, "host_pull_manifests": manifests}
    path = Path(config["state_root"]) / receipt_name
    _write(path, receipt)
    return path


def _preserve_runpod_session_outputs(config, transport, controller, session, state):
    outputs = state.get("phase4", {}).get("preservation_roots")
    g2_limit = state.get("g2_max_download_bytes", PAYLOAD_CEILING_BYTES)
    if outputs is None:
        return _preserve_runpod_outputs(config, transport, controller, session, max_download_bytes=g2_limit)
    receipts = []
    for index, output in enumerate(outputs):
        path = _preserve_runpod_outputs(config, transport, controller, session, output=output,
                                       receipt_name=f"preservation_receipt_{index}.json",
                                       max_download_bytes=state["phase4"]["max_download_bytes"] if index else g2_limit,
                                       host_state_root=state["phase4"].get("host_state_root") if index else None)
        receipts.append({"path": str(path.resolve()), "sha256": _sha(path)})
    path = Path(config["state_root"]) / "preservation_set.json"
    _write(path, {"format": "deepmzyme_ec_clean30_runpod_preservation_set_v1",
                  "pod_id": session["pod_id"], "session_id": session["session_id"],
                  "config_sha256": controller.config_sha, "receipts": receipts})
    return path


def _storage_facts(path: Path) -> tuple[int, int]:
    while not path.exists():
        path = path.parent
    return path.stat().st_dev, shutil.disk_usage(path).free


def _overlay_stage(config: Mapping[str, Any], components: PurePosixPath, fold: int, manifest: str) -> None:
    """Before allocation: one reviewed tar puts the selector's overlay copy on the pod, and nothing else.

    Phase-4 stage_files refuse identity paths, so only a G2 stage tar extracting to <overlay> can carry
    <overlay>/identity/fold<K>; without it the pod's first check would come after paid CLEAN work.
    """
    overlay, folder = components.parent.parent, f"identity/fold{fold}"
    items = [item for item in config.get("stage", []) if item.get("kind") == "tar"
             and item.get("extract_to") == str(overlay)]
    _require(len(items) == 1, "The overlay identity copy needs exactly one reviewed stage tar "
             "extracting to <remote campaign>/overlay/<overlay id>.")
    local = Path(items[0]["local"])
    _require(local.is_file() and not local.is_symlink() and _sha(local) == items[0].get("sha256"),
             "The staged overlay identity copy is missing or changed.")
    files = set()
    with tarfile.open(local) as archive:
        for member in archive:
            name = member.name.removeprefix("./").rstrip("/")
            if name in ("", ".") and member.isdir():
                continue  # The archive's own root entry.
            inside = _relative(name).startswith(folder + "/")
            _require((member.isdir() and (inside or name in ("identity", folder)))
                     or (member.isfile() and (inside or name == manifest)),
                     "The staged overlay carries only this fold's identity copy and the overlay manifest, "
                     "as regular files.")
            if member.isfile():
                files.add(name)
    _require(manifest in files and any(name.startswith(folder + "/") for name in files),
             "The staged overlay needs this fold's identity copy and the overlay manifest.")


def validate_phase4_plan(config: Mapping[str, Any], source: Mapping[str, Any],
                         request: Mapping[str, Any], preapproval: Mapping[str, Any]) -> dict[str, Any]:
    """Read-only transport validation before allocation; measured G2 fields are added later."""
    from ec_clean30 import campaign, container_caps, phase4, provenance, views

    _require(source.get("format") == phase3.EXECUTION_V5_FORMAT and request.get("fold") in range(5)
             and type(request.get("fold")) is int, "The reviewed v5 continuation must be a complete canonical fold plan.")
    fold = request["fold"]
    required = {"local_plan", "execution_manifest", "remote_runner", "durable_root", "mirror_root",
                "host_state_root", "max_download_bytes", "remote_request_template", "queue_actions",
                "family_forecasts", "family_timeouts", "clean_forecasts", "clean_timeouts",
                "embedding_forecasts", "embedding_timeouts", "scientific_forecasts", "scientific_timeouts",
                "refit_forecast_seconds", "refit_timeout_seconds", "stage_files"}
    _require(required <= set(request), "The reviewed Phase-4 transport request is incomplete.")
    _require(request["queue_actions"] == ["clean_embeddings", "clean_fits", "s1"]
             and request.get("queue_requests", {}) == {}, "V5 uses its fixed pipeline without queue overrides.")
    attempts = request.get("attempts", {})
    _require(isinstance(attempts, dict) and all(isinstance(name, str) and name
             and type(value) is int and value > 0 for name, value in attempts.items()),
             "Phase-4 attempts must be explicit positive integers.")
    _require(request.get("clean_scope", "inner") == "inner", "The initial CLEAN queue must use its inner closure.")
    _require(type(request.get("run_final_clean", False)) is bool, "Final CLEAN dispatch must be an explicit boolean.")
    local_plan = Path(request["local_plan"])
    _require(local_plan.is_absolute() and local_plan.resolve() == local_plan and local_plan.is_file(),
             "The reviewed local fold plan must be an existing canonical file.")
    plan = campaign.build_fold_plan_v2(fold)
    _require(_json(local_plan) == plan, "The reviewed fold plan differs from the canonical amendment-2 plan.")
    clean_ids = {unit["unit_id"] for unit in phase4.clean_units()} if fold == 0 else set()
    authorized_clean = set(clean_ids)
    if request.get("run_final_clean", False):
        authorized_clean.update(unit["unit_id"] for unit in phase4.clean_final_units(fold))
    _require(preapproval.get("campaign_id") == campaign.CAMPAIGN_ID
             and preapproval.get("scope") == phase4.V5_SCOPE and preapproval.get("allowed_folds") == [fold]
             and preapproval.get("plan_sha256_by_fold") == {str(fold): plan["plan_sha256"]}
             and preapproval.get("allowed_steps") == {str(fold): ["S1", "S4"]}
             and preapproval.get("clean_unit_ids") == sorted(authorized_clean),
             "G3 preapproval must cover exactly the reviewed fold-0 fitted and CLEAN units.")
    _require(not {"allocations", "freeze1c_sha256", "git_head", "created_utc"}.intersection(preapproval),
             "G3 preapproval cannot contain fabricated measured fields.")

    forecasts = []
    def bounds(forecast, timeout, name):
        seconds = _number(forecast, name + " forecast")
        _require(_number(timeout, name + " timeout") <= seconds, "A Phase-4 timeout exceeds its forecast.")
        _require(1.25 * seconds + RESERVE_SECONDS < config["runpod"]["max_runtime_seconds"],
                 "A Phase-4 unit cannot fit the reviewed allocation and shutdown reserve.")
        forecasts.append(seconds)

    if fold > 0:
        _require(config.get("resume") is not None, "Later folds require preserved fold-0 evidence.")
        bounds(request.get("attachment_forecast_seconds"), request.get("attachment_timeout_seconds"), "F6 attachment")
        _require({"corpus_table", "holdout_receipt", "identity_work", "official_split_dir"}
                 <= set(request["remote_request_template"]), "Later folds need their bound F6 attachment inputs.")
    families = {unit.family for unit in phase4.s1_units(plan)}
    groups = {"family": families, "clean": clean_ids, "embedding": {"base", "masked"} if fold == 0 else set(),
              "scientific": set(phase4.SCIENTIFIC_STEPS) if fold == 0 else {"select_s1", "select_s2", "select_s4"}}
    for group, names in groups.items():
        for suffix in ("forecasts", "timeouts"):
            values = request[f"{group}_{suffix}"]
            _require(isinstance(values, dict) and set(values) == names,
                     f"Phase-4 {group} {suffix} must cover the exact canonical queue.")
        for name in names:
            bounds(request[f"{group}_forecasts"][name], request[f"{group}_timeouts"][name], str(name))
    bounds(request["refit_forecast_seconds"], request["refit_timeout_seconds"], "refit")

    template = request["remote_request_template"]
    needed = {"plan", "view", "campaign_root", "execution_root", "durable_root", "python", "device",
              "persistence_mode", "step_chain", "fold0_step_chain", "scientific_inputs", "science_cap_profile",
              "clean_cap_profile", "cap_prefix", "distance_cap_prefix", "memory_limit_bytes", "threads",
              "clean_inputs", "base_farm", "masked_farm", "clean_python", "clean_clone", "training_script",
              "distance_helper", "export_script", "embedding_python", "harness", "extract_script", "weights",
              "weights_sha256", "contact_weights_sha256", "hot_work_root", "working_bytes", "distance_map_bytes"}
    _require(isinstance(template, dict) and needed <= set(template), "The reviewed Phase-4 worker template is incomplete.")
    _require(template.get("clean_scope", "inner") == "inner", "The initial CLEAN worker must use its inner closure.")
    def remote_path(value):
        _require(isinstance(value, str) and bool(value), "Every remote Phase-4 path must be explicit.")
        path = PurePosixPath(_absolute(value))
        _require(str(path) == value, "Remote Phase-4 paths must be canonical lexical paths.")
        return path

    campaign_root, repo = remote_path(config["remote_campaign"]), remote_path(config["remote_repo"])
    execution_root = campaign_root / f"runs/fold{fold}/phase4"
    _require(request["remote_runner"] == str(repo / "scripts/run_ec_clean30_phase4.py")
             and template["view"] == str(campaign_root / f"views/fold{fold}")
             and template["campaign_root"] == str(campaign_root)
             and template["execution_root"] == str(execution_root)
             and template["durable_root"] == request["durable_root"]
             and template["python"] == config["remote_python"]
             and template["clean_python"] == config["embedding"]["esm1b"]["clean_python"]
             and template["embedding_python"] == template["clean_python"]
             and template["device"] == "cuda" and template["persistence_mode"] == "host_pull",
             "The reviewed worker changes the owned runtime, view or output roots.")
    path_keys = {"plan", "step_chain", "fold0_step_chain", "clean_inputs", "base_farm", "masked_farm",
                 "clean_clone", "training_script", "distance_helper", "export_script", "harness", "extract_script",
                 "weights", "hot_work_root", "python", "clean_python", "embedding_python"}
    paths = {key: remote_path(template[key]) for key in path_keys}
    for kind, key in (("base", "base_farm"), ("masked", "masked_farm")):
        unit_id = f"ecc30_clean_embed_{kind}"
        expected = execution_root / "runs" / unit_id / f"attempt{attempts.get(unit_id, 1)}" / "farm"
        _require(paths[key] == expected, "CLEAN farms must name the reviewed embedding attempts' generated outputs.")
    _require(execution_root in paths["step_chain"].parents and (paths["fold0_step_chain"] == paths["step_chain"] if fold == 0 else
                 campaign_root / "runs/fold0/phase4" in paths["fold0_step_chain"].parents),
             "Fold-0 step chains must share the owned execution root.")
    scientific = template["scientific_inputs"]
    scientific_names = {"components", "clean_sanity_record", "clean_sanity_dispositions", "clean_sanity_verify_all",
                        "futility_out", "f6_corpus", "r1_inner_lists"}
    _require(isinstance(scientific, dict) and set(scientific) == scientific_names
             and isinstance(scientific["clean_sanity_dispositions"], list)
             and isinstance(scientific["r1_inner_lists"], list) and len(scientific["r1_inner_lists"]) == 4
             and len(set(scientific["r1_inner_lists"])) == 4,
             "The reviewed scientific inputs must include all sanity, selection and futility paths.")
    # The real selector reads only the frozen overlay's identity copy; no other input may name such a folder.
    overlay_id = ((((source.get("frozen_views") or {}).get(str(fold)) or {}).get("overlay")) or {}).get("id")
    _require(isinstance(overlay_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", overlay_id),
             "The frozen fold view names no data overlay for the selector's identity copy.")
    components = campaign_root / "overlay" / overlay_id / "identity" / f"fold{fold}"
    for name, value in scientific.items():
        if name == "components":
            item = PurePosixPath(value) if isinstance(value, str) else None
            _require(item is not None and item.is_absolute() and ".." not in item.parts and str(item) == value
                     and not any(c in value for c in "\n\r\0") and item == components
                     and not provenance.is_test_named_path(value, relative_to=str(campaign_root)),
                     "Phase-4 components must be the frozen overlay copy "
                     "<remote campaign>/overlay/<overlay id>/identity/fold<K>.")
            _overlay_stage(config, components, fold, views.OVERLAY_MANIFEST_NAME)
            continue
        values = value if isinstance(value, list) else [value]
        for item in values:
            path = remote_path(item)
            if name in {"clean_sanity_record", "clean_sanity_verify_all", "futility_out"}:
                _require(execution_root in path.parents, "Scientific outputs must stay under the owned execution root.")
    profile = template["clean_cap_profile"]
    _require(isinstance(profile, dict) and set(profile) ==
             {"cgroup_root", "cpu_cores", "memory_bytes", "distance_map_memory_bytes"}
             and profile == preapproval.get("clean_cap_profile")
             and profile.get("cgroup_root") == config["cgroup_root"]
             and all(type(profile[key]) is int and profile[key] > 0
                     for key in ("cpu_cores", "memory_bytes", "distance_map_memory_bytes"))
             and template["threads"] == profile["cpu_cores"]
             and template["memory_limit_bytes"] == profile["memory_bytes"],
             "The reviewed CLEAN resource profile differs from G3 or the qualified runtime.")
    for key, memory in (("cap_prefix", "memory_bytes"), ("distance_cap_prefix", "distance_map_memory_bytes")):
        _require(template[key] == container_caps.cap_prefix(template["python"], Path(repo), profile["cgroup_root"],
                    cpu_cores=profile["cpu_cores"], memory_bytes=profile[memory]),
                 "The reviewed CLEAN command differs from its explicit kernel resource profile.")
    science = template["science_cap_profile"]
    _require(isinstance(science, dict) and set(science) == {"cgroup_root", "cpu_cores", "memory_bytes"}
             and science == preapproval.get("science_cap_profile") and science.get("cgroup_root") == config["cgroup_root"]
             and all(type(science[key]) is int and science[key] > 0 for key in ("cpu_cores", "memory_bytes")),
             "The reviewed scientific resource profile differs from G3 or the qualified runtime.")
    _require(type(template["working_bytes"]) is int and template["working_bytes"] > 0
             and type(template["distance_map_bytes"]) is int and 0 < template["distance_map_bytes"] <= phase4.MAX_DISTANCE_BYTES
             and template["hot_work_root"] == preapproval.get("hot_work_root"),
             "CLEAN needs its explicit approved working-space forecasts and hot root.")
    protected = [repo, campaign_root / f"views/fold{fold}", execution_root,
                 paths["python"].parent.parent, paths["clean_python"].parent.parent,
                 *(paths[key] for key in ("clean_clone", "clean_inputs", "base_farm", "masked_farm"))]
    hot = paths["hot_work_root"]
    _require(all(hot != path and hot not in path.parents and path not in hot.parents for path in protected),
             "CLEAN hot storage overlaps source, runtime, view, inputs or canonical outputs.")
    input_pins = preapproval.get("clean_inputs_sha256_by_scope")
    if input_pins is None:
        input_pins = {"fold0_inner": preapproval.get("clean_inputs_sha256")}
    _require(isinstance(input_pins, dict) and set(input_pins) ==
             (({"fold0_inner"} if fold == 0 else set()) |
              ({f"fold{fold}_final"} if request.get("run_final_clean", False) else set()))
             and all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in input_pins.values())
             and all(isinstance(template[key], str) and re.fullmatch(r"[0-9a-f]{64}", template[key])
                     for key in ("weights_sha256", "contact_weights_sha256")),
             "Reviewed CLEAN inputs and embedding weights require SHA-256 pins.")
    if request.get("run_final_clean", False):
        final = request.get("final_clean_request")
        _require(isinstance(final, dict) and {"clean_scope", "clean_inputs", "clean_forecasts", "clean_timeouts"} <= set(final)
                 and set(final) <= {"clean_scope", "clean_inputs", "clean_forecasts", "clean_timeouts", "farms_by_unit",
                                   "embedding_forecasts", "embedding_timeouts"}
                 and final["clean_scope"] == "final", "Final CLEAN needs its separately reviewed input closure and forecasts.")
        final_ids = {unit["unit_id"] for unit in phase4.clean_final_units(fold)}
        for suffix in ("forecasts", "timeouts"):
            _require(isinstance(final[f"clean_{suffix}"], dict) and set(final[f"clean_{suffix}"]) == final_ids,
                     "Final CLEAN forecasts must cover its exact frozen unit set.")
        for name in final_ids:
            bounds(final["clean_forecasts"][name], final["clean_timeouts"][name], name)
        final_input = remote_path(final["clean_inputs"])
        _require(final_input != paths["clean_inputs"] and final_input != hot
                 and final_input not in hot.parents and hot not in final_input.parents,
                 "Final CLEAN needs a separate closure outside hot working storage.")
        final_request = {**request, **final}
        _require(final.get("farms_by_unit") == phase4.final_farms_by_unit(final_request),
                 "Final CLEAN farms must name their exact per-seed producer outputs.")
        embedding_ids = {unit["unit_id"] for unit in phase4.embedding_units(final_request)}
        for suffix in ("forecasts", "timeouts"):
            _require(set(final.get(f"embedding_{suffix}", {})) == embedding_ids,
                     "Final farm forecasts must cover every base and seed-specific masked producer.")
        for name in embedding_ids:
            bounds(final["embedding_forecasts"][name], final["embedding_timeouts"][name], name)
        for key in ("hf_home", "esmc_weights"):
            remote_path(template.get(key))

    else:
        _require("final_clean_request" not in request, "An unused final CLEAN override is outside the reviewed pipeline.")

    staged = {}
    _require(isinstance(request["stage_files"], list), "Phase-4 staging must be an explicit file list.")
    for row in request["stage_files"]:
        _require(isinstance(row, dict) and set(row) == {"local", "remote", "sha256"}, "Invalid Phase-4 staged file.")
        local, remote = Path(row["local"]), remote_path(row["remote"])
        _require(local.is_absolute() and local.resolve() == local and local.is_file()
                 and _sha(local) == row["sha256"] and campaign_root in remote.parents and str(remote) not in staged,
                 "A reviewed Phase-4 staged file is missing, changed or outside the campaign.")
        _require(not any(remote == campaign_root / name or campaign_root / name in remote.parents
                         for name in ("views", "runs", "g2_requests")),
                 "Reviewed staging files cannot overwrite views, output roots or generated session metadata.")
        phase3._no_restricted_path(local)
        staged[str(remote)] = row["sha256"]
    if str(paths["plan"]) not in staged:
        _require(repo in paths["plan"].parents, "The reviewed remote fold plan has no bound staged file.")
        candidate = phase3.REPO_ROOT / paths["plan"].relative_to(repo)
        _require(candidate.is_file() and _sha(candidate) == _sha(local_plan), "The checked-out fold plan differs.")
    else:
        _require(staged[str(paths["plan"])] == _sha(local_plan), "The staged fold plan differs from the reviewed plan.")
    limit = request["max_download_bytes"]
    _require(type(limit) is int and limit > 0, "Phase 4 requires an explicit positive preservation-byte ceiling.")
    roots = [Path(config[key]) for key in ("durable_root", "mirror_root")]
    roots += [Path(request[key]) for key in ("durable_root", "mirror_root")]
    for root in [*roots, Path(request["host_state_root"])]:
        _require(root.is_absolute() and root.resolve() == root, "Phase-4 preservation paths must be canonical absolute paths.")
    _require(all(first != second and first not in second.parents and second not in first.parents
                 for index, first in enumerate(roots) for second in roots[index + 1:]),
             "G2 and Phase-4 preservation roots cannot overlap.")
    _require(Path(request["host_state_root"]) != Path(config["state_root"]),
             "G2 and Phase 4 need separate preservation-budget state roots.")
    facts = [_storage_facts(root) for root in roots]
    _require(facts[0][0] != facts[1][0] and facts[2][0] != facts[3][0],
             "G2 and Phase 4 require independent primary and mirror devices.")
    required_by_device = {}
    for (device, _free), allowance in zip(facts, [_g2_download_limit(config, source)] * 2 + [limit] * 2):
        required_by_device[device] = required_by_device.get(device, 5 * GIB) + allowance
    _require(all(free >= required_by_device[device] for device, free in facts),
             "Independent storage cannot hold the combined G2 and Phase-4 allowances plus reserve.")
    return {"status": "passed", "plan_sha256": plan["plan_sha256"], "largest_unit_forecast_seconds": max(forecasts),
            "max_download_bytes": limit}


def _continue_runpod_phase4(config, transport, state, state_path, session, ready, manifest, probe,
                           check_session, prepare_phase4):
    """Plan locally, then register outputs before explicit queue work on the owned allocation.

    prepare_phase4 may create local metadata only. Its transport argument is
    deliberately None: it must not stage files, produce embeddings or fit models.
    """
    from ec_clean30 import phase4

    source = ready.get("source", {})
    _require(state.get("completed") == [name for _, name in measurement_queue(source)],
             "Complete G2 receipts are required before Phase 4.")
    check_session()
    request = dict(prepare_phase4(config, None, state, session, ready, manifest, probe, check_session))
    expected = {key: session[key] for key in ("pod_id", "session_id", "allocation_started_unix", "deadline_unix", "max_total_seconds")}
    expected.update(provider="runpod", runpod_config_sha256=transport.stop_controller.config_sha)
    _require(all(key not in request or request[key] == value for key, value in expected.items()),
             "Prepared Phase 4 request changes the owned allocation.")
    request.update(expected)
    _require(Path(request["execution_manifest"]).resolve() == Path(config["execution_manifest"]).resolve(),
             "Phase 4 must use the same published execution manifest as G2.")
    actions = request.get("queue_actions")
    _require(isinstance(actions, list) and actions and len(set(actions)) == len(actions)
             and all(action in {"s1", "clean_embeddings", "clean_fits"} for action in actions),
             "An explicit ordered Phase 4 queue is required.")
    overrides = request.get("queue_requests", {})
    allowed_overrides = {"s1": {"family_forecasts", "family_timeouts", "attempts"},
                         "clean_embeddings": {"embedding_forecasts", "embedding_timeouts", "attempts"},
                         "clean_fits": {"clean_forecasts", "clean_timeouts", "attempts"}}
    _require(isinstance(overrides, dict) and set(overrides) <= set(actions)
             and all(isinstance(values, dict) and set(values) <= allowed_overrides[action]
                     for action, values in overrides.items()),
             "Queue overrides may contain only declared forecasts, timeouts and attempts; runtime and authority are fixed.")
    _require(config.get("execution_scope") not in {"g2_then_fold0_v5", "fold_execution_v5"}
             or source.get("format") == phase3.EXECUTION_V5_FORMAT,
             "Phase 4 v5 requires the authenticated execution-v5 source.")
    chain_fields = _remote_chain_request(config, source)
    local_chain_fields = ({"amendment_chain_receipt": str(session_chain_receipt_path(config)),
                           "expected_chain": chain_fields["expected_chain"]} if chain_fields else {})
    _require(all(key not in request or request[key] == value for key, value in local_chain_fields.items()),
             "Phase 4 preparation changes the authenticated amendment chain.")
    # The outer coordinator verifies source locally; the worker checks the shipped receipt on the pod.
    request.update(local_chain_fields)
    template = dict(request["remote_request_template"])
    _require(all(key not in template or template[key] == value for key, value in expected.items()),
             "Remote Phase 4 request changes the owned allocation.")
    template.update(expected)
    _require(all(key not in template or template[key] == value for key, value in chain_fields.items()),
             "The remote Phase 4 request changes the authenticated amendment chain.")
    template.update(chain_fields)
    _require(template["execution_root"] == f"{config['remote_campaign']}/runs/fold{request.get('fold', 0)}/phase4"
             and template["durable_root"] == request["durable_root"], "Phase 4 execution or durable root differs.")
    request["remote_request_template"] = template
    outputs = [{"execution_root": f"{config['remote_campaign']}/runs/fold0/g2",
                "primary_root": str(Path(config["durable_root"]).resolve()),
                "mirror_root": str(Path(config["mirror_root"]).resolve())},
               {"execution_root": template["execution_root"], "primary_root": request["durable_root"],
                "mirror_root": request["mirror_root"]}]
    for inherited in state.get("resume_output_roots", []):
        if inherited["execution_root"] not in {item["execution_root"] for item in outputs}:
            outputs.append(inherited)
    for output in outputs:
        devices = []
        for key in ("primary_root", "mirror_root"):
            root = Path(output[key])
            _require(root.is_absolute() and root.resolve() == root, "Phase 4 destinations must be canonical absolute paths.")
            while not root.exists():
                root = root.parent
            devices.append(root.stat().st_dev)
        _require(devices[0] != devices[1], "Phase 4 requires independent primary and mirror devices.")
    transport.stop_controller.register_output_roots(outputs)
    binding = {"request_sha256": hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":"),
                                                          allow_nan=False).encode()).hexdigest(),
               "preservation_roots": outputs, "max_download_bytes": request["max_download_bytes"],
               "host_state_root": request["host_state_root"]}
    _require(not state.get("phase4", {}).get("request_sha256") or all(state["phase4"].get(key) == value for key, value in binding.items()),
             "A prepared Phase 4 continuation cannot change during recovery.")
    state.setdefault("phase4", {}).update(binding)
    _write(state_path, state)
    if source.get("format") == phase3.EXECUTION_V5_FORMAT:
        from ec_clean30 import execution_v5

        _require(actions == ["clean_embeddings", "clean_fits", "s1"],
                 "Execution v5 requires its declared CLEAN preparation and fold-selection pipeline.")
        _require(state.get("freeze1c", {}).get("remote")
                 == f"{config['remote_campaign']}/g2_requests/freeze1c_v2.json",
                 "Phase 4 requires the completed, staged Freeze 1c v2.")

        def record_progress(name, result):
            check_session()
            _require(isinstance(name, str) and name and isinstance(result, Mapping),
                     "A pipeline progress entry needs a name and a result record.")
            state["phase4"].setdefault("completed_queues", {})[name] = dict(result)
            _write(state_path, state)

        result = execution_v5.run_phase4_pipeline(request, transport=transport, check_session=check_session,
                                                  record_progress=record_progress)
        _require(result.get("status") in {"passed", "stop_futility"},
                 "The v5 pipeline did not reach a verified terminal boundary.")
        state["phase4"]["result"] = result
        state["status"] = "stopped_futility" if result["status"] == "stop_futility" else "phase4_complete"
        _write(state_path, state)
        return
    runners = {"s1": phase4.run_s1_queue, "clean_embeddings": phase4.run_embedding_queue,
               "clean_fits": phase4.run_clean_remote_queue}
    for action in actions:
        check_session()
        current = {**request, **overrides.get(action, {})}
        _require(all(current.get(key) == value for key, value in expected.items())
                 and all(current.get(key) == request.get(key) for key in
                         ("execution_manifest", "durable_root", "mirror_root", "host_state_root")),
                 "Phase 4 queue override changes source, ownership or persistence.")
        result = runners[action](current, transport=transport, check_session=check_session, close_session=False)
        state["phase4"].setdefault("completed_queues", {})[action] = result
        _write(state_path, state)
    state["status"] = "phase4_complete"
    _write(state_path, state)


def _operator_root(config, *, prepare=False) -> Path:
    # Data1 may expose synthetic 0777 modes. Controls live on the independent
    # POSIX mirror; ordinary status can still be preserved on both devices.
    root = (Path(config['mirror_root']).parent / 'operator' if config.get('mirror_root')
            else Path(config['state_root']))
    _require(root.is_absolute() and root == root.resolve(), 'Operator control root must be canonical without symlinks.')
    if prepare:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(root, 0o700)
    info = root.stat()
    _require(stat.S_ISDIR(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o700
             and info.st_uid == os.getuid() and not root.is_symlink(),
             'Operator control root must be a private owned directory with genuine POSIX permissions.')
    if prepare:
        probe = root / ('permission-probe-' + secrets.token_hex(8) + '.json')
        try:
            _operator_write(probe, {'private': True})
            _operator_json(probe)
        finally:
            probe.unlink(missing_ok=True)
    return root


def _operator_write(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + '.' + secrets.token_hex(8) + '.partial')
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'w') as handle:
            json.dump(value, handle, sort_keys=True, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _operator_json(path: Path) -> dict[str, Any]:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor) as handle:
        info = os.fstat(handle.fileno())
        _require(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o600
                 and info.st_uid == os.getuid() and info.st_nlink == 1 and info.st_size <= 128 * 1024,
                 'Operator controls require a private owned regular file.')
        value = json.load(handle)
    _require(isinstance(value, dict), 'Malformed operator control.')
    return value


def request_operator(config, config_path: Path, action: str) -> dict[str, Any]:
    """Queue one session-bound control; never allocate or operate the provider."""
    _require(config.get('provider') == 'runpod' and config.get('runpod', {}).get('keep_alive_for_operator') is True,
             'Operator controls require explicit Runpod keep-alive mode.')
    _require(action in {'retry-qualification', 'closeout'}, 'Unknown operator action.')
    root = _operator_root(config)
    with (root / 'operator.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        hold = _operator_json(root / 'operator_hold.json')
        now = time.time()
        _require(hold.get('active') is True and hold.get('config_sha256') == _sha(config_path)
                 and 0 <= now - hold.get('updated_unix', 0) <= 45 and now < hold['hold_until_unix'],
                 'Operator hold is stale, closed or belongs to another configuration.')
        _require(action != 'retry-qualification' or hold.get('retry_allowed') is True,
                 'Qualification retry is forbidden after workload admission.')
        request = {key: hold[key] for key in ('hold_id', 'config_sha256', 'session_id', 'pod_id',
                                              'deadline_unix', 'source_identity_sha256')}
        request.update(action=action, created_unix=now)
        path = root / 'operator_request.json'
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, 'w') as handle:
            json.dump(request, handle, sort_keys=True, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
    return {'status': 'operator_request_recorded', **request}


def _operator_hold(config, config_path, state, state_path, session, source, check_session, *, retry_allowed, reason):
    """Keep the existing owner alive without admitting work or extending its window."""
    root = _operator_root(config)
    until = session['deadline_unix'] - RESERVE_SECONDS
    identity = {key: session[key] for key in ('session_id', 'pod_id', 'allocation_started_unix',
                                             'deadline_unix', 'hard_stop_unix')}
    hold = {**identity, 'hold_id': secrets.token_hex(16), 'config_sha256': state['config_sha256'],
            'source_identity_sha256': hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
            'started_unix': time.time(), 'hold_until_unix': until, 'active': True,
            'retry_allowed': bool(retry_allowed), 'reason': reason, 'control_root': str(root)}
    previous_status = state.get('status')
    result = 'closeout-reserve'
    try:
        while time.time() < until:
            current = check_session()
            _require(all(current.get(key) == value for key, value in identity.items()),
                     'Operator hold session identity or deadline changed.')
            _require(_sha(config_path) == state['config_sha256'], 'Held launch configuration changed.')
            if time.time() >= until:
                break
            hold['updated_unix'] = time.time()
            _operator_write(root / 'operator_hold.json', hold)
            state.update(status='operator_hold', operator_hold=dict(hold))
            _write(state_path, state)
            request_path = root / 'operator_request.json'
            if request_path.exists() or request_path.is_symlink():
                with (root / 'operator.lock').open('a+') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    accepted = False
                    action = None
                    try:
                        request = _operator_json(request_path)
                        _require(all(request.get(key) == hold[key] for key in ('hold_id', 'config_sha256',
                            'session_id', 'pod_id', 'deadline_unix', 'source_identity_sha256')),
                            'Operator request is foreign or belongs to an earlier hold.')
                        now = time.time()
                        _require(hold['started_unix'] <= request.get('created_unix', 0) <= now + 5
                                 and now - request['created_unix'] <= 60,
                                 'Operator request is stale or future-dated.')
                        action = request.get('action')
                        _require(action in {'retry-qualification', 'closeout'}, 'Unknown operator action.')
                        if action == 'retry-qualification':
                            _require(retry_allowed and not state.get('workload_admission_attempted')
                                     and not state.get('completed'), 'Qualification retry is forbidden after workload admission.')
                            measured = _execution_source(config, published=True)
                            _require(measured == source, 'Qualification retry requires the exact same published source.')
                            _local_storage(config)
                            _require(time.time() < until, 'Qualification retry would consume the closeout reserve.')
                        accepted = True
                    except (RuntimeError, OSError, ValueError, KeyError, TypeError) as exc:
                        state['operator_control_error'] = f'{type(exc).__name__}: {exc}'
                    finally:
                        request_path.unlink(missing_ok=True)
                        _operator_write(root / 'operator_control_result.json', {
                            'hold_id': hold['hold_id'], 'session_id': session['session_id'], 'pod_id': session['pod_id'],
                            'accepted': accepted, 'action': action, 'reason': state.get('operator_control_error') if not accepted else None,
                            'observed_unix': time.time()})
                if accepted:
                    result = action
                    return result
            time.sleep(min(15, max(0, until - time.time())))
        return result
    finally:
        hold.update(active=False, updated_unix=time.time(), exit_action=result)
        _operator_write(root / 'operator_hold.json', hold)
        state.update(status=previous_status, operator_hold=dict(hold))
        _write(state_path, state)


def _host_controller(config, config_path, host_try):
    from ec_clean30 import runpod_controller
    return runpod_controller.RunpodController(config, config_path, host_try=host_try)


def _host_transport(config, controller):
    return RunpodTransport(config, controller)


def _run_runpod_g2(config, config_path, *, transport=None, controller=None, pre_allocation=None,
                   prepare_phase4=None, prepare_g2=None, complete_g2=None, recover_window=False):
    from ec_clean30 import runpod_controller
    continuation = config.get("execution_scope") in {"g2_then_fold0_v5", "fold_execution_v5"}
    _require(not continuation or (config.get("provider") == "runpod"
                                 and callable(pre_allocation) and callable(prepare_phase4)
                                 and callable(prepare_g2) and callable(complete_g2)),
             "Execution v5 requires the Runpod launch gate and same-session Phase 4 preparation.")
    _require(type(recover_window) is bool and (not recover_window or (continuation and callable(pre_allocation))),
             "Only the v5 launch gate may recover an already reserved launch window.")
    keep_alive = config.get('runpod', {}).get('keep_alive_for_operator', False)
    _require(type(keep_alive) is bool, 'keep_alive_for_operator must be a boolean.')
    if keep_alive:
        _operator_root(config, prepare=True)
    state_root = Path(config["state_root"])
    state_root.mkdir(parents=True, exist_ok=True)
    state_path = state_root / "driver_state.json"
    with (state_root / "driver.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = _json(state_path) if state_path.exists() else {"config_sha256": _sha(config_path), "completed": []}
        _require(state["config_sha256"] == _sha(config_path) and not state.get("closed"),
                 "Runpod configuration changed or its allocation is already closed.")
        owned = bool(state.get("start_attempted"))
        session, heartbeat_context, heartbeat = None, None, None
        try:
            # A recovering window may hold an aged quote only to reach its owned allocation (start already
            # attempted); a window that never attempted a start re-checks quote freshness below.
            ready = _prepared_preflight(config, config_path, published=True, fresh=not recover_window)
            if continuation:
                _require(ready["source"].get("format") == phase3.EXECUTION_V5_FORMAT,
                         "The v5 continuation requires its published execution-v5 manifest.")
                state["g2_max_download_bytes"] = _g2_download_limit(config, ready["source"])
            if pre_allocation is not None:
                binding = pre_allocation(config, config_path, ready)
                _require(isinstance(binding, dict)
                         and binding.get("config_sha256") == _sha(config_path)
                         and binding.get("execution_manifest_sha256") == ready["source"]["execution_manifest_sha256"],
                         "Launch gate must bind this configuration and published execution source.")
                _require("launch_binding" not in state or state["launch_binding"] == binding,
                         "The launch authorization binding cannot change during this session.")
                state["launch_binding"] = binding
                _write(state_path, state)
            host_try = state.get("host_try", 1)
            _require(type(host_try) is int and 1 <= host_try <= runpod_controller.HOST_TRY_LIMIT,
                     "Invalid recorded same-window host try.")
            controller = controller or runpod_controller.RunpodController(config, config_path, host_try=host_try)
            transport = transport or RunpodTransport(config, controller)
            cached = bool(config.get("artifact_cache"))
            if cached:
                from ec_clean30 import runpod_bootstrap
                _require(continuation or ready["source"]["format"] == phase3.BOOTSTRAP_FORMAT,
                         "Cached bootstrap requires its authenticated execution-v4 manifest.")
                runpod_bootstrap.verify_seeded_cache(config, source=ready["source"])
                staging_scratch = runpod_bootstrap.staging_scratch_bytes(config)
            gcp = Path(config["gcp_controller_root"])
            status = subprocess.run([str(gcp / "bin/vm-status")], check=True, capture_output=True, text=True,
                                    timeout=90, env=phase3.clean_repository_environment()).stdout
            _require(_status(status, config["gcp_stopped_identity"]) == "TERMINATED", "The retained GCP VM must remain stopped.")
            _write(state_root / "gcp_before.json", {"status": status})
            if not owned:
                if recover_window:
                    # A stock search's create admission never checks quote age, so a reserved window
                    # that never attempted a start needs the fresh quote a new window needs.
                    runpod_controller.validate_config(config)
                state["start_attempted"] = True
                _write(state_path, state)
                owned = True
                adopt_id = config.get("runpod", {}).get("adopt_pod_id")
                session = controller.adopt(adopt_id) if adopt_id else controller.start()
            else:
                session = controller.session(state.get("session_id"))
            state.update(session)
            _write(state_path, state)
            heartbeat_context = controller.maintain_heartbeat(interval=30)
            heartbeat = heartbeat_context.__enter__()

            def check_session():
                heartbeat.check()
                return controller.session(session["session_id"])

            transport.deadline_unix = session["deadline_unix"]
            transport.reserve_seconds = RESERVE_SECONDS
            while True:
                problem = None
                try:
                    until = min(time.time() + 300, session["deadline_unix"] - RESERVE_SECONDS)
                    while True:
                        check_session()
                        try:
                            transport.ssh_config = controller.write_ssh_config()
                            transport.remote([config["bootstrap_python"], "-c", "print('ready')"], timeout=15)
                            break
                        except (OSError, subprocess.SubprocessError, RuntimeError):
                            _require(time.time() + 10 < until, "The owned pod did not become reachable within its startup bound.")
                            time.sleep(10)
                    if cached:
                        _bootstrap_runpod(config, transport, ready["source"], session)
                    else:
                        _stage_checkout(config, transport, ready["source"])
                        _container_probe(config, transport, early=True)
                    check_session()
                    free = int(transport.remote([config["bootstrap_python"], "-c",
                        "import shutil; print(shutil.disk_usage('/workspace').free)"]).strip())
                    extra = staging_scratch + _g2_download_limit(config, ready["source"]) if cached else 0
                    _require(free >= ready["staging_peak_bytes"] + extra + 5 * GIB,
                             "Pod disk lacks staging and closeout space.")
                    _prepare_runpod_storage(config, transport, ready, session)
                    if cached:
                        _stage_runpod_runtime(config, transport, ready["source"], check_session)
                    else:
                        _stage_runtime(config, transport, ready["source"])
                    check_session()
                    manifest, probe = _runtime_preflight(config, transport, ready["source"])
                    check_session()
                    # An uncertain admission must never enable a qualification retry.
                    state['workload_admission_attempted'] = True
                    _write(state_path, state)
                    controller.admit_workload()
                    state['workload_admitted'] = True
                    _write(state_path, state)
                    if continuation and config.get("resume"):
                        from ec_clean30 import execution_v5
                        execution_v5.restore_session(config, transport, state, state_path, ready["source"], check_session)
                    if continuation and not state.get("pretrain_preparation"):
                        _prepare_g2_v5(config, transport, state, state_path, session, ready, manifest, probe,
                                       check_session, prepare_g2)
                    _execute_queue(config, transport, state, state_path, session, ready, manifest, probe,
                                   check_session)
                    if continuation:
                        _complete_g2_v5(config, transport, state, state_path, session, ready, manifest, probe,
                                        check_session, complete_g2)
                        _continue_runpod_phase4(config, transport, state, state_path, session, ready, manifest, probe,
                                               check_session, prepare_phase4)
                except Exception as exc:
                    state.update(status='interrupted', error=f'{type(exc).__name__}: {exc}')
                    state.setdefault('qualification_errors', []).append(state['error'])
                    if (isinstance(exc, HostRejected) and continuation and not keep_alive and session is not None
                            and host_try < runpod_controller.HOST_TRY_LIMIT
                            and not state.get('workload_admission_attempted') and not state.get('completed')):
                        # Same window, new host: the rejected pod stays allocated until the replacement
                        # create returns, then is deleted (nothing ran on it, so nothing is preserved).
                        previous, previous_session, previous_heartbeat = controller, session, heartbeat_context
                        rejection = state.pop('error')
                        state.pop('status', None)
                        state.setdefault('host_rejections', []).append(
                            {'host_try': host_try, 'pod_id': previous_session['pod_id'],
                             'session_id': previous_session['session_id'], 'error': rejection})
                        for key in previous_session:
                            state.pop(key, None)
                        host_try += 1
                        state['host_try'] = host_try
                        _write(state_path, state)
                        controller = _host_controller(config, config_path, host_try)
                        transport = _host_transport(config, controller)
                        session = heartbeat_context = heartbeat = None
                        try:
                            session = controller.start(replacing={'pod_id': previous_session['pod_id'],
                                                                  'state_root': str(previous.root)})
                        finally:
                            try:
                                state['host_rejections'][-1]['release'] = previous.release_rejected_host(rejection)
                            finally:
                                previous_heartbeat.__exit__(None, None, None)
                                _write(state_path, state)
                        controller.replacement_released()
                        state.update(session)
                        _write(state_path, state)
                        heartbeat_context = controller.maintain_heartbeat(interval=30)
                        heartbeat = heartbeat_context.__enter__()
                        transport.deadline_unix = session["deadline_unix"]
                        transport.reserve_seconds = RESERVE_SECONDS
                        continue
                    if not keep_alive:
                        raise
                    problem = exc
                if not keep_alive:
                    break
                check_session()  # Identity, watchdog and ownership failures still close safely.
                action = _operator_hold(config, config_path, state, state_path, session, ready['source'], check_session,
                    retry_allowed=problem is not None and not state.get('workload_admission_attempted')
                                  and not state.get('completed'),
                    reason=state.get('error') if problem else 'measurements_complete')
                if action == 'retry-qualification':
                    state.pop('error', None)
                    state['qualification_retry_count'] = state.get('qualification_retry_count', 0) + 1
                    _write(state_path, state)
                    continue
                if problem:
                    raise problem
                break
        except BaseException as exc:
            state.update(status="interrupted", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if owned and controller is not None:
                try:
                    preservation = None
                    if session:
                        transport.reserve_seconds = 15
                        try:
                            preservation = _preserve_runpod_session_outputs(config, transport, controller, session, state)
                        except Exception as exc:
                            state["preservation_error"] = f"{type(exc).__name__}: {exc}"
                    stopped = controller.stop()
                    state.update(closed=True, provider_stop_verified=True, stop_status=stopped)
                    if preservation:
                        state["closeout"] = controller.closeout(preservation)
                finally:
                    try:
                        if heartbeat_context is not None:
                            heartbeat_context.__exit__(None, None, None)
                    finally:
                        _write(state_path, state)
        return state


def run_g2(config: Mapping[str, Any], config_path: Path, *, transport: ControllerTransport | None = None) -> dict[str, Any]:
    _require(config.get("execution_scope") not in {"g2_then_fold0_v5", "fold_execution_v5"},
             "Execution v5 must enter its published-source launch dispatcher with explicit authorization.")
    if config.get("provider") == "runpod":
        return _run_runpod_g2(config, config_path, transport=transport)
    config = dict(config)
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
            if state.get("fallback_pass_id"):
                _selected_fallback(config, state, state_path)
            ready = preflight(config, config_path, published=True)
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
                try:
                    transport.controller("start", "--confirm", "--hours", "2", timeout=1200)
                except WorkstationError as exc:
                    if not _fallback_policy(config) or "ZONE_RESOURCE_POOL_EXHAUSTED" not in str(exc):
                        raise
                    _fallback_after_stockout(config, state, state_path, transport, exc)
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
            _execute_queue(config, transport, state, state_path, session, ready, manifest, probe,
                           lambda: _session(config, transport, session["session_id"]))
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
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--run", action="store_true", help="Deliberately start once, or recover this still-active owned allocation.")
    actions.add_argument("--run-v5", action="store_true", help="Run the published v5 dispatcher under explicit reviewed launch authority.")
    actions.add_argument("--seed-cache", action="store_true", help="Verify and seed the configured private artifact cache without allocating a GPU.")
    actions.add_argument('--operator', choices=('retry-qualification', 'closeout'),
                         help='Send one private control to this still-active Runpod operator hold.')
    parser.add_argument("--authorization", type=Path, help="External reviewed launch-authorization record; --run-v5 only.")
    parser.add_argument("--window-id", help="The approved launch-window identity; --run-v5 only.")
    args = parser.parse_args(argv)
    try:
        _require((args.authorization is not None and bool(args.window_id)) if args.run_v5
                 else (args.authorization is None and args.window_id is None),
                 "--run-v5 requires --authorization and --window-id; those options are refused for every other action.")
        config = _json(args.config)
        if args.operator:
            result = request_operator(config, args.config, args.operator)
        elif args.run or args.run_v5:
            def interrupted(signum, frame):
                raise KeyboardInterrupt(f"signal {signum}")
            signal.signal(signal.SIGTERM, interrupted)
            if args.run_v5:
                from ec_clean30.execution_v5 import run_v5

                result = run_v5(config, args.config, authorization_path=args.authorization, window_id=args.window_id)
            else:
                result = run_g2(config, args.config)
        elif args.seed_cache:
            from ec_clean30 import runpod_bootstrap
            ready = _prepared_preflight(config, args.config, published=True)
            result = runpod_bootstrap.seed_cache(config, args.config, source=ready["source"])
        else:
            result = _prepared_preflight(config, args.config)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (RuntimeError, OSError, ValueError, KeyError, subprocess.SubprocessError, KeyboardInterrupt) as exc:
        print(f"Workstation refused: {exc}", file=sys.stderr)
        return 2
