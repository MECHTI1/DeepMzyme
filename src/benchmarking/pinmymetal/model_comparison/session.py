#!/usr/bin/env python3
"""Workstation coordinator for the metal-completion successor (standard library plus the frozen host-pull tool).

It binds the maintainer's approval to the frozen manifest and permitted units, refreshes the controller session and
gross-spending snapshots the worker requires, deploys the sealed package to the GPU host at the identical absolute
paths, launches one unit per lane through the exit-file wrapper, pulls and acknowledges each lane's artifacts, places
them where the collector reads, and builds the collect run map. It never starts, stops or extends the VM: the
external VM controller ($DEEPMZYME_VM_CONTROLLER/bin) owns billing, and ``run_metal_completion.py`` on the host refuses
every unit whose authorization, session or spending snapshot is missing, stale or outside the recorded ceilings.
"""

from __future__ import annotations


import sys as _sys
from pathlib import Path as _Path
for _p in (str(_Path(__file__).resolve().parents[4]), str(_Path(__file__).resolve().parents[4] / "src")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
from project_paths import RELOCATED_FILES, install_relocated_module_aliases, resolve_existing_path, resolve_repo_file
install_relocated_module_aliases()
import argparse
import datetime as dt
import getpass
import hashlib
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import pmm_v3_host_pull as host_pull  # noqa: E402

CAMPAIGN_ID = "metal_completion_2026-10"
DEFAULT_PREP = ROOT / "DeepMzyme_Data" / "campaigns" / "metal_completion_2026-10" / "preparation_20261014T164111Z"
TRAIN_DIR = ROOT / "DeepMzyme_Data" / "DeepMzyme_PMM_Zenodo_Exact_Dataset" / "dataset" / "train"
ESM_DIR = ROOT / "DeepMzyme_Data" / "campaigns" / "pmm_ion_metal_v2_context" / "inputs" / "esm_embeddings_esmc600m_v1"
CONTROLLER = host_pull.CONTROLLER  # DEEPMZYME_VM_CONTROLLER overrides the default
HOST = host_pull.SSH_HOST  # DEEPMZYME_SSH_HOST overrides the default
REMOTE_HOME = os.environ.get("DEEPMZYME_REMOTE_HOME", str(Path.home()))
REMOTE_PY = f"{REMOTE_HOME}/venvs/deepmzyme/bin/python"
# Where the restored VM disk already holds the frozen inputs (v3 campaign layout); bind-mounted onto the local paths.
REMOTE_TRAIN_SOURCE = f"{REMOTE_HOME}/deepmzyme_data/pmm/train_and_test_sets_structures_zenodo_pmm_exact/train"
REMOTE_ESM_SOURCE = f"{REMOTE_HOME}/deepmzyme_runs/pmm_ion_metal_v2_context/inputs/esm_embeddings_esmc600m_v1"
# A gcloud missing from PATH raises FileNotFoundError at the first snapshot (fails closed, before any deploy).
GCLOUD = os.environ.get("DEEPMZYME_GCLOUD") or shutil.which("gcloud") or "gcloud"
FIT_SECONDS = 2400.0  # one 50-epoch late-fusion fit with three lanes and warm caches (v3 step C forecast)
LAUNCH_MARGIN_SECONDS = 300.0  # the worker re-verifies the package (minutes) before its own admission check
SNAPSHOT_MAX_AGE = 900.0
SNAPSHOT_REFRESH_AFTER = 480.0
POLL_SECONDS = 30.0
STARTING_GRACE_SECONDS = 120.0
LANES = (0, 1, 2)


class CoordinatorError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CoordinatorError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with resolve_existing_path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def write_json_once(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def write_json_atomic(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)
    return path


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stamp(value: str) -> float:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


# ---------------------------------------------------------------------------
# Package layout (identical on the workstation and the GPU host)
# ---------------------------------------------------------------------------

class Prep:
    def __init__(self, root: Path = DEFAULT_PREP):
        self.root = Path(root)
        require(self.root.is_absolute(), "The preparation root must be absolute")

    code = property(lambda self: self.root / "code")
    parent = property(lambda self: self.root / "parent")
    execution = property(lambda self: self.root / "execution")
    evidence = property(lambda self: self.root / "evidence")
    authorization = property(lambda self: self.root / "execution" / "authorization")
    launch = property(lambda self: self.root / "execution" / "launch")
    host_pull = property(lambda self: self.root / "execution" / "host_pull")
    lanes = property(lambda self: self.root / "execution" / "lanes")

    def manifest(self, phase: str) -> Path:
        return self.execution / f"manifest_{phase}.json"

    def policy(self, phase: str) -> Path:
        return self.authorization / f"policy_{phase}.json"

    decision = property(lambda self: self.authorization / "user_decision.txt")
    session = property(lambda self: self.authorization / "session.json")
    spending = property(lambda self: self.authorization / "spending.json")

    def lane(self, index: int) -> Path:
        return self.lanes / f"lane{index}"


def manifest_units(prep: Prep, phase: str) -> tuple[dict[str, Any], str, list[str]]:
    path = prep.manifest(phase)
    manifest = read_json(path)
    require(manifest.get("campaign_id") == CAMPAIGN_ID and manifest.get("phase") == phase, "Wrong manifest")
    return manifest, sha256_file(path), list(manifest["authorized_grid_units"])


def new_units(prep: Prep, phase: str, packet_path: Path | None = None) -> list[str]:
    """Grid units that no verified historical control covers (the only units a worker may run)."""
    _, _, units = manifest_units(prep, phase)
    packet = read_json(packet_path or prep.evidence / "control_records.json")
    reused = set(packet["records"])
    return [name for name in units if name not in reused]


# ---------------------------------------------------------------------------
# Approval record and controller snapshots
# ---------------------------------------------------------------------------

def record_approval(prep: Prep, *, phase: str, decision_file: Path, decision_sentence: str, units: list[str],
                    gross_ceiling_usd: float, max_sessions: int, max_session_seconds: float,
                    max_total_gpu_seconds: float, lanes: int, estimated_fit_seconds: float, reserve_usd: float,
                    expires_unix: float, controller_owner: str, superseded: list[str], approval_unix: float,
                    storage_usd_per_day: float) -> Path:
    """Write the immutable authorization policy the worker binds; the decision text is the maintainer's own words."""
    manifest, manifest_sha, grid = manifest_units(prep, phase)
    require(units and set(units) <= set(grid) and len(set(units)) == len(units), "Units must be a subset of the grid")
    text = Path(decision_file).read_text()
    require(decision_sentence.strip() and decision_sentence in text, "The decision sentence must occur in the file")
    if prep.decision.exists():
        require(prep.decision.read_bytes() == Path(decision_file).read_bytes(), "A different decision record exists")
    else:
        prep.decision.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(decision_file, prep.decision)
    policy = {"schema_version": 1, "campaign_id": CAMPAIGN_ID, "phase": phase, "execution_authorized": True,
              "manifest_path": str(prep.manifest(phase)), "manifest_sha256": manifest_sha,
              "user_decision": decision_sentence, "user_decision_path": str(prep.decision),
              "user_decision_sha256": sha256_file(prep.decision), "units": list(units),
              "expires_unix": float(expires_unix), "max_sessions": int(max_sessions), "lanes": int(lanes),
              "persistence_mode": "host_pull", "durable_root": str(prep.host_pull),
              "max_session_seconds": float(max_session_seconds), "snapshot_max_age_seconds": SNAPSHOT_MAX_AGE,
              "reserve_usd": float(reserve_usd), "gross_ceiling_usd": float(gross_ceiling_usd),
              "estimated_fit_seconds": float(estimated_fit_seconds), "controller_owner": controller_owner,
              "max_total_gpu_seconds": float(max_total_gpu_seconds), "superseded_policy_sha256": list(superseded),
              "approval_unix": float(approval_unix), "storage_usd_per_day": float(storage_usd_per_day),
              "recorded_utc": utc_now(),
              "scope_note": "Validation-only discovery/confirmation fits on the frozen grid; no refit, test access, "
                            "migration, larger hardware or additional experiments."}
    return write_json_once(prep.policy(phase), policy)


def read_config(controller: Path = CONTROLLER) -> dict[str, str]:
    lines = (controller / "config.env").read_text().splitlines()
    return dict(line.split("=", 1) for line in lines if line and not line.startswith("#") and "=" in line)


def controller_session(controller: Path = CONTROLLER) -> dict[str, Any]:
    state = read_json(controller / "state" / "current_session.json")
    require(state.get("active") is True, "No active controller session: the VM is not running under the controller")
    config = read_config(controller)
    for key, setting in (("project", "PROJECT_ID"), ("zone", "ZONE"), ("vm_name", "VM_NAME"),
                         ("instance_id", "SELECTED_INSTANCE_ID")):
        require(str(state.get(key)) == config.get(setting), f"Session {key} differs from config.env {setting}")
    return state


def describe_instance(config: dict[str, str]) -> dict[str, Any]:
    env = dict(os.environ, CLOUDSDK_PYTHON=host_pull.CONDA_PYTHON, CLOUDSDK_PYTHON_SITEPACKAGES="1")
    result = subprocess.run([GCLOUD, "compute", "instances", "describe", config["VM_NAME"], "--zone", config["ZONE"],
                             "--project", config["PROJECT_ID"], "--format=json"],
                            env=env, capture_output=True, text=True, timeout=180, stdin=subprocess.DEVNULL)
    require(result.returncode == 0, f"gcloud describe failed: {result.stderr.strip()[-500:]}")
    return json.loads(result.stdout)


def ledger_events(controller: Path = CONTROLLER) -> list[dict[str, Any]]:
    path = controller / "state" / "usage_ledger.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prior_sessions(events: list[dict[str, Any]], *, since_unix: float, exclude: str) -> dict[str, dict[str, Any]]:
    """Controller sessions started at or after the approval (except the current one): gross cost and allocation."""
    sessions: dict[str, dict[str, Any]] = {}
    for event in events:
        sid = event.get("session_id")
        if not sid or sid == exclude or "session_start" not in event:
            continue
        if stamp(event["session_start"]) < since_unix:
            continue
        record = sessions.setdefault(sid, {"gross_usd": 0.0, "allocated_seconds": 0.0, "stopped": False})
        if event.get("max_run_duration_s") is not None:
            record["allocated_seconds"] = max(record["allocated_seconds"], float(event["max_run_duration_s"]))
        if event.get("event") == "STOPPED":
            record["stopped"] = True
            record["gross_usd"] = max(record["gross_usd"], float(event.get("estimated_gross_cost_usd", 0.0)))
            record["elapsed_running_s"] = float(event.get("elapsed_running_s", 0.0))
    return sessions


def build_snapshots(prep: Prep, phase: str, *, controller: Path = CONTROLLER, now: float | None = None,
                    instance: dict[str, Any] | None = None, events: list[dict[str, Any]] | None = None,
                    price: dict[str, Any] | None = None) -> dict[str, Any]:
    """Refresh session.json and spending.json from the controller's own records plus the provider's scheduling."""
    now = time.time() if now is None else now
    policy = read_json(prep.policy(phase))
    state = controller_session(controller)
    config = read_config(controller)
    instance = describe_instance(config) if instance is None else instance
    require(instance.get("status") == "RUNNING" and str(instance.get("id")) == config["SELECTED_INSTANCE_ID"],
            "Provider does not report the selected instance RUNNING")
    scheduling = instance.get("scheduling") or {}
    max_run = float((scheduling.get("maxRunDuration") or {}).get("seconds") or 0)
    stop_action = scheduling.get("instanceTerminationAction")
    last_start = stamp(instance["lastStartTimestamp"])
    start = stamp(state["session_start"])
    maximum = float(state["max_run_duration_s"])
    deadline = stamp(state["termination_ts"])
    require(abs(deadline - (start + maximum)) <= 60, "Controller termination time is not start plus duration")
    deadline = min(deadline, start + maximum)  # the earlier bound is the one admission must respect
    provider_stop = last_start + max_run if max_run else 0.0
    verified = bool(max_run) and stop_action == "STOP" and deadline - 60 <= provider_stop <= deadline + 900
    require(verified, f"Provider automatic STOP not verified: action={stop_action}, maxRunDuration={max_run}, "
                      f"provider_stop={provider_stop:.0f}, controller_deadline={deadline:.0f}")
    owner = policy["controller_owner"]
    session = {"schema_version": 1, "session_id": state["session_id"], "instance_id": str(state["instance_id"]),
               "state": "RUNNING", "allocation_started_unix": start, "deadline_unix": deadline,
               "max_total_seconds": maximum, "captured_unix": now, "controller_owner": owner,
               "provider_hard_stop_verified": True,
               # the earlier of the controller deadline and the provider's own stop bounds admission
               "provider_hard_stop_unix": min(provider_stop, deadline),
               "provider_stop_action": stop_action, "provider_max_run_duration_seconds": max_run,
               "provider_last_start_unix": last_start, "captured_utc": utc_now()}
    price = read_json(controller / "state" / "price_cache.json") if price is None else price
    require(now - stamp(price["verified_at"]) <= 24 * 3600, "Price verification older than 24 hours; run vm-price")
    rate = float(price["gross_usd_per_hour"])
    events = ledger_events(controller) if events is None else events
    prior = prior_sessions(events, since_unix=float(policy["approval_unix"]), exclude=state["session_id"])
    require(all(record["stopped"] for record in prior.values()),
            f"A previous campaign session has no STOPPED record: {sorted(k for k, v in prior.items() if not v['stopped'])}")
    days = max(1.0, (now - float(policy["approval_unix"])) / 86400 + 2.0)
    storage = float(policy["storage_usd_per_day"]) * days
    spending = {"schema_version": 1, "session_id": state["session_id"], "captured_unix": now, "captured_utc": utc_now(),
                "gross_running_usd_per_hour": rate, "price_verified_at": price["verified_at"],
                "spent_before_session_usd": sum(r["gross_usd"] for r in prior.values()),
                "allocated_seconds_before_session": sum(r["allocated_seconds"] for r in prior.values()),
                "prior_sessions": prior, "storage_and_snapshot_forecast_usd": storage,
                "storage_basis": f"{policy['storage_usd_per_day']:.2f} USD/day (controller-conservative disk plus "
                                 f"two snapshots) over {days:.1f} days from approval to two days ahead"}
    session_path = write_json_atomic(prep.session, session)
    spending["session_sha256"] = sha256_file(session_path)
    write_json_atomic(prep.spending, spending)
    history = prep.authorization / "snapshots" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(now))
    history.mkdir(parents=True, exist_ok=True)
    shutil.copy2(prep.session, history / "session.json")
    shutil.copy2(prep.spending, history / "spending.json")
    return {"session": session, "spending": spending}


def local_admission(prep: Prep, phase: str, unit: str, *, now: float | None = None) -> dict[str, Any]:
    """The worker's pure authorization check, run on the workstation before a launch is attempted."""
    sys.path.insert(0, str(prep.code))
    import metal_completion_safety as safety  # noqa: E402  (the package copy: identical bytes to the manifest)

    now = time.time() if now is None else now
    manifest = read_json(prep.manifest(phase))
    try:
        runtime = safety.check_authorization(manifest=manifest, manifest_sha256=sha256_file(prep.manifest(phase)),
                                             policy=read_json(prep.policy(phase)), session=read_json(prep.session),
                                             spending=read_json(prep.spending), unit_name=unit, now=now)
    except ValueError as exc:  # the worker's own refusal reasons, surfaced as a coordinator refusal
        raise CoordinatorError(f"Worker admission would refuse {unit}: {exc}") from exc
    require(runtime["deadline_unix"] - now >= 1.25 * runtime["estimated_fit_seconds"] + 900 + LAUNCH_MARGIN_SECONDS,
            "Launch margin exhausted for this session")
    return runtime


# ---------------------------------------------------------------------------
# Remote operations
# ---------------------------------------------------------------------------

def check_ssh_config(controller: Path = CONTROLLER) -> Path:
    path = controller / "state" / "ssh_config"
    require(path.is_file(), f"{path} is missing; run the VM controller's bin/vm-setup --stages ssh")
    instance = read_config(controller).get("SELECTED_INSTANCE_ID", "")
    aliases = [line.split()[1] for line in path.read_text().splitlines() if line.strip().startswith("HostKeyAlias")]
    require(bool(instance) and aliases and all(alias.endswith("-" + instance) for alias in aliases),
            "ssh_config names another instance; run the VM controller's bin/vm-setup --stages ssh first")
    return path


class Remote:
    def __init__(self, ssh_config: Path, host: str = HOST):
        self.host = host
        self.transport = host_pull.SshTransport(host, ssh_config)
        self.ssh = self.transport.ssh + [host]

    def run(self, script: str, *, check: bool = True, timeout: float | None = 900) -> subprocess.CompletedProcess:
        result = subprocess.run(self.ssh + ["bash -c " + shlex.quote(script)], env=self.transport.env,
                                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)
        require(not check or result.returncode == 0,
                f"remote command failed ({result.returncode}): {result.stderr.strip()[-2000:]}")
        return result

    def rsync_to(self, local: Path, remote: str, *, excludes: tuple[str, ...] = ()) -> None:
        command = ["rsync", "-a", "--no-links", "--delete-excluded", *[f"--exclude={e}" for e in excludes],
                   "-e", shlex.join(self.transport.ssh), f"{local}/", f"{self.host}:{remote}/"]
        result = subprocess.run(command, env=self.transport.env, stdin=subprocess.DEVNULL, capture_output=True, text=True)
        require(result.returncode == 0, f"rsync upload failed: {result.stderr.strip()[-1000:]}")

    def rsync_from(self, remote: str, local: Path) -> None:
        local.mkdir(parents=True, exist_ok=True)
        command = ["rsync", "-a", "--no-links", "-e", shlex.join(self.transport.ssh), f"{self.host}:{remote}/", f"{local}/"]
        result = subprocess.run(command, env=self.transport.env, stdin=subprocess.DEVNULL, capture_output=True, text=True)
        require(result.returncode == 0, f"rsync download failed: {result.stderr.strip()[-1000:]}")


def deploy_script(prep: Prep) -> str:
    """Create the identical absolute paths on the host: real directories for the package, bind mounts for inputs."""
    q = shlex.quote
    lines = ["set -eu",
             f"sudo mkdir -p {q(str(prep.root))} {q(str(TRAIN_DIR))} {q(str(ESM_DIR))}",
             f"sudo chown -R \"$(id -un):$(id -gn)\" {q(str(prep.root.parent))} {q(str(TRAIN_DIR.parent.parent))} "
             f"{q(str(ESM_DIR.parent.parent.parent))}",
             f"mkdir -p {q(str(prep.code))} {q(str(prep.parent))} {q(str(prep.execution))}"]
    for source, target in ((REMOTE_TRAIN_SOURCE, str(TRAIN_DIR)), (REMOTE_ESM_SOURCE, str(ESM_DIR))):
        lines += [f"test -d {q(source)}",
                  f"if ! mountpoint -q {q(target)}; then sudo mount --bind {q(source)} {q(target)}; fi",
                  f"grep -qF {q(f'{source} {target} none bind')} /etc/fstab || "
                  f"echo {q(f'{source} {target} none bind 0 0')} | sudo tee -a /etc/fstab > /dev/null",
                  f"mountpoint -q {q(target)}"]
    lines.append("echo deployed-paths")
    return "\n".join(lines)


def verify_script(prep: Prep, phase: str) -> str:
    manifest_sha = sha256_file(prep.manifest(phase))
    codes = read_json(prep.manifest(phase))["code_sha256"]
    q = shlex.quote
    lines = ["set -eu", f"cd {q(str(prep.code))}"]
    for name, digest in sorted(codes.items()):
        lines.append(f"echo {q(f'{digest}  {name}')} | sha256sum -c --quiet -")
    lines.append(f"echo {q(f'{manifest_sha}  {prep.manifest(phase)}')} | sha256sum -c --quiet -")
    lines.append(f"test -f {q(str(TRAIN_DIR / 'structure_manifest.csv'))}")
    lines.append(f"test -d {q(str(ESM_DIR))}")
    lines.append(f"{q(REMOTE_PY)} -c 'import torch, torch_geometric; print(torch.__version__, torch_geometric.__version__, torch.cuda.is_available())'")
    lines.append("echo verified-package")
    return "\n".join(lines)


def plan_command(prep: Prep, phase: str, unit: str) -> list[str]:
    return [REMOTE_PY, "-B", str(prep.code / "run_metal_completion.py"), "--action", "plan",
            "--manifest", str(prep.manifest(phase)), "--manifest-sha256", sha256_file(prep.manifest(phase)),
            "--unit", unit, "--train-dir", str(TRAIN_DIR), "--esm-dir", str(ESM_DIR), "--python", REMOTE_PY]


def run_command(prep: Prep, phase: str, unit: str, lane: int) -> list[str]:
    return [REMOTE_PY, "-B", str(prep.code / "run_metal_completion.py"), "--action", "run",
            "--manifest", str(prep.manifest(phase)), "--manifest-sha256", sha256_file(prep.manifest(phase)),
            "--unit", unit, "--lane", str(lane), "--train-dir", str(TRAIN_DIR), "--esm-dir", str(ESM_DIR),
            "--python", REMOTE_PY, "--device", "cuda",
            "--policy", str(prep.policy(phase)), "--session", str(prep.session), "--spending", str(prep.spending)]


def launch_script(prep: Prep, phase: str, unit: str, lane: int, session_id: str) -> str:
    """One detached wrapper per unit: rotates earlier launch files, records the launch, writes <unit>.rc last."""
    q = shlex.quote
    base = f"{prep.launch}/{unit}"
    record = {"unit": unit, "lane": lane, "phase": phase, "session_id": session_id, "launched_at": utc_now(),
              "argv": run_command(prep, phase, unit, lane)}
    inner = (f"( {shlex.join(record['argv'])} > {q(base + '.out')} 2>&1; echo $? > {q(base + '.rc.tmp')}; "
             f"mv {q(base + '.rc.tmp')} {q(base + '.rc')} ) & echo $! > {q(base + '.pid')}\nwait")
    return "\n".join([
        "set -eu", f"cd {q(str(prep.code))}", f"mkdir -p {q(str(prep.launch))}",
        f"for f in {q(base)}.out {q(base)}.rc {q(base)}.pid {q(base)}.boot {q(base)}.launch.json; do "
        "if [ -e \"$f\" ]; then n=1; while [ -e \"$f.$n\" ]; do n=$((n+1)); done; mv \"$f\" \"$f.$n\"; fi; done",
        f"printf '%s\\n' {q(json.dumps(record, sort_keys=True))} > {q(base + '.launch.json')}",
        f"cat /proc/sys/kernel/random/boot_id > {q(base + '.boot')}",
        f"setsid nohup bash -c {q(inner)} > {q(base + '.launcher.log')} 2>&1 < /dev/null &",
        "echo launched"])


STATE_PROBE = r"""
import fcntl, json, sys, time
from pathlib import Path
root, launch_dir, grace = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
out = {"statuses": {}, "claims": [], "archived": {}, "launches": {}, "lanes": {}}
for path in sorted(root.glob("lanes/lane*/run_status_*.json")):
    record = json.loads(path.read_text())
    out["statuses"][record["run_name"]] = {"status": record.get("status"), "lane": record.get("lane"),
                                           "elapsed_seconds": record.get("elapsed_seconds")}
out["claims"] = sorted(path.name[:-5] for path in (root / "claims").glob("*.json"))
for directory in sorted((root / "failed_attempts").glob("*")):
    out["archived"][directory.name] = len(list(directory.glob("attempt*")))
for record_path in sorted(launch_dir.glob("*.launch.json")):
    name = record_path.name[:-len(".launch.json")]
    record = json.loads(record_path.read_text())
    rc, pid, boot_file = (launch_dir / f"{name}.{suffix}" for suffix in ("rc", "pid", "boot"))
    code = rc.read_text().strip() if rc.is_file() else None
    if code is not None:
        state = "exited"
    elif not boot_file.is_file() or boot_file.read_text().strip() != boot:
        state = "lost"
    elif not pid.is_file():
        state = "starting" if time.time() - record_path.stat().st_mtime < grace else "lost"
    else:
        try:
            alive = name.encode() in Path(f"/proc/{int(pid.read_text())}/cmdline").read_bytes()
        except (OSError, ValueError):
            alive = False
        if not alive and rc.is_file():
            code = rc.read_text().strip()
        state = "running" if alive else "exited" if code is not None else "lost"
    out["launches"][name] = {"unit": record.get("unit"), "lane": record.get("lane"), "phase": record.get("phase"),
                             "session_id": record.get("session_id"), "state": state, "rc": code}
for lane in sys.argv[4:]:
    lane_root = root / "lanes" / f"lane{lane}"
    state_path, lock_path = lane_root / "execution_state.json", lane_root / "execution.lock"
    info = {"exists": state_path.is_file(), "lock_free": True}
    if lock_path.is_file():
        with lock_path.open("a+") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(handle, fcntl.LOCK_UN)
            except BlockingIOError:
                info["lock_free"] = False
    if info["exists"]:
        state = json.loads(state_path.read_text())
        pending, transfer, child = state.get("pending_unit"), state.get("pending_transfer"), state.get("active_child")
        info["pending_unit"] = {k: pending.get(k) for k in ("run_name", "status", "persisted")} if pending else None
        info["pending_transfer"] = bool(transfer)
        info["ack_uploaded"] = bool(transfer) and (lane_root / transfer["ack_path"]).is_file()
        alive = False
        if child:
            try:
                stat = Path(f"/proc/{int(child['pid'])}/stat").read_text()
                alive = stat[stat.rfind(")") + 2:].split()[19] == str(child.get("start_ticks"))
            except (OSError, ValueError, IndexError):
                alive = False
        info["active_child_alive"] = alive
    out["lanes"][f"lane{lane}"] = info
print(json.dumps(out))
"""


CUDA_CHECK = "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 3)"


def host_cuda_ready(remote: Remote) -> bool:
    """The GPU driver must be loaded in the host venv's view before any unit is launched."""
    result = remote.run(f"{shlex.quote(REMOTE_PY)} -c {shlex.quote(CUDA_CHECK)}", check=False, timeout=120)
    return result.returncode == 0


def probe(remote: Remote, prep: Prep, lanes: tuple[int, ...] = LANES) -> dict[str, Any]:
    script = (f"{shlex.quote(REMOTE_PY)} -c {shlex.quote(STATE_PROBE)} {shlex.quote(str(prep.execution))} "
              f"{shlex.quote(str(prep.launch))} {STARTING_GRACE_SECONDS} " + " ".join(str(k) for k in lanes))
    return json.loads(remote.run(script, timeout=300).stdout)


def lane_busy(info: dict[str, Any]) -> bool:
    return not info.get("lock_free", True) or bool(info.get("active_child_alive"))


def lanes_awaiting_pull(state: dict[str, Any]) -> list[int]:
    out = []
    for lane, info in sorted(state["lanes"].items()):
        pending = info.get("pending_unit") or {}
        if info.get("pending_transfer") and not info.get("ack_uploaded") and pending.get("status") != "running":
            out.append(int(lane[4:]))
    return out


def interrupted_lanes(state: dict[str, Any]) -> dict[int, str]:
    live = {record["lane"] for record in state["launches"].values() if record["state"] in ("running", "starting")}
    out = {}
    for lane, info in sorted(state["lanes"].items()):
        pending = info.get("pending_unit") or {}
        if (pending.get("status") == "running" and not pending.get("persisted") and not info.get("pending_transfer")
                and info.get("lock_free", True) and not info.get("active_child_alive") and int(lane[4:]) not in live):
            out[int(lane[4:])] = pending.get("run_name")
    return out


def place_lane(prep: Prep, lane: int) -> dict[str, Any]:
    """Copy the acknowledged durable lane copy to lanes/laneK (the layout the collector reads), byte-verified."""
    source, target = prep.host_pull / f"lane{lane}", prep.lane(lane)
    require(source.is_dir(), f"No pulled copy for lane {lane}")
    copied = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(source)
        if relative.parts[0] == "persistence_receipts" or relative.name in (".host_pull.lock",) \
                or relative.name.endswith(".partial"):
            continue
        destination = target / relative
        if destination.is_file() and sha256_file(destination) == sha256_file(path):
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".partial")
        shutil.copy2(path, partial)
        require(sha256_file(partial) == sha256_file(path), f"Copy differs: {relative}")
        os.replace(partial, destination)
        copied.append(str(relative))
    return {"lane": lane, "copied": len(copied), "target": str(target)}


def pull_lanes(remote: Remote, prep: Prep, lanes: list[int]) -> dict[str, Any]:
    result = host_pull.pull(remote.transport, str(prep.execution), prep.host_pull, lanes, source_root=prep.code)
    require(result["ok"], "Host pull failed for some lane (the lane stays closed): "
            + json.dumps({k: v for k, v in result["lanes"].items() if v["status"] == "error"}))
    placements = {}
    for lane in lanes:
        status = result["lanes"][f"lane{lane}"]["status"]
        if status in ("acknowledged", "already_acknowledged"):
            placements[f"lane{lane}"] = place_lane(prep, lane)
    return {"pull": result, "placement": placements}


def pull_root_records(remote: Remote, prep: Prep) -> None:
    for name in ("claims", "authorization_sessions", "launch", "failed_attempts"):
        if remote.run(f"test -d {shlex.quote(str(prep.execution / name))}", check=False).returncode == 0:
            remote.rsync_from(str(prep.execution / name), prep.execution / name)


ARCHIVE_FAILED = r"""
import json, sys
from pathlib import Path
code, root, name = sys.argv[1:4]
sys.path[:0] = [code, code + "/src"]
import pmm_v3_campaign as v3
paths = v3.V3Paths(Path(root))
target = v3.archive_failed_attempt(paths, name)
print(json.dumps({"archived": str(target)}))
"""

RECOVER_LANE = r"""
import importlib.util, json, sys
from pathlib import Path
code, lane_root, durable, session_id, deadline, started, max_seconds = sys.argv[1:8]
spec = importlib.util.spec_from_file_location("pmm_execution_recovery", Path(code) / "src/benchmarking/pmm_execution.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
policy = module.ExecutionPolicy(deadline_unix=float(deadline), max_total_seconds=float(max_seconds),
                                allocation_started_unix=float(started))
execution = module.CampaignExecution(Path(lane_root), policy=policy, session_id=session_id,
                                     durable_root=Path(durable), persistence_mode="host_pull")
try:
    with execution:
        print(json.dumps({"recovered": False, "message": "nothing to recover"}))
except module.ExecutionBlocked as exc:
    print(json.dumps({"recovered": True, "message": str(exc)}))
"""


def archive_failed(remote: Remote, prep: Prep, unit: str) -> dict[str, Any]:
    state = probe(remote, prep)
    require(unit not in {n for n, r in state["launches"].items() if r["state"] in ("running", "starting")},
            f"{unit} still has a running launch")
    require(not lanes_awaiting_pull(state), f"lanes {lanes_awaiting_pull(state)} await their host pull first")
    require(not any(run == unit for run in interrupted_lanes(state).values()), f"{unit} needs recover-lane first")
    out = remote.run(f"{shlex.quote(REMOTE_PY)} -B -c {shlex.quote(ARCHIVE_FAILED)} {shlex.quote(str(prep.code))} "
                     f"{shlex.quote(str(prep.execution))} {shlex.quote(unit)}").stdout
    return json.loads(out)


def recover_lane(remote: Remote, prep: Prep, lane: int) -> dict[str, Any]:
    state = probe(remote, prep)
    stranded = interrupted_lanes(state)
    require(lane in stranded, f"lane{lane} holds no interrupted unit; nothing to recover")
    session = read_json(prep.session)
    out = remote.run(f"{shlex.quote(REMOTE_PY)} -B -c {shlex.quote(RECOVER_LANE)} {shlex.quote(str(prep.code))} "
                     f"{shlex.quote(str(prep.lane(lane)))} {shlex.quote(str(prep.host_pull / f'lane{lane}'))} "
                     f"{shlex.quote(session['session_id'])} {float(session['deadline_unix'])!r} "
                     f"{float(session['allocation_started_unix'])!r} {float(session['max_total_seconds'])!r}").stdout
    recovered = json.loads(out)
    require(recovered["recovered"], f"lane{lane} recovery recorded nothing: {recovered}")
    pulled = pull_lanes(remote, prep, [lane])
    return {"lane": lane, "unit": stranded[lane], "recovery": recovered, "pull": pulled}


def upload_authorization(remote: Remote, prep: Prep) -> None:
    remote.run(f"mkdir -p {shlex.quote(str(prep.authorization))}")
    remote.rsync_to(prep.authorization, str(prep.authorization), excludes=("snapshots",))


def launch(remote: Remote, prep: Prep, phase: str, unit: str, lane: int) -> dict[str, Any]:
    """Refresh snapshots, check admission locally, upload the authorization files, start the wrapper."""
    permitted = read_json(prep.policy(phase))["units"]
    require(unit in permitted, f"{unit} is not a permitted unit of the {phase} policy")
    snapshots = build_snapshots(prep, phase)
    local_admission(prep, phase, unit)
    upload_authorization(remote, prep)
    out = remote.run(launch_script(prep, phase, unit, lane, snapshots["session"]["session_id"])).stdout
    require("launched" in out, f"launch did not confirm: {out}")
    return {"unit": unit, "lane": lane, "session_id": snapshots["session"]["session_id"], "launched_at": utc_now()}


def run_session(remote: Remote, prep: Prep, phase: str, queue: list[str], *, lanes: tuple[int, ...] = LANES,
                poll: float = POLL_SECONDS, log_path: Path | None = None, sleep=time.sleep) -> dict[str, Any]:
    """Keep every lane busy with the queue until the admission window closes, pulling after each unit."""
    permitted = set(read_json(prep.policy(phase))["units"])
    require(set(queue) <= permitted, "Queue contains units outside the policy")
    log_path = log_path or prep.evidence / "sessions" / f"{phase}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    launched: dict[str, dict[str, Any]] = {}
    ended: dict[str, dict[str, Any]] = {}
    reruns: dict[str, int] = {}
    rerun_decided: set[str] = set()
    closed = False

    def note(event: str, **fields: Any) -> None:
        with log_path.open("a") as handle:
            handle.write(json.dumps({"event": event, "at": utc_now(), **fields}, sort_keys=True) + "\n")

    note("session_begin", queue=list(queue), lanes=list(lanes))
    require(host_cuda_ready(remote), "Host CUDA is not available; repair the driver before launching any unit")
    note("host_cuda_ready")
    while True:
        state = probe(remote, prep, lanes)
        stranded = interrupted_lanes(state)
        for lane, unit in stranded.items():
            # a dead worker's unit is recorded as interrupted and offered for pull; it is never launched over
            recovered = recover_lane(remote, prep, lane)
            note("recovered_lane", lane=lane, unit=unit, pull=recovered["pull"]["pull"]["lanes"][f"lane{lane}"]["status"])
        if stranded:
            state = probe(remote, prep, lanes)
        waiting = lanes_awaiting_pull(state)
        for lane in waiting:
            pulled = pull_lanes(remote, prep, [lane])
            note("pulled", lane=lane, result=pulled["pull"]["lanes"][f"lane{lane}"]["status"],
                 placed=pulled["placement"].get(f"lane{lane}", {}).get("copied"))
        if waiting:
            state = probe(remote, prep, lanes)
        for name, record in state["launches"].items():
            if record["state"] == "exited" and name not in ended and name in launched:
                status = (state["statuses"].get(name) or {}).get("status")
                ended[name] = {"rc": record["rc"], "status": status}
                note("unit_exited", unit=name, rc=record["rc"], status=status, lane=record["lane"])
        if not lanes_awaiting_pull(state):
            for name, info in list(ended.items()):
                if info["rc"] == "0" or name in rerun_decided:
                    continue
                rerun_decided.add(name)
                if reruns.get(name, 0) >= 1:
                    note("rerun_exhausted", unit=name)
                    continue
                # one unchanged rerun after the lane was pulled and the attempt archived (v3 contract)
                try:
                    archived = archive_failed(remote, prep, name)
                except (CoordinatorError, subprocess.SubprocessError, ValueError) as exc:
                    note("rerun_refused", unit=name, error=str(exc)[:500])
                    continue
                reruns[name] = 1
                rerun_decided.discard(name)
                queue.insert(0, name)
                launched.pop(name, None)
                ended.pop(name, None)
                note("rerun_queued", unit=name, archived=archived)
        running = {n for n, r in state["launches"].items() if r["state"] in ("running", "starting")}
        busy_lanes = {r["lane"] for n, r in state["launches"].items() if n in running}
        for lane, info in state["lanes"].items():
            # an uploaded acknowledgement is accepted by the next worker on entry; only a missing one blocks
            if lane_busy(info) or (info.get("pending_transfer") and not info.get("ack_uploaded")):
                busy_lanes.add(int(lane[4:]))
        pending_reruns = [n for n, i in ended.items() if i["rc"] != "0" and n not in rerun_decided]
        if closed or not queue:
            if not running and not pending_reruns and not lanes_awaiting_pull(state):
                break
        else:
            for lane in lanes:
                if lane in busy_lanes or not queue:
                    continue
                unit = queue[0]
                try:
                    result = launch(remote, prep, phase, unit, lane)
                except CoordinatorError as exc:
                    if "Launch margin exhausted" in str(exc) or "Insufficient time" in str(exc):
                        closed = True
                        note("window_closed", reason=str(exc)[:300])
                        break
                    note("launch_refused", unit=unit, lane=lane, error=str(exc)[:500])
                    closed = True
                    break
                queue.pop(0)
                launched[unit] = result
                busy_lanes.add(lane)
                note("launched", **result)
        sleep(poll)
    state = probe(remote, prep, lanes)
    for lane in lanes_awaiting_pull(state):
        pulled = pull_lanes(remote, prep, [lane])
        note("pulled", lane=lane, result=pulled["pull"]["lanes"][f"lane{lane}"]["status"])
    pull_root_records(remote, prep)
    summary = {"launched": launched, "ended": ended, "remaining_queue": list(queue), "closed": closed,
               "log": str(log_path), "finished_at": utc_now()}
    note("session_end", **{k: v for k, v in summary.items() if k != "log"})
    return summary


def collect_map(prep: Prep, phase: str) -> dict[str, dict[str, str]]:
    """Run map for completed new units: exit file 0 and a completed status in the placed lane copy."""
    units = new_units(prep, phase)
    out = {}
    for unit in units:
        rc_path = prep.launch / f"{unit}.rc"
        if not rc_path.is_file() or rc_path.read_text().strip() != "0":
            continue
        for lane in LANES:
            status_path = prep.lane(lane) / f"run_status_{unit}.json"
            if status_path.is_file() and read_json(status_path).get("status") == "completed":
                out[unit] = {"run_dir": str(prep.lane(lane) / "runs" / unit), "status_path": str(status_path),
                             "worker_rc_path": str(rc_path)}
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prep", type=Path, default=DEFAULT_PREP)
    parser.add_argument("--phase", default="discovery", choices=("discovery", "confirmation"))
    sub = parser.add_subparsers(dest="action", required=True)
    approve = sub.add_parser("record-approval")
    approve.add_argument("--decision-file", type=Path, required=True)
    approve.add_argument("--decision-sentence", required=True)
    approve.add_argument("--gross-ceiling-usd", type=float, required=True)
    approve.add_argument("--max-sessions", type=int, required=True)
    approve.add_argument("--max-session-seconds", type=float, default=4 * 3600)
    approve.add_argument("--max-total-gpu-seconds", type=float, required=True)
    approve.add_argument("--lanes", type=int, default=3)
    approve.add_argument("--estimated-fit-seconds", type=float, default=FIT_SECONDS)
    approve.add_argument("--reserve-usd", type=float, default=0.5)
    approve.add_argument("--expires-days", type=float, default=7.0)
    approve.add_argument("--controller-owner",
                         default=os.environ.get("DEEPMZYME_CONTROLLER_OWNER")
                         or f"{getpass.getuser()}@{socket.gethostname()}")
    approve.add_argument("--superseded", nargs="*", default=[])
    approve.add_argument("--approval-utc", required=True, help="ISO-8601 time of the maintainer's decision")
    approve.add_argument("--storage-usd-per-day", type=float, default=1.0)
    approve.add_argument("--units", nargs="*", help="default: every grid unit without a verified control")
    sub.add_parser("units")
    sub.add_parser("snapshot")
    check = sub.add_parser("check")
    check.add_argument("--unit", required=True)
    deploy = sub.add_parser("deploy")
    deploy.add_argument("--plan-unit", required=True)
    one = sub.add_parser("launch")
    one.add_argument("requests", nargs="+", help="UNIT@LANE")
    sub.add_parser("status")
    pull = sub.add_parser("pull")
    pull.add_argument("--lanes", type=int, nargs="+", default=list(LANES))
    rec = sub.add_parser("recover-lane")
    rec.add_argument("lane", type=int)
    arc = sub.add_parser("archive-failed")
    arc.add_argument("unit")
    sess = sub.add_parser("run-session")
    sess.add_argument("--units", nargs="*", help="default: permitted units without a completed status, in order")
    sub.add_parser("collect-map")
    args = parser.parse_args(argv)
    prep = Prep(args.prep)

    def remote() -> Remote:
        return Remote(check_ssh_config())

    if args.action == "record-approval":
        units = args.units or new_units(prep, args.phase)
        path = record_approval(prep, phase=args.phase, decision_file=args.decision_file,
                               decision_sentence=args.decision_sentence, units=units,
                               gross_ceiling_usd=args.gross_ceiling_usd, max_sessions=args.max_sessions,
                               max_session_seconds=args.max_session_seconds,
                               max_total_gpu_seconds=args.max_total_gpu_seconds, lanes=args.lanes,
                               estimated_fit_seconds=args.estimated_fit_seconds, reserve_usd=args.reserve_usd,
                               expires_unix=time.time() + args.expires_days * 86400,
                               controller_owner=args.controller_owner, superseded=args.superseded,
                               approval_unix=stamp(args.approval_utc), storage_usd_per_day=args.storage_usd_per_day)
        result = {"policy": str(path), "policy_sha256": sha256_file(path), "units": len(units)}
    elif args.action == "units":
        result = {"new_units": new_units(prep, args.phase)}
    elif args.action == "snapshot":
        result = build_snapshots(prep, args.phase)
    elif args.action == "check":
        result = local_admission(prep, args.phase, args.unit)
    elif args.action == "deploy":
        r = remote()
        r.run(deploy_script(prep), timeout=600)
        r.rsync_to(prep.code, str(prep.code), excludes=("__pycache__",))
        r.rsync_to(prep.parent, str(prep.parent))
        r.run(f"mkdir -p {shlex.quote(str(prep.execution))}")
        r.transport.put(prep.manifest(args.phase), str(prep.manifest(args.phase)))
        receipt_path = read_json(prep.manifest(args.phase)).get("selection_receipt_path")
        if receipt_path:  # confirmation binds the discovery selection receipt at its identical path
            r.transport.put(Path(receipt_path), receipt_path)
        r.run(verify_script(prep, args.phase), timeout=600)
        plan = r.run(shlex.join(plan_command(prep, args.phase, args.plan_unit)), timeout=1800).stdout
        receipt = json.loads(plan)
        require(receipt.get("unit") == args.plan_unit and receipt.get("execution_authorized") is False, "Unexpected plan output")
        out = prep.evidence / f"vm_plan_{args.plan_unit}.json"
        out.write_text(plan)
        result = {"deployed": True, "plan_receipt": str(out), "identity_manifest": receipt["identity"]["completion_manifest_sha256"]}
    elif args.action == "launch":
        r = remote()
        result = []
        for item in args.requests:
            unit, lane = item.rsplit("@", 1)
            result.append(launch(r, prep, args.phase, unit, int(lane)))
    elif args.action == "status":
        result = probe(remote(), prep)
    elif args.action == "pull":
        result = pull_lanes(remote(), prep, args.lanes)
    elif args.action == "recover-lane":
        result = recover_lane(remote(), prep, args.lane)
    elif args.action == "archive-failed":
        result = archive_failed(remote(), prep, args.unit)
    elif args.action == "run-session":
        r = remote()
        if args.units:
            queue = list(args.units)
        else:
            state = probe(r, prep)
            done = {n for n, s in state["statuses"].items() if s.get("status") == "completed"}
            queue = [u for u in read_json(prep.policy(args.phase))["units"] if u not in done]
        result = run_session(r, prep, args.phase, queue)
    else:
        result = collect_map(prep, args.phase)
        write_json_atomic(prep.execution / f"runs_map_{args.phase}.json", result)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CoordinatorError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
