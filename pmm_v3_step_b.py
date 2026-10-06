#!/usr/bin/env python3
"""Workstation launcher for v3 plan step B (standard library only).

It never starts, stops, restores or extends a VM: the controller (~/deepmzyme-vm/bin) owns billing, and a
GPU start needs the user's typed authorization. Every command below talks to an already RUNNING VM over
the controller's IAP SSH configuration (BatchMode, no automatic retries of a launch).

Commands (in step-B order; see the metal playbook, "PMM ion metal v3 campaign"):
  session                 print the controller session (refuses if inactive, expired or mismatched)
  verify-inputs           read-only check of the restored disk (GPU, venv, v2 root, ESMC count, train dir)
  push-bundle --bundle-dir DIR   copy a built bundle and apply it on the VM (hashes re-verified there)
  prepare                 prepare the v3 campaign root once from the applied code
  sampler-start | sampler-check | sampler-stop    host sampler for the whole session (detached)
  step NAME [--fit-seconds S]    run one manifest step: a serial probe, a batch started together in its
                          lanes, a full run, or a permitted whole-batch retry; waits for the runners and
                          then pulls and acknowledges every lane used
  wait NAME / pull [--lanes ...] resume waiting or pulling after a dropped connection
  report                  run the speed report on the VM and keep a dated copy on the workstation
  evidence                copy the step-B evidence (manifest, settings, claims, step_b/, lane state and
                          events, statuses, acknowledgments, archive receipts, bundle receipt) to the
                          workstation with SHA-256 checked on both ends; run before every vm-stop

Each runner and the sampler start under setsid/nohup on the VM, so an SSH drop does not stop them.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pmm_v3_host_pull as host_pull  # noqa: E402
import pmm_v3_probes as probes  # noqa: E402

CONTROLLER = Path.home() / "deepmzyme-vm"
HOST = "deepmzyme-vm"
REMOTE_HOME = "/home/mechti"
REMOTE_CODE = f"{REMOTE_HOME}/projects/DeepMzyme_v3"
REMOTE_PY = f"{REMOTE_HOME}/venvs/deepmzyme/bin/python"
REMOTE_V3 = f"{REMOTE_HOME}/deepmzyme_runs/pmm_ion_metal_v3"
REMOTE_V2 = f"{REMOTE_HOME}/deepmzyme_runs/pmm_ion_metal_v2_context"
REMOTE_TRAIN = f"{REMOTE_HOME}/deepmzyme_data/pmm/train_and_test_sets_structures_zenodo_pmm_exact/train"
REMOTE_ESM = f"{REMOTE_V2}/inputs/esm_embeddings_esmc600m_v1"
REMOTE_FOLDS_PARENT = f"{REMOTE_HOME}/deepmzyme_runs/pmm_ion_metal_v3_folds"
REMOTE_FOLDS = f"{REMOTE_FOLDS_PARENT}/v3-seqid90-s42-b2"
REMOTE_BUNDLES = f"{REMOTE_HOME}/deepmzyme_runs/pmm_ion_metal_v3_bundles"
DURABLE = Path("/media/mechti/Data1/DeepMzyme_Data/campaigns/pmm_ion_metal_v3/durable")
EVIDENCE = Path("/media/mechti/Data1/DeepMzyme_Data/campaigns/pmm_ion_metal_v3/step_b_evidence")
LAUNCH_DIR = f"{REMOTE_V3}/step_b/launch"
# Admission forecasts (seconds) until measured values replace them (--fit-seconds). The runner admits a
# unit only if 1.25 x forecast + 900 s remain before the session's hard stop.
FIT_SECONDS = {"cold": 2700, "short": 900, "full": 2100}
LAUNCH_MARGIN_SECONDS = 60
SAMPLER_FRESH_SECONDS = 15
POLL_SECONDS = 20


class LauncherError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LauncherError(message)


def steps() -> dict[str, tuple[str, ...]]:
    """Manifest steps in plan order: a name -> the probe tags started together."""
    out: dict[str, tuple[str, ...]] = {}
    for (precision, lanes), tags in probes.BATCHES.items():
        if lanes == 1:
            out.update({tag: (tag,) for tag in tags})
        else:
            out[f"w{lanes}-{precision}"] = tags
            out[f"w{lanes}-{precision}-retry"] = probes.retry_tags(precision, lanes)
    out.update({tag: (tag,) for tag in probes.FULL.values()})
    return out


def default_fit_seconds(name: str) -> int:
    if name == "w1-fp32-r1":
        return FIT_SECONDS["cold"]  # builds the parse and graph caches
    return FIT_SECONDS["full"] if name in probes.FULL.values() else FIT_SECONDS["short"]


# ---------------------------------------------------------------------------
# Controller session and SSH configuration (read-only)
# ---------------------------------------------------------------------------

def read_config(controller: Path = CONTROLLER) -> dict[str, str]:
    lines = (controller / "config.env").read_text().splitlines()
    return dict(line.split("=", 1) for line in lines if line and not line.startswith("#") and "=" in line)


def _stamp(value: str) -> float:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def read_session(controller: Path = CONTROLLER, *, now: float | None = None) -> dict[str, Any]:
    """The active controller session as runner allocation fields (unix seconds)."""
    now = time.time() if now is None else now
    state = json.loads((controller / "state" / "current_session.json").read_text())
    require(state.get("active") is True, "No active controller session: the VM is not running under the controller")
    config = read_config(controller)
    for key, setting in (("project", "PROJECT_ID"), ("zone", "ZONE"), ("vm_name", "VM_NAME"),
                         ("instance_id", "SELECTED_INSTANCE_ID")):
        require(str(state.get(key)) == config.get(setting), f"Session {key} differs from config.env {setting}")
    start, deadline = _stamp(state["session_start"]), _stamp(state["termination_ts"])
    require(start <= now < deadline, "The session has not started or its hard stop has passed")
    return {"session_id": str(state["session_id"]), "allocation_started": start, "deadline": deadline,
            "max_seconds": float(state["max_run_duration_s"]), "remaining_seconds": deadline - now,
            "instance_id": config["SELECTED_INSTANCE_ID"]}


def check_ssh_config(controller: Path = CONTROLLER) -> Path:
    """The IAP SSH config must name the current instance (vm-setup --stages ssh rewrites it after a restore)."""
    path = controller / "state" / "ssh_config"
    require(path.is_file(), f"{path} is missing; run ~/deepmzyme-vm/bin/vm-setup --stages ssh")
    instance = read_config(controller).get("SELECTED_INSTANCE_ID", "")
    aliases = [line.split()[1] for line in path.read_text().splitlines() if line.strip().startswith("HostKeyAlias")]
    require(bool(instance) and aliases and all(alias.endswith("-" + instance) for alias in aliases),
            "ssh_config names another instance; run ~/deepmzyme-vm/bin/vm-setup --stages ssh first")
    return path


class Remote:
    """Runs bash scripts on the VM over SSH. Tests replace it with a recorder."""

    def __init__(self, ssh_config: Path, host: str = HOST):
        self.ssh = ["ssh", "-F", str(ssh_config), "-o", "BatchMode=yes", host]
        self.transport = host_pull.SshTransport(host, ssh_config)

    def run(self, script: str, *, check: bool = True, timeout: float | None = 600) -> subprocess.CompletedProcess:
        result = subprocess.run(self.ssh + ["bash -c " + shlex.quote(script)], env=self.transport.env,
                                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)
        require(not check or result.returncode == 0,
                f"remote command failed ({result.returncode}): {result.stderr.strip()[-2000:]}")
        return result


# ---------------------------------------------------------------------------
# Commands built for the VM
# ---------------------------------------------------------------------------

def allocation_args(session: dict[str, Any], *, fit_seconds: float) -> list[str]:
    """Runner allocation and persistence flags for the controller session (shared with pmm_v3_step_c)."""
    return ["--device", "cuda", "--session-id", session["session_id"],
            "--allocation-started", repr(float(session["allocation_started"])),
            "--execution-deadline", repr(float(session["deadline"])),
            "--execution-max-seconds", repr(float(session["max_seconds"])),
            "--estimated-fit-seconds", repr(float(fit_seconds)),
            "--durable-root", str(DURABLE), "--persistence-mode", "host_pull"]


def run_command(tag: str, session: dict[str, Any], *, fit_seconds: float) -> list[str]:
    """The runner command of one probe, exactly as the manifest fixes it."""
    spec = probes.probe_spec(tag)
    command = [REMOTE_PY, "-u", "run_pmm_v3_campaign.py", "--action", "run", "--campaign-dir", REMOTE_V3,
               "--train-dir", REMOTE_TRAIN, "--esm-dir", REMOTE_ESM, "--unit", probes.PROBE_UNIT_NAME,
               "--probe", tag, "--amp", "on" if spec["amp"] else "off", "--lane", str(spec["lane"]),
               *allocation_args(session, fit_seconds=fit_seconds)]
    if spec["kind"] != "full":
        command += ["--epochs", str(spec["epochs"])]
    return command


def launch_script(name: str, tags: tuple[str, ...], session: dict[str, Any], *, fit_seconds: float) -> str:
    """One detached shell that starts every member at once and records each exit code."""
    lines = ["set -eu", f"cd {shlex.quote(REMOTE_CODE)}", f"mkdir -p {shlex.quote(LAUNCH_DIR)}"]
    for tag in tags:
        base = f"{LAUNCH_DIR}/{tag}"
        lines.append(f"for f in {shlex.quote(base)}.out {shlex.quote(base)}.rc; do "
                     f"if [ -e \"$f\" ]; then n=1; while [ -e \"$f.$n\" ]; do n=$((n+1)); done; mv \"$f\" \"$f.$n\"; fi; done")
    members = []
    for tag in tags:
        base = f"{LAUNCH_DIR}/{tag}"
        command = shlex.join(run_command(tag, session, fit_seconds=fit_seconds))
        members.append(f"( {command} > {shlex.quote(base + '.out')} 2>&1; echo $? > {shlex.quote(base + '.rc.tmp')}; "
                       f"mv {shlex.quote(base + '.rc.tmp')} {shlex.quote(base + '.rc')} ) &")
    inner = "\n".join(members + ["wait"])
    lines.append(f"setsid nohup bash -c {shlex.quote(inner)} > {shlex.quote(LAUNCH_DIR + '/' + name + '.launcher.log')} "
                 f"2>&1 < /dev/null &")
    lines.append("echo launched")
    return "\n".join(lines)


def admission_check(session: dict[str, Any], fit_seconds: float) -> None:
    needed = 1.25 * fit_seconds + 900 + LAUNCH_MARGIN_SECONDS
    require(session["remaining_seconds"] >= needed,
            f"Only {session['remaining_seconds']:.0f} s remain before the hard stop; this step needs {needed:.0f} s "
            "(1.25 x forecast + 900 s + launch margin). Stop here (evidence, then vm-stop).")


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

LANE_PROBE = r"""
import fcntl, json, os, sys
from pathlib import Path
root, out = Path(sys.argv[1]), {}
for lane in sys.argv[2:]:
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
    out[f"lane{lane}"] = info
print(json.dumps(out))
"""


def lane_blockers(states: dict[str, dict[str, Any]]) -> list[str]:
    """Reasons a lane cannot admit a new unit now (each would refuse its member after preflight)."""
    problems = []
    for lane, info in sorted(states.items()):
        if not info.get("lock_free", True):
            problems.append(f"{lane}: another worker holds the lane lock")
        if not info.get("exists"):
            continue
        pending = info.get("pending_unit")
        if info.get("active_child_alive"):
            problems.append(f"{lane}: a training child is still alive")
        if info.get("pending_transfer") and not info.get("ack_uploaded"):
            problems.append(f"{lane}: the last unit awaits its host pull (run 'pull --lanes {lane[4:]}')")
        if pending and not pending.get("persisted"):
            if pending.get("status") == "running":
                problems.append(f"{lane}: {pending.get('run_name')} is running or was interrupted "
                                "(wait, or archive-failed after checking it stopped)")
            elif not info.get("pending_transfer"):
                problems.append(f"{lane}: {pending.get('run_name')} ended without a persisted copy; recover it first")
    return problems


def require_lanes_ready(remote: Remote, lanes: list[int], tags: tuple[str, ...]) -> None:
    states = json.loads(remote.run(f"{REMOTE_PY} -c {shlex.quote(LANE_PROBE)} {REMOTE_V3} "
                                   + " ".join(str(k) for k in lanes)).stdout)
    problems = lane_blockers(states)
    unfinished = [tag for tag, code in member_states(remote, tags, launched_only=True).items() if code is None]
    if unfinished:
        problems.append(f"an earlier launch of {unfinished} has no exit code yet (use 'wait', never relaunch)")
    require(not problems, "Lanes are not ready; nothing was launched:\n  " + "\n  ".join(problems))


SAMPLER_ALIVE = (f"pid=$(cat {REMOTE_V3}/step_b/sampler.pid 2>/dev/null || true); alive=no; "
                 "if [ -n \"$pid\" ] && tr '\\0' ' ' < /proc/$pid/cmdline 2>/dev/null | grep -q -- '--sample-host'; "
                 "then alive=yes; fi; ")


def sampler_status(remote: Remote) -> dict[str, Any]:
    script = (SAMPLER_ALIVE +
              f"age=$(( $(date +%s) - $(stat -c %Y {REMOTE_V3}/step_b/host.jsonl 2>/dev/null || echo 0) )); "
              "echo \"$alive $age\"")
    alive, age = remote.run(script).stdout.split()
    return {"alive": alive == "yes", "seconds_since_last_sample": int(age)}


def require_fresh_sampler(remote: Remote) -> None:
    status = sampler_status(remote)
    require(status["alive"] and status["seconds_since_last_sample"] <= SAMPLER_FRESH_SECONDS,
            f"The host sampler is not running or stale ({status}); run sampler-start first")


def cmd_sampler_start(remote: Remote, session: dict[str, Any]) -> dict[str, Any]:
    status = sampler_status(remote)
    require(not status["alive"], "A host sampler is already running")
    seconds = max(60, int(session["remaining_seconds"]))
    script = (f"set -eu; cd {REMOTE_CODE}; mkdir -p {REMOTE_V3}/step_b; "
              f"setsid nohup {REMOTE_PY} pmm_v3_speed_report.py --campaign-dir {REMOTE_V3} --sample-host "
              f"--seconds {seconds} >> {REMOTE_V3}/step_b/sampler.log 2>&1 < /dev/null & "
              f"echo $! > {REMOTE_V3}/step_b/sampler.pid; echo started")
    remote.run(script)
    return {"started": True, "seconds": seconds}


def cmd_sampler_stop(remote: Remote) -> dict[str, Any]:
    remote.run(SAMPLER_ALIVE + f"if [ \"$alive\" = yes ]; then kill \"$pid\"; fi; rm -f {REMOTE_V3}/step_b/sampler.pid; echo ok")
    return sampler_status(remote)


def member_states(remote: Remote, tags: tuple[str, ...], *, launched_only: bool = False) -> dict[str, str | None]:
    """Exit code per tag (None while running). ``launched_only`` reports only tags with a launch output."""
    script = "; ".join(f"if [ -e {LAUNCH_DIR}/{tag}.out ] || [ {int(not launched_only)} = 1 ]; then "
                       f"echo {tag} $(cat {LAUNCH_DIR}/{tag}.rc 2>/dev/null || echo -); fi" for tag in tags)
    states = {}
    for line in remote.run(script).stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in tags:
            states[parts[0]] = None if parts[1] == "-" or not parts[1].lstrip("-").isdigit() else parts[1]
        elif len(parts) == 1 and parts[0] in tags:  # exit code being written: not finished yet
            states[parts[0]] = None
    return states


def wait_members(remote: Remote, tags: tuple[str, ...], session: dict[str, Any], *, poll: float = POLL_SECONDS,
                 sleep=time.sleep) -> dict[str, str]:
    failures = 0
    while True:
        try:
            states = member_states(remote, tags)
            failures = 0
        except (LauncherError, subprocess.SubprocessError, OSError) as exc:
            failures += 1
            require(failures < 10, f"Lost contact with the VM ({exc}); the runners continue there. Resume with 'wait'.")
            states = {}
        if states and all(code is not None for code in states.values()):
            return states  # type: ignore[return-value]
        require(time.time() < session["deadline"], "The session hard stop passed while waiting")
        sleep(poll)


def lanes_of(tags: tuple[str, ...]) -> list[int]:
    return sorted({probes.probe_spec(tag)["lane"] for tag in tags})


def cmd_pull(remote: Remote, lanes: list[int]) -> dict[str, Any]:
    result = host_pull.pull(remote.transport, REMOTE_V3, DURABLE, lanes, source_root=ROOT)
    require(result["ok"], "Host pull failed for some lane (the lane stays closed): "
            + json.dumps({k: v for k, v in result["lanes"].items() if v["status"] == "error"}))
    return result


def cmd_step(remote: Remote, session: dict[str, Any], name: str, *, fit_seconds: float | None,
             sleep=time.sleep) -> dict[str, Any]:
    table = steps()
    require(name in table, f"Unknown step {name!r}; choose from {sorted(table)}")
    tags = table[name]
    fit = float(fit_seconds or default_fit_seconds(name))
    admission_check(session, fit)
    specs = [probes.probe_spec(tag) for tag in tags]
    require_lanes_ready(remote, lanes_of(tags), tags)
    if any(spec["kind"] == "batch" for spec in specs):
        require_fresh_sampler(remote)
    if any(spec["attempt"] == 2 for spec in specs):
        report = json.loads(remote.run(f"cd {REMOTE_CODE} && {REMOTE_PY} pmm_v3_speed_report.py "
                                       f"--campaign-dir {REMOTE_V3} >/dev/null && cat {REMOTE_V3}/step_b/speed_report.json").stdout)
        precision, lanes = specs[0]["batch"]
        require(f"{precision}x{lanes}" in report.get("retry_required", []),
                f"The report does not require a retry of {precision} x{lanes}; a retry is allowed only after an "
                "invalid first attempt")
    try:
        out = remote.run(launch_script(name, tags, session, fit_seconds=fit)).stdout
    except (LauncherError, subprocess.SubprocessError, OSError) as exc:
        raise LauncherError(f"The launch command failed ({exc}). The runners may have started: check with "
                            f"'wait {name}' and never relaunch the step") from exc
    require("launched" in out, f"launch did not confirm: {out}; check with 'wait {name}' before anything else")
    codes = wait_members(remote, tags, session, sleep=sleep)
    pulled = cmd_pull(remote, lanes_of(tags))
    return {"step": name, "tags": list(tags), "exit_codes": codes, "pull": pulled,
            "next": "rerun 'pull' if any lane shows an error; then 'report'"}


def cmd_report(remote: Remote) -> dict[str, Any]:
    result = remote.run(f"cd {REMOTE_CODE} && {REMOTE_PY} pmm_v3_speed_report.py --campaign-dir {REMOTE_V3}", check=False)
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    (EVIDENCE / "reports").mkdir(parents=True, exist_ok=True)
    if result.returncode == 0:
        data = remote.transport.read_bytes(f"{REMOTE_V3}/step_b/speed_report.json")
        require(data is not None, "the report ran but wrote no speed_report.json")
        saved = EVIDENCE / "reports" / f"speed_report_{stamp}.json"
        saved.write_bytes(data)
    else:  # a refusal writes nothing on the VM; keep the refusal, never a stale report under a new date
        saved = EVIDENCE / "reports" / f"speed_report_{stamp}.refused.txt"
        saved.write_text(result.stdout + "\n" + result.stderr)
    return {"returncode": result.returncode, "stdout": result.stdout[-6000:], "stderr": result.stderr[-2000:],
            "saved_copy": str(saved) if saved else None}


EVIDENCE_PATTERNS = ("campaign_manifest.json", "execution_settings.json", "fold_class_weights.json", "claims/*.json",
                     "step_b/**/*", "lanes/lane*/execution_state.json", "lanes/lane*/execution_events.jsonl",
                     "lanes/lane*/run_status_*.json", "lanes/lane*/persistence_receipts/*.ack.json",
                     "failed_attempts/**/*")
EVIDENCE_LISTER = r"""
import hashlib, json, sys
from pathlib import Path
root, code, patterns = Path(sys.argv[1]), Path(sys.argv[2]), json.loads(sys.argv[3])
out = {}
for pattern in patterns:
    for path in sorted(root.glob(pattern)):
        relative = str(path.relative_to(root))
        if (path.is_file() and not path.is_symlink() and relative != "step_b/host.jsonl"  # live file: snapshot below
                and not path.name.endswith((".tmp", ".partial"))):  # transient files renamed while copying
            out[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
receipt = code / "v3_bundle_apply_receipt.json"
extra = {"code/v3_bundle_apply_receipt.json": hashlib.sha256(receipt.read_bytes()).hexdigest()} if receipt.is_file() else {}
print(json.dumps({"campaign": out, "code": extra}))
"""


def cmd_evidence(remote: Remote, *, destination: Path | None = None, patterns: tuple[str, ...] = EVIDENCE_PATTERNS,
                 evidence_root: Path | None = None, code: str | None = None) -> dict[str, Any]:
    """Copy the evidence while no step runs; the sampler's live file is copied as a frozen snapshot.
    pmm_v3_step_c passes its own patterns and workstation folder (and, for steps D-E, the code directory)."""
    code = code or REMOTE_CODE
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if "step_b/**/*" in patterns:
        remote.run(f"if [ -f {REMOTE_V3}/step_b/host.jsonl ]; then mkdir -p {REMOTE_V3}/step_b/host_snapshots && "
                   f"cp {REMOTE_V3}/step_b/host.jsonl {REMOTE_V3}/step_b/host_snapshots/host_{stamp}.jsonl; fi")
    listing = json.loads(remote.run(f"{REMOTE_PY} -c {shlex.quote(EVIDENCE_LISTER)} {REMOTE_V3} {code} "
                                    f"{shlex.quote(json.dumps(list(patterns)))}").stdout)
    target = destination or (evidence_root or EVIDENCE) / f"evidence_{stamp}"
    target.mkdir(parents=True, exist_ok=False)
    campaign = listing["campaign"]
    if campaign:
        remote.transport.fetch(REMOTE_V3, sorted(campaign), target / "campaign")
    if listing["code"]:
        remote.transport.fetch(code, ["v3_bundle_apply_receipt.json"], target / "code")
    import hashlib

    mismatched = [name for name, digest in {**{f"campaign/{k}": v for k, v in campaign.items()}, **listing["code"]}.items()
                  if hashlib.sha256((target / name).read_bytes()).hexdigest() != digest]
    manifest = {"copied_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "remote_campaign": REMOTE_V3,
                "files": {**{f"campaign/{k}": v for k, v in campaign.items()}, **listing["code"]},
                "mismatched": mismatched}
    (target / "evidence_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    require(not mismatched, f"Evidence copies differ from the VM: {mismatched}")
    return {"evidence_dir": str(target), "files": len(manifest["files"])}


def cmd_push_bundle(remote: Remote, bundle_dir: Path) -> dict[str, Any]:
    manifest_path = bundle_dir / "v3_bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    with tarfile.open(bundle_dir / manifest["code"]["path"], "r:gz") as tar:
        script = tar.extractfile(f"{manifest['code_root']}/pmm_v3_bundle.py").read()
    staged = bundle_dir / "pmm_v3_bundle.py"
    if not staged.exists():
        staged.write_bytes(script)
    require(staged.read_bytes() == script, "pmm_v3_bundle.py next to the bundle differs from the bundled copy")
    remote_dir = f"{REMOTE_BUNDLES}/{manifest['git_commit'][:12]}"
    remote.run(f"mkdir -p {shlex.quote(remote_dir)} && test ! -e {shlex.quote(REMOTE_CODE)}")
    for name in ("v3_bundle_manifest.json", manifest["code"]["path"], manifest["folds"]["path"], "pmm_v3_bundle.py"):
        remote.transport.put(bundle_dir / name, f"{remote_dir}/{name}")
    applied = remote.run(f"cd {shlex.quote(remote_dir)} && {REMOTE_PY} pmm_v3_bundle.py apply --bundle-dir . "
                         f"--code-dest {REMOTE_CODE} --folds-parent {REMOTE_FOLDS_PARENT}").stdout
    return {"remote_bundle": remote_dir, "apply": json.loads(applied)}


def cmd_prepare(remote: Remote) -> dict[str, Any]:
    out = remote.run(f"cd {REMOTE_CODE} && {REMOTE_PY} run_pmm_v3_campaign.py --action prepare --campaign-dir {REMOTE_V3} "
                     f"--v2-root {REMOTE_V2} --fold-dir {REMOTE_FOLDS}").stdout
    return json.loads(out)


VERIFY_INPUTS = f"""
set -u
echo "disk: $(df -h / | tail -1)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "nvidia-smi FAILED"
{REMOTE_PY} -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())' || echo "torch FAILED"
for f in train_cohort.csv esm_generation_plan.csv feature_inventory.json campaign_manifest.json train_context_audit.json; do
  if [ -f {REMOTE_V2}/$f ]; then echo "v2 $f ok"; else echo "v2 $f MISSING"; fi
done
echo "esm files: $(ls {REMOTE_ESM} 2>/dev/null | wc -l)"
if [ -d {REMOTE_TRAIN} ]; then echo "train dir ok"; else echo "train dir MISSING"; fi
echo "load workers: ${{DEEPMZYME_LOAD_WORKERS:-unset}}"
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("session", "verify-inputs", "push-bundle", "prepare", "sampler-start",
                                            "sampler-check", "sampler-stop", "step", "wait", "pull", "report", "evidence",
                                            "steps"))
    parser.add_argument("name", nargs="?", help="step / wait: manifest step name (see 'steps')")
    parser.add_argument("--fit-seconds", type=float, help="step: admission forecast (default by step kind)")
    parser.add_argument("--lanes", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--bundle-dir", type=Path)
    args = parser.parse_args(argv)
    if args.command == "steps":
        print(json.dumps({name: list(tags) for name, tags in steps().items()}, indent=2))
        return 0
    session = read_session()
    if args.command == "session":
        print(json.dumps(session, indent=2, sort_keys=True))
        return 0
    remote = Remote(check_ssh_config())
    if args.command == "verify-inputs":
        print(remote.run(VERIFY_INPUTS, check=False).stdout)
        return 0
    actions = {
        "push-bundle": lambda: cmd_push_bundle(remote, args.bundle_dir),
        "prepare": lambda: cmd_prepare(remote),
        "sampler-start": lambda: cmd_sampler_start(remote, session),
        "sampler-check": lambda: sampler_status(remote),
        "sampler-stop": lambda: cmd_sampler_stop(remote),
        "step": lambda: cmd_step(remote, session, args.name, fit_seconds=args.fit_seconds),
        "wait": lambda: {"exit_codes": wait_members(remote, steps()[args.name], session),
                         "pull": cmd_pull(remote, lanes_of(steps()[args.name]))},
        "pull": lambda: cmd_pull(remote, args.lanes),
        "report": lambda: cmd_report(remote),
        "evidence": lambda: cmd_evidence(remote),
    }
    if args.command in ("step", "wait"):
        require(args.name is not None, f"{args.command} needs a step name (see 'steps')")
    if args.command == "push-bundle":
        require(args.bundle_dir is not None, "push-bundle needs --bundle-dir")
    print(json.dumps(actions[args.command](), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (LauncherError, host_pull.HostPullError) as exc:
        print(f"step-B launcher refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
