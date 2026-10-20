#!/usr/bin/env python3
"""Workstation launcher of the v3 10-epoch schedule comparison (extension 2, ``v3-ext2-sched10``; standard library only).

Rules: docs/campaigns/pmm_ion_metal_v3/sched10_assessment_spec.json (handoff section 11; log v3-039; this tool pins
the file's SHA-256 and refuses another). Four fits: the selected cell ``gvp_late_fusion__four_class__baseline`` on
folds 1-4, seed 42, with a complete 10-epoch cosine schedule and its terminal checkpoint, against the four verified
50-epoch fits of the same cell (reused, never rerun). The fits run through run_pmm_v3_sched10.py, which this launcher
copies next to the hash-frozen runner in the extension code directory ('deploy'), and through the step C lane,
wait, pull, recovery and evidence logic (pmm_v3_step_c) unchanged. Like the other launchers it never starts, stops,
restores or extends a VM: the external VM controller ($DEEPMZYME_VM_CONTROLLER/bin) owns billing.

Commands:
  units                   offline: the four fits, their controls and the development-fold reference
  gates                   offline: manifest, ceiling, authorization, spending and readiness records
  build-manifest --log-entry v3-NNN
                          CPU, once: binds the frozen specification, the A4 specification, the final E2 assessment and
                          its selection, the E2 manifest, extension 1, the runner files, this tooling (VM runner
                          included) and the evidence of every reused control and development-fold run
  record-ceiling --usd X --spent-usd S --spent-at UTC --spent-source TEXT --log-entry v3-NNN [--reserve-usd R]
                          once: the campaign ceiling every launch must respect (estimated gross charges, storage)
  record-authorization --log-entry v3-NNN
                          once, only after the user's GPU-start authorization for this comparison
  spend                   offline: estimated campaign charges now and the ceiling check
  deploy                  copy run_pmm_v3_sched10.py into the extension code directory on the running VM (only with
                          a verified manifest and the recorded authorization; never overwrites a different file)
  session / status / launch UNIT@LANE ... [--fit-seconds S] [--no-wait] / wait [UNIT ...] [--any] /
  pull [--lanes K ...] / recover-lane K / archive-failed UNIT / evidence [--allow-running]
                          as for step E2, with this comparison's launch folder (step_s10/launch on the VM) and
                          evidence folder (step_s10_evidence under DeepMzyme_Data/)

Every launch is refused before anything starts unless: the CPU readiness report and the VM extension record agree
with this worktree; the manifest verifies; the VM copy of run_pmm_v3_sched10.py is the bound one; the recorded
authorization names this manifest and this ceiling record; and the estimated campaign charges, with the running
session counted to its hard stop, plus the recorded reserve stay within the recorded ceiling. The assessment runs
on the workstation from the verified local copies (pmm_v3_sched10_assessment.py), after 'evidence'.
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
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import pmm_v3_step_b as b  # noqa: E402
import pmm_v3_step_c as c  # noqa: E402
import pmm_v3_step_d as d  # noqa: E402
import pmm_v3_step_e2 as e2  # noqa: E402

LauncherError = b.LauncherError
require = b.require
sha256_file = d.sha256_file
read_json = e2.read_json
write_json_once = e2.write_json_once

CAMPAIGN = d.CAMPAIGN
SPEC_DIR = ROOT / "docs" / "campaigns" / "pmm_ion_metal_v3"
SPEC_JSON = SPEC_DIR / "sched10_assessment_spec.json"
# Frozen on 2026-10-12 before any comparison fit (log v3-039); a different file is refused.
FROZEN_SPEC_SHA256 = "14c4b5721aa3d15ce8c14dc1e0bbac2c8b47edd9140daa9dfbb8a1391dd7cd7c"
A4_SPEC_FILES = e2.A4_SPEC_FILES
VM_RUNNER = "run_pmm_v3_sched10.py"
SCHEDULE_TOOL_FILES = ("pmm_v3_sched10.py", "pmm_v3_sched10_assessment.py", VM_RUNNER)
REUSED_FILES = ("pmm_v3_assessment.py", "pmm_v3_step_b.py", "pmm_v3_step_c.py", "pmm_v3_step_d.py",
                "pmm_v3_host_pull.py", "pmm_v3_step_e2.py", "pmm_v3_e2_assessment.py")
TOOL_FILES = SCHEDULE_TOOL_FILES + REUSED_FILES
SCHEDULE_ID = "v3-ext2-sched10"
STEP_KEY = "s10"
EVIDENCE_DIR = "step_s10_evidence"
MANIFEST_NAME, CEILING_NAME, AUTHORIZATION_NAME = "s10_manifest.json", "s10_ceiling.json", "s10_authorization.json"
E2_EVIDENCE = Path("step_e2_evidence")
E2_MANIFEST = E2_EVIDENCE / e2.MANIFEST_NAME
EVIDENCE_PATTERNS_S10 = tuple(f"step_{STEP_KEY}/**/*" if pattern == "step_c/**/*" else pattern
                              for pattern in c.EVIDENCE_PATTERNS) + ("campaign_extension.json",)
REMOTE_RUNNER = f"{d.REMOTE_CODE}/{VM_RUNNER}"
# Admission forecast of one 10-epoch late-fusion fit at three lanes, launch to verified pull: the seven complete
# step-B probes took 647-795 s of training, verification and replay plus 31-36 s preflight and 33-60 s host pull
# (log v3-032, r5 review: median 987 s launch to pull at three identical lanes), rounded up.
FIT_SECONDS = 1100.0
# Kept free after the last launch: step F on its longer (50-epoch) path, up to 2.25 VM hours ($1.93), and seven more
# calendar days of storage until step F completes ($3.85), rounded up (as for E2).
DEFAULT_RESERVE_USD = e2.DEFAULT_RESERVE_USD
DEFAULT_RESERVE_BASIS = e2.DEFAULT_RESERVE_BASIS


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# The frozen specification
# ---------------------------------------------------------------------------

def load_spec(path: Path | None = None, *, expected_sha256: str | None = None) -> dict[str, Any]:
    """The frozen specification (refused unless its SHA-256 is the one frozen in log v3-039)."""
    path = SPEC_JSON if path is None else path
    expected = FROZEN_SPEC_SHA256 if expected_sha256 is None else expected_sha256
    require(sha256_file(path) == expected, f"{path} differs from the frozen 10-epoch comparison specification")
    spec = read_json(path)
    require(spec.get("step") == "S10" and spec.get("id") == SCHEDULE_ID and spec.get("target") == "four_class"
            and list(spec.get("folds", [])) == [1, 2, 3, 4] and spec.get("seed") == 42
            and spec.get("candidate_epochs") == 10 and spec.get("control_epochs") == 50
            and spec.get("checkpoint_rule") == "terminal" and spec.get("lr_schedule") == "cosine"
            and spec.get("held_out_test_accessed") is False,
            "Not the 10-epoch comparison specification this launcher was built for")
    expected_units = [f"{spec['family']}__{spec['target']}__{spec['candidate_label']}__fold{k}__seed{spec['seed']}"
                      for k in spec["folds"]]
    expected_controls = [f"{spec['selection_cell']}__fold{k}__seed{spec['seed']}" for k in spec["folds"]]
    require(spec["units"] == expected_units and spec["controls"] == expected_controls
            and spec["selection_cell"] == f"{spec['family']}__{spec['target']}__{spec['recipe']}",
            "The specification's units or controls are not the selected cell's four folds")
    return spec


# ---------------------------------------------------------------------------
# The launch manifest (CPU, once)
# ---------------------------------------------------------------------------

def latest_e2_assessment(campaign: Path) -> Path | None:
    return e2.latest_kind(campaign / E2_EVIDENCE / "assessments", "E2")


def e2_assessment_problems(assessment: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    """The final E2 assessment must be complete and select the cell the specification names."""
    result = assessment.get("result") or {}
    problems = []
    if assessment.get("kind") != "E2" or result.get("status") != "complete":
        problems.append("not a complete E2 assessment")
    if (result.get("final_selection") or {}).get("selected") != spec["selection_cell"]:
        problems.append(f"its final selection is not {spec['selection_cell']}")
    if assessment.get("held_out_test_accessed") is not False or result.get("held_out_test_accessed") is not False:
        problems.append("held_out_test_accessed is not false")
    if (assessment.get("a4_parity") or {}).get("equal") is not True:
        problems.append("its A4 parity self-check did not pass")
    return problems


def development_evidence(campaign: Path, run_name: str) -> dict[str, Any]:
    """Checkpoint and prediction hashes of a fold-0 development run from its durable receipt and status record."""
    found = sorted((campaign / "durable").glob(f"lane*/runs/{run_name}/selected_checkpoint.json"))
    require(len(found) == 1, f"{run_name}: expected one durable run directory, found {len(found)}")
    receipt = read_json(found[0])
    status = read_json(found[0].parent.parent.parent / f"run_status_{run_name}.json")
    require(status.get("status") == "completed" and receipt.get("fit_status") == "completed"
            and receipt.get("checkpoint_rule") == "terminal" and receipt.get("reconciliation_status") == "match"
            and status.get("selected_checkpoint_sha256") == receipt.get("selected_checkpoint_sha256"),
            f"{run_name}: not a completed, reconciled terminal fit")
    return {"checkpoint_sha256": receipt["selected_checkpoint_sha256"],
            "predictions_sha256": receipt["validation_predictions"]["sha256"],
            "epochs": receipt["selected_epoch"], "lane": found[0].parent.parent.parent.name}


def build_manifest(campaign: Path = CAMPAIGN, *, log_entry: str, root: Path = ROOT) -> dict[str, Any]:
    """The comparison's launch manifest: refused unless every binding holds now."""
    require(str(log_entry).startswith("v3-"), "Name the campaign log entry (--log-entry v3-NNN)")
    spec = load_spec()
    a4 = {name: sha256_file(SPEC_DIR / name) for name in A4_SPEC_FILES}
    e2_path = latest_e2_assessment(campaign)
    require(e2_path is not None, "No final E2 assessment copy on the workstation")
    require(str(e2_path).endswith(spec["selection_source"]["e2_assessment"])
            and sha256_file(e2_path) == spec["selection_source"]["e2_assessment_sha256"],
            f"The newest E2 assessment {e2_path} is not the one the specification names")
    assessment = read_json(e2_path)
    problems = e2_assessment_problems(assessment, spec)
    require(not problems, f"The E2 assessment {e2_path}: {problems}")
    require(assessment.get("a4_spec_sha256") == a4, "The E2 assessment was made under another A4 specification")
    e2_manifest_path = campaign / E2_MANIFEST
    e2_manifest = read_json(e2_manifest_path)
    require(assessment["manifest"]["sha256"] == sha256_file(e2_manifest_path),
            "The E2 assessment does not bind the recorded E2 manifest")
    extension_path = campaign / e2.EXTENSION_COPY
    extension = read_json(extension_path)
    extension_sha = sha256_file(extension_path)
    require(extension.get("extension_id") == d.EXTENSION_ID and e2_manifest["extension"]["sha256"] == extension_sha,
            "The extension record copy is not the one the E2 manifest binds")
    runner = d.local_runner_sha256(root)
    require(runner == extension["runner_sha256"] == e2_manifest["runner_sha256"],
            "This worktree's runner files differ from the extension record's")
    controls = {}
    for name in spec["controls"]:
        evidence = assessment["unit_evidence"].get(name)
        bound = e2_manifest["reused_units"].get(name)
        require(evidence is not None and bound is not None
                and all(evidence[key] == bound[key] for key in ("checkpoint_sha256", "predictions_sha256")),
                f"{name}: the E2 assessment and the E2 manifest do not bind the same verified evidence")
        controls[name] = {**{key: evidence[key] for key in ("checkpoint_sha256", "predictions_sha256")},
                          "source": "final E2 assessment unit evidence (= E2 manifest reused unit)"}
    development = spec["development_fold"]
    dev_control_unit = f"{spec['selection_cell']}__fold0__seed42"
    dev = {"candidate": {"run": development["candidate_run"],
                         **development_evidence(campaign, development["candidate_run"])},
           "control": {"run": development["control_run"], "unit": dev_control_unit,
                       **development_evidence(campaign, development["control_run"])}}
    require(dev["candidate"]["epochs"] == spec["candidate_epochs"] and dev["control"]["epochs"] == spec["control_epochs"],
            "The development-fold runs do not have the specified epoch counts")
    control_evidence = assessment["unit_evidence"].get(dev_control_unit) or {}
    require(all(control_evidence.get(key) == dev["control"][key] for key in ("checkpoint_sha256", "predictions_sha256")),
            "The fold-0 50-epoch run is not the one the E2 assessment verified")
    return {
        "schema_version": 1, "step": "S10", "id": SCHEDULE_ID, "campaign_id": "pmm_ion_metal_v3",
        "created_at": utc_now(), "log_entry": log_entry, "git_commit": e2._git_commit(root),
        "spec": {"path": str(SPEC_JSON), "sha256": sha256_file(SPEC_JSON)},
        "a4_spec_sha256": a4,
        "e2_assessment": {"path": str(e2_path), "sha256": sha256_file(e2_path),
                          "selection": assessment["result"]["final_selection"]["selected"]},
        "e2_manifest": {"path": str(e2_manifest_path), "sha256": sha256_file(e2_manifest_path)},
        "extension": {"id": extension["extension_id"], "sha256": extension_sha, "copy": str(extension_path),
                      "parent_manifest_sha256": extension["parent"]["manifest_sha256"]},
        "runner_sha256": runner,
        "tooling_sha256": {name: sha256_file(root / name) for name in TOOL_FILES},
        "vm_runner": {"file": VM_RUNNER, "remote_path": REMOTE_RUNNER, "sha256": sha256_file(root / VM_RUNNER)},
        "units": list(spec["units"]), "controls": controls, "development_fold": dev,
        "remote_code": d.REMOTE_CODE, "launch_folder": f"step_{STEP_KEY}/launch", "held_out_access": False,
    }


def verify_manifest(manifest: dict[str, Any], *, root: Path = ROOT, vm_extension: dict[str, Any] | None = None,
                    vm_runner_sha256: str | None = None) -> list[str]:
    """Every binding of a recorded manifest still holds (empty list) or the reasons it does not."""
    problems = []
    try:
        spec = load_spec()
    except (LauncherError, OSError, ValueError) as exc:
        return [f"specification: {exc}"]
    if (manifest.get("spec") or {}).get("sha256") != FROZEN_SPEC_SHA256:
        problems.append("the manifest binds another specification")
    if manifest.get("a4_spec_sha256") != {name: sha256_file(SPEC_DIR / name) for name in A4_SPEC_FILES}:
        problems.append("the A4 specification differs from the manifest")
    for key in ("e2_assessment", "e2_manifest"):
        bound = manifest.get(key) or {}
        path = Path(bound.get("path", ""))
        if not path.is_file() or sha256_file(path) != bound.get("sha256"):
            problems.append(f"{key}: {path} is missing or differs from its bound SHA-256")
    extension = manifest.get("extension") or {}
    copy = Path(extension.get("copy", ""))
    if not copy.is_file() or sha256_file(copy) != extension.get("sha256"):
        problems.append("the workstation copy of the extension record differs from the manifest")
    if vm_extension is not None and (vm_extension.get("sha256") != extension.get("sha256")
                                     or vm_extension.get("runner_sha256") != manifest.get("runner_sha256")):
        problems.append("the VM extension record is not the one the manifest binds")
    if d.local_runner_sha256(root) != manifest.get("runner_sha256"):
        problems.append("this worktree's runner files differ from the manifest")
    if {name: sha256_file(root / name) for name in TOOL_FILES} != manifest.get("tooling_sha256"):
        problems.append("this comparison's tooling or a reused launcher/assessor module changed since the manifest "
                        "was recorded (record a new manifest and a new authorization)")
    if (manifest.get("vm_runner") or {}).get("sha256") != sha256_file(root / VM_RUNNER):
        problems.append("the bound VM runner differs from this worktree's run_pmm_v3_sched10.py")
    if vm_runner_sha256 is not None and vm_runner_sha256 != (manifest.get("vm_runner") or {}).get("sha256"):
        problems.append(f"the VM copy of {VM_RUNNER} is missing or not the bound one ('deploy')")
    if manifest.get("units") != spec["units"] or sorted(manifest.get("controls") or {}) != sorted(spec["controls"]):
        problems.append("the manifest's units or controls are not the specification's")
    return problems


# ---------------------------------------------------------------------------
# Ceiling and authorization (execution controls; never permission by themselves)
# ---------------------------------------------------------------------------

def ceiling_record(*, usd: float, spent_usd: float, spent_at: str, spent_source: str, log_entry: str,
                   reserve_usd: float = DEFAULT_RESERVE_USD, reserve_basis: str = DEFAULT_RESERVE_BASIS) -> dict[str, Any]:
    record = e2.ceiling_record(usd=usd, spent_usd=spent_usd, spent_at=spent_at, spent_source=spent_source,
                               log_entry=log_entry, reserve_usd=reserve_usd, reserve_basis=reserve_basis)
    record["scope"] = "campaign total, gross, storage included; 10-epoch comparison launches are refused above it"
    record["grants"] = "nothing: a GPU start needs the user's recorded authorization for this comparison"
    return record


def authorization_record(*, manifest_path: Path, ceiling_path: Path, log_entry: str) -> dict[str, Any]:
    record = e2.authorization_record(manifest_path=manifest_path, ceiling_path=ceiling_path, log_entry=log_entry)
    record["decision"] = "the user's GPU-start authorization for the 10-epoch schedule comparison"
    return record


def record_paths(campaign: Path) -> dict[str, Path]:
    evidence = campaign / EVIDENCE_DIR
    return {"manifest": evidence / MANIFEST_NAME, "ceiling": evidence / CEILING_NAME,
            "authorization": evidence / AUTHORIZATION_NAME}


# ---------------------------------------------------------------------------
# Launch gates
# ---------------------------------------------------------------------------

def remote_runner_sha256(remote: b.Remote) -> str | None:
    out = remote.run(f"if [ -f {shlex.quote(REMOTE_RUNNER)} ]; then sha256sum {shlex.quote(REMOTE_RUNNER)}; fi").stdout
    return out.split()[0] if out.strip() else None


def s10_blockers(names: list[str], state: dict[str, Any], *, campaign: Path, readiness: dict[str, Any] | None,
                 runner: dict[str, str], vm_runner: Callable[[], str | None], root: Path = ROOT,
                 spend: Callable[[dict[str, Any]], dict[str, Any]] = e2.estimated_spend) -> list[str]:
    problems = d.readiness_blockers(state, readiness, runner)
    paths = record_paths(campaign)
    manifest = d.read_record(paths["manifest"])
    if manifest is None:
        problems.append("no manifest ('build-manifest'); nothing of the comparison starts without it")
    else:
        problems += [f"manifest: {p}" for p in verify_manifest(manifest, root=root, vm_extension=state.get("extension"),
                                                               vm_runner_sha256=vm_runner() or "")]
    ceiling = d.read_record(paths["ceiling"])
    if ceiling is None:
        problems.append("no ceiling record ('record-ceiling'); the ceiling must be in the execution controls")
    authorization = d.read_record(paths["authorization"])
    if authorization is None:
        problems.append("no recorded GPU-start authorization for this comparison ('record-authorization' after the "
                        "user's decision is logged)")
    elif manifest is not None and ceiling is not None:
        if authorization.get("manifest_sha256") != sha256_file(paths["manifest"]):
            problems.append("the recorded authorization names another manifest")
        if authorization.get("ceiling_sha256") != sha256_file(paths["ceiling"]):
            problems.append("the recorded authorization names another ceiling record")
    if ceiling is not None and not problems:
        estimate = spend(ceiling)
        if not estimate["within_ceiling"]:
            problems.append(f"estimated campaign charges {estimate['committed_usd']:.2f} USD with the running session "
                            f"to its hard stop, plus the {estimate['reserve_usd']:.2f} USD reserve, exceed the "
                            f"recorded {estimate['ceiling_usd']:.2f} USD ceiling: stop and close out, or ask the user "
                            "to record a larger ceiling")
    return problems


def build_step(campaign: Path = CAMPAIGN, *, vm_runner: Callable[[], str | None], runner: dict[str, str] | None = None,
               root: Path = ROOT, spend: Callable[[dict[str, Any]], dict[str, Any]] = e2.estimated_spend) -> c.Step:
    """The step C lane logic with this comparison's four fits, gates, forecast and folders."""
    units = tuple(load_spec()["units"])
    readiness = d.latest_readiness(campaign)
    runner = d.local_runner_sha256(root) if runner is None else runner
    return c.Step(
        key=STEP_KEY, label="10-epoch schedule comparison",
        outside="the comparison admits only its four fits ('units'); other steps have their own launchers",
        second_failure="the comparison stops: a second failure blocks the decision (specification stopping rule)",
        done_hint="all four fits have run: 'evidence', vm-stop, then pmm_v3_sched10_assessment.py on the workstation",
        units=lambda state: units,
        batch_blockers=lambda names, state: s10_blockers(names, state, campaign=campaign, readiness=readiness,
                                                         runner=runner, vm_runner=vm_runner, root=root, spend=spend),
        forecast=lambda name, state: FIT_SECONDS,
        code=d.REMOTE_CODE, evidence=campaign / EVIDENCE_DIR, evidence_patterns=EVIDENCE_PATTERNS_S10)


# ---------------------------------------------------------------------------
# Commands built for the VM (pmm_v3_step_c.unit_command / launch_script / start with this comparison's runner)
# ---------------------------------------------------------------------------

def unit_command(name: str, lane: int, session: dict[str, Any], *, fit_seconds: float) -> list[str]:
    return [b.REMOTE_PY, "-u", VM_RUNNER, "--action", "run", "--campaign-dir", b.REMOTE_V3, "--unit", name,
            "--train-dir", b.REMOTE_TRAIN, "--esm-dir", b.REMOTE_ESM, "--lane", str(lane),
            *b.allocation_args(session, fit_seconds=fit_seconds)]


def launch_script(batch: str, planned: list[dict[str, Any]], session: dict[str, Any], step: c.Step) -> str:
    """pmm_v3_step_c.launch_script, line for line, with this comparison's runner command."""
    directory = c.launch_dir(step)
    q = shlex.quote
    lines = ["set -eu", f"cd {q(c.remote_code(step))}", f"mkdir -p {q(directory)}"]
    for item in planned:  # keep earlier attempts' files (a rerun after archive-failed reuses the name)
        base = f"{directory}/{item['unit']}"
        files = " ".join(q(f"{base}.{suffix}") for suffix in ("out", "rc", "pid", "boot", "launch.json"))
        lines.append(f"for f in {files}; do if [ -e \"$f\" ]; then n=1; while [ -e \"$f.$n\" ]; do n=$((n+1)); done; "
                     "mv \"$f\" \"$f.$n\"; fi; done")
    lines.append("boot=$(cat /proc/sys/kernel/random/boot_id)")
    members = []
    stamp = utc_now()
    for item in planned:
        base = f"{directory}/{item['unit']}"
        command = unit_command(item["unit"], item["lane"], session, fit_seconds=item["fit_seconds"])
        record = {"unit": item["unit"], "lane": item["lane"], "batch": batch, "session_id": session["session_id"],
                  "kind": "unit", "fit_seconds": item["fit_seconds"], "launched_at": stamp, "argv": command}
        lines.append(f"printf '%s\\n' {q(json.dumps(record, sort_keys=True))} > {q(base + '.launch.json')}")
        lines.append(f"printf '%s\\n' \"$boot\" > {q(base + '.boot')}")
        members.append(f"( {shlex.join(command)} > {q(base + '.out')} 2>&1; echo $? > {q(base + '.rc.tmp')}; "
                       f"mv {q(base + '.rc.tmp')} {q(base + '.rc')} ) & echo $! > {q(base + '.pid')}")
    inner = "\n".join(members + ["wait"])
    lines.append(f"setsid nohup bash -c {q(inner)} > {q(directory + '/' + batch + '.launcher.log')} 2>&1 < /dev/null &")
    lines.append("echo launched")
    return "\n".join(lines)


def start(remote: b.Remote, session: dict[str, Any], planned: list[dict[str, Any]], *, wait: bool, step: c.Step,
          sleep=time.sleep) -> dict[str, Any]:
    """pmm_v3_step_c.start with this comparison's launch script."""
    b.admission_check(session, max(item["fit_seconds"] for item in planned))
    batch = f"{step.key}-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    try:
        out = remote.run(launch_script(batch, planned, session, step)).stdout
    except (LauncherError, subprocess.SubprocessError, OSError) as exc:
        raise LauncherError(f"The launch command failed ({exc}). The runners may have started: check with "
                            "'status' and resume with 'wait'; never relaunch") from exc
    require("launched" in out, f"launch did not confirm: {out}; check with 'status' and 'wait' before anything else")
    result: dict[str, Any] = {"batch": batch, "launched": planned}
    if wait:
        result.update(c.cmd_wait(remote, session, names=[item["unit"] for item in planned], sleep=sleep, step=step))
    else:
        result["next"] = "wait (or wait --any to refill a lane as soon as its unit ends)"
    return result


def cmd_launch(remote: b.Remote, session: dict[str, Any], requests: list[tuple[str, int]], *, step: c.Step,
               fit_seconds: float | None = None, wait: bool = True) -> dict[str, Any]:
    planned = c.plan_units(requests, c.vm_state(remote, step=step), fit_seconds=fit_seconds, step=step)
    return start(remote, session, planned, wait=wait, step=step)


def cmd_deploy(remote: b.Remote, campaign: Path = CAMPAIGN, *, root: Path = ROOT) -> dict[str, Any]:
    """Copy the bound VM runner next to the frozen runner (only into the extension code directory, never over a
    different file), then verify its hash on the VM."""
    paths = record_paths(campaign)
    manifest = d.read_record(paths["manifest"])
    require(manifest is not None, "Record the manifest first ('build-manifest')")
    problems = verify_manifest(manifest, root=root)
    require(not problems, "The manifest does not verify: " + "; ".join(problems))
    authorization = d.read_record(paths["authorization"])
    require(authorization is not None and authorization.get("manifest_sha256") == sha256_file(paths["manifest"]),
            "No recorded GPU-start authorization for this manifest; nothing is copied to the VM before it")
    bound = manifest["vm_runner"]["sha256"]
    current = remote_runner_sha256(remote)
    if current == bound:
        return {"remote_path": REMOTE_RUNNER, "sha256": current, "copied": False}
    require(current is None, f"{REMOTE_RUNNER} exists with another SHA-256 ({current}); inspect it, never overwrite")
    remote.run(f"test -f {shlex.quote(d.REMOTE_CODE + '/run_pmm_v3_campaign.py')}")
    remote.transport.put(root / VM_RUNNER, REMOTE_RUNNER)
    copied = remote_runner_sha256(remote)
    require(copied == bound, f"The copied {VM_RUNNER} has SHA-256 {copied}, not the bound {bound}")
    return {"remote_path": REMOTE_RUNNER, "sha256": copied, "copied": True}


def cmd_units() -> dict[str, Any]:
    spec = load_spec()
    return {"step": "S10", "units": spec["units"], "controls": spec["controls"],
            "development_fold": spec["development_fold"], "single_change": spec["single_change"],
            "forecast_seconds_per_fit": FIT_SECONDS}


def cmd_gates(campaign: Path = CAMPAIGN) -> dict[str, Any]:
    paths = record_paths(campaign)
    manifest, ceiling = d.read_record(paths["manifest"]), d.read_record(paths["ceiling"])
    readiness = d.latest_readiness(campaign)
    out = {"readiness_report": None if readiness is None else readiness["path"],
           "readiness_matches_worktree_runner": None if readiness is None
           else readiness.get("runner_sha256") == d.local_runner_sha256(),
           "manifest": None if manifest is None else {"path": str(paths["manifest"]),
                                                      "sha256": sha256_file(paths["manifest"]),
                                                      "problems": verify_manifest(manifest)},
           "ceiling": ceiling, "authorization": d.read_record(paths["authorization"])}
    if ceiling is not None:
        out["spend"] = e2.estimated_spend(ceiling)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("units", "gates", "build-manifest", "record-ceiling", "record-authorization",
                                            "spend", "deploy", "session", "status", "launch", "wait", "pull",
                                            "recover-lane", "archive-failed", "evidence"))
    parser.add_argument("items", nargs="*", help="launch: UNIT@LANE ...; wait: optional UNIT ...; archive-failed: "
                                                 "UNIT; recover-lane: K")
    parser.add_argument("--fit-seconds", type=float, help="launch: admission forecast for every unit")
    parser.add_argument("--no-wait", action="store_true", help="launch: return after the launch")
    parser.add_argument("--any", action="store_true", help="wait: return after the first unit ends")
    parser.add_argument("--lanes", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--allow-running", action="store_true", help="evidence: emergency copy while units run")
    parser.add_argument("--log-entry", help="build-manifest / record-*: the dated campaign log entry")
    parser.add_argument("--usd", type=float, help="record-ceiling: the campaign ceiling (gross, storage included)")
    parser.add_argument("--spent-usd", type=float, help="record-ceiling: estimated charges at --spent-at")
    parser.add_argument("--spent-at", help="record-ceiling: UTC time of that calculation (YYYY-MM-DDTHH:MM:SSZ)")
    parser.add_argument("--spent-source", help="record-ceiling: where that calculation is recorded (log entry)")
    parser.add_argument("--reserve-usd", type=float, default=DEFAULT_RESERVE_USD,
                        help="record-ceiling: kept free after the last launch (step F and its storage days)")
    parser.add_argument("--reserve-basis", default=DEFAULT_RESERVE_BASIS, help="record-ceiling: how the reserve is sized")
    args = parser.parse_args(argv)
    paths = record_paths(CAMPAIGN)
    if args.command in ("units", "gates"):
        print(json.dumps(cmd_units() if args.command == "units" else cmd_gates(), indent=2, sort_keys=True))
        return 0
    if args.command == "build-manifest":
        written = write_json_once(paths["manifest"], build_manifest(log_entry=args.log_entry))
        print(json.dumps({"manifest": str(paths["manifest"]), "sha256": sha256_file(paths["manifest"]),
                          "units": written["units"], "controls": sorted(written["controls"]),
                          "vm_runner_sha256": written["vm_runner"]["sha256"]}, indent=2))
        return 0
    if args.command == "record-ceiling":
        require(None not in (args.usd, args.spent_usd, args.spent_at, args.spent_source),
                "record-ceiling needs --usd, --spent-usd, --spent-at and --spent-source")
        record = ceiling_record(usd=args.usd, spent_usd=args.spent_usd, spent_at=args.spent_at,
                                spent_source=args.spent_source, log_entry=args.log_entry, reserve_usd=args.reserve_usd,
                                reserve_basis=args.reserve_basis)
        written = write_json_once(paths["ceiling"], record)
        print(json.dumps({**written, "sha256": sha256_file(paths["ceiling"])}, indent=2, sort_keys=True))
        return 0
    if args.command == "record-authorization":
        manifest = d.read_record(paths["manifest"])
        require(manifest is not None, "Record the manifest first ('build-manifest')")
        problems = verify_manifest(manifest)
        require(not problems, "The manifest does not verify: " + "; ".join(problems))
        record = authorization_record(manifest_path=paths["manifest"], ceiling_path=paths["ceiling"],
                                      log_entry=args.log_entry)
        print(json.dumps(write_json_once(paths["authorization"], record), indent=2, sort_keys=True))
        return 0
    if args.command == "spend":
        ceiling = d.read_record(paths["ceiling"])
        require(ceiling is not None, "No ceiling record ('record-ceiling')")
        print(json.dumps(e2.estimated_spend(ceiling), indent=2, sort_keys=True))
        return 0
    session = b.read_session()
    if args.command == "session":
        print(json.dumps(session, indent=2, sort_keys=True))
        return 0
    remote = b.Remote(b.check_ssh_config())
    step = build_step(vm_runner=lambda: remote_runner_sha256(remote))
    if args.command == "launch":
        require(bool(args.items), "launch needs UNIT@LANE ...")
    if args.command == "archive-failed":
        require(len(args.items) == 1, "archive-failed needs one unit")
    if args.command == "recover-lane":
        require(len(args.items) == 1 and args.items[0].isdigit(), "recover-lane needs one lane number")
    actions = {
        "deploy": lambda: cmd_deploy(remote),
        "status": lambda: {**c.cmd_status(remote, step), "gates": cmd_gates(),
                           "vm_runner_sha256": remote_runner_sha256(remote)},
        "launch": lambda: cmd_launch(remote, session, [c.parse_request(item) for item in args.items],
                                     fit_seconds=args.fit_seconds, wait=not args.no_wait, step=step),
        "wait": lambda: c.cmd_wait(remote, session, names=args.items or None, any_unit=args.any, step=step),
        "pull": lambda: b.cmd_pull(remote, args.lanes),
        "recover-lane": lambda: c.cmd_recover_lane(remote, session, int(args.items[0]), step),
        "archive-failed": lambda: c.cmd_archive_failed(remote, args.items[0], step),
        "evidence": lambda: c.cmd_evidence(remote, allow_running=args.allow_running, step=step),
    }
    print(json.dumps(actions[args.command](), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (LauncherError, b.host_pull.HostPullError) as exc:
        print(f"10-epoch comparison launcher refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
