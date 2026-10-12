#!/usr/bin/env python3
"""Workstation launcher and freeze records of v3 step F (standard library only; rules: step_f_spec.json, draft v3).

Order (spec "freeze_order"; every step needs the user's logged go-ahead, recorded by the command that names it):
  F1  bind-selection --log-entry v3-NNN     CPU, once: the selection record (E2 selection, the 10-epoch schedule
                                            decision "not adopted", E_sel = 50, terminal checkpoint, seed 42)
  F2  build-manifest / record-ceiling / record-authorization / deploy / launch / wait / pull / evidence
                                            the validation-free refit on the VM (training data only) through the step
                                            C lane, wait, pull, recovery and evidence logic (pmm_v3_step_c) unchanged
      refit-receipt                         offline: the pulled refit, its reload check and its bindings, verified
  Freeze 1  freeze1-template / record-freeze1 --log-entry v3-NNN
                                            before any test-side read: rules, code (equal to the refit manifest's),
                                            checkpoint, seed, epochs, report specification and disclosures, plus the
                                            DATASETS ledger text; an amendment (a fix before Freeze 2) is recorded with
                                            --amends SHA --prior-test-side-read none|yes after the old record is
                                            archived by hand (spec freeze1_amendment)
  F3b run_pmm_v3_step_f.py --action prepare-test / test-embeddings --campaign-dir C (workstation CPU; label-blind)
  Freeze 2  freeze2-template / record-freeze2 --log-entry v3-NNN --prepared-dir D --test-esm-dir E
                                            before any prediction or label read: memberships and inventories by
                                            SHA-256, made under the recorded Freeze 1
  F4  after its own go-ahead: run_pmm_v3_step_f.py --action predict --campaign-dir C --log-entry v3-NNN (one
      campaign-wide claim that records the go-ahead), then pmm_v3_step_f_report.py --campaign-dir C (the only
      component that uses a label value)

The records that name a git commit (F1, the manifest and Freeze 1) are written only when every bound step F file is
committed unchanged at that commit. Offline: units, gates, spend, the templates and the dry runs (--dry-run prints a
record without writing it). Like the
other launchers this never starts, stops, restores or extends a VM: the controller (~/deepmzyme-vm/bin) owns billing.
Every refit launch is refused unless the CPU readiness report and the VM extension record agree with this worktree; the
selection record and the manifest verify; the three VM files are the bound ones; the recorded authorization names this
manifest and ceiling; and the estimated campaign charges with the running session to its hard stop, plus the reserve,
stay within the recorded ceiling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent
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
SPEC_JSON = SPEC_DIR / "step_f_spec.json"
SPEC_REL = "docs/campaigns/pmm_ion_metal_v3/step_f_spec.json"
FROZEN_SPEC_SHA256 = "04d22cc432de6aaece0ea5a140cbaac0799d1194f79eb37f50aa1bec24d9b77c"  # = pmm_v3_test_prep.SPEC_SHA256 (log v3-043); another file is refused
VM_RUNNER = "run_pmm_v3_step_f.py"
VM_FILES = (VM_RUNNER, "pmm_v3_test_prep.py", SPEC_REL)  # deployed next to the frozen runner
STEP_F_TOOL_FILES = ("pmm_v3_step_f.py", VM_RUNNER, "pmm_v3_test_prep.py", "pmm_v3_step_f_report.py")
REUSED_FILES = ("pmm_v3_step_b.py", "pmm_v3_step_c.py", "pmm_v3_step_d.py", "pmm_v3_host_pull.py", "pmm_v3_step_e2.py")
TOOL_FILES = STEP_F_TOOL_FILES + REUSED_FILES
STEP_KEY = "f"
EVIDENCE_DIR = "step_f_evidence"
SELECTION_NAME, MANIFEST_NAME = "f1_selection.json", "f_manifest.json"
CEILING_NAME, AUTHORIZATION_NAME = "f_ceiling.json", "f_authorization.json"
FREEZE1_NAME, FREEZE2_NAME = "freeze1.json", "freeze2.json"
PREDICTION_CLAIM_NAME = "f4_prediction_claim.json"  # written once by run_pmm_v3_step_f.py predict (= prep.RECORD_NAMES)
SUPERSEDED_FREEZE1 = "freeze1.superseded_*.json"  # an amended Freeze 1's predecessor, archived by hand
MEMBERSHIP_FILE_NAMES = ("prepared_rows.csv", "scored_source_uids.txt", "link_flagged_source_uids.txt",
                         "unscorable_rows.csv", "clean_subset_source_uids.txt", "clean_exclusions.csv",
                         "previously_opened_pdb_ids.txt", "training_pdb_ids.txt", "test_esm_plan.csv")  # = prep's
BOUND_FILES = STEP_F_TOOL_FILES + (SPEC_REL,) + REUSED_FILES  # committed before a record names the git commit
LOG_ENTRY = re.compile(r"v3-\d{3}")
DRY_RUN_ENTRY = "v3-dry-run"
EVIDENCE_PATTERNS_F = tuple(f"step_{STEP_KEY}/**/*" if pattern == "step_c/**/*" else pattern
                            for pattern in c.EVIDENCE_PATTERNS) + ("campaign_extension.json",)
REMOTE_FILES = {name: f"{d.REMOTE_CODE}/{name}" for name in VM_FILES}
# Admission forecast of the refit, launch to verified pull: the four 50-epoch fits of the selected cell in step E took
# 1,834-2,183 s (training, verification, replay; three lanes) plus 32-85 s preflight. The refit trains on all 7,398
# ions instead of about 5,920 (x 1.25), skips validation and the independent replay, adds an in-process reload check
# of the full cohort (warm caches) and runs alone; 2,183 x 1.25 + 300 s reload + 90 s preflight + 60 s pull, rounded up.
FIT_SECONDS = 3300.0
# Kept free after the refit launch: storage until the report and the disk's retirement, about five calendar days at
# $0.55 ($2.75), rounded up. The test-side ESMC, preparation, prediction and report run on the workstation CPU ($0).
# The user may record another reserve.
DEFAULT_RESERVE_USD = 3.0
DEFAULT_RESERVE_BASIS = ("storage until the report and the disk's retirement, about 5 calendar days at $0.55/day "
                         "($2.75), rounded up; the test side runs on the workstation CPU")
LEDGER_PLACEHOLDER = "<fill in>"
RELOAD_TOLERANCE = 0.001  # spec refit.reload_check: "within 0.001"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_spec(path: Path | None = None) -> dict[str, Any]:
    path = SPEC_JSON if path is None else path
    require(sha256_file(path) == FROZEN_SPEC_SHA256, f"{path} differs from the pinned step F specification")
    spec = read_json(path)
    selection, refit = spec.get("selection") or {}, spec.get("refit") or {}
    require(spec.get("step") == "F" and spec.get("campaign_id") == "pmm_ion_metal_v3"
            and selection.get("epochs") == refit.get("epochs") == 50 and selection.get("seed") == refit.get("seed") == 42
            and refit.get("checkpoint_rule") == "terminal" and spec.get("held_out_test_accessed") is False
            and selection.get("cell") == f"{selection.get('family')}__{selection.get('target')}__{selection.get('recipe')}",
            "Not the step F specification this launcher was built for")
    return spec


def refit_name() -> str:
    return load_spec()["refit"]["run_name"]


def record_paths(campaign: Path) -> dict[str, Path]:
    evidence = campaign / EVIDENCE_DIR
    return {"selection": evidence / SELECTION_NAME, "manifest": evidence / MANIFEST_NAME,
            "ceiling": evidence / CEILING_NAME, "authorization": evidence / AUTHORIZATION_NAME,
            "freeze1": evidence / FREEZE1_NAME, "freeze2": evidence / FREEZE2_NAME,
            "prediction": evidence / PREDICTION_CLAIM_NAME}


def require_log_entry(log_entry: str | None, what: str, *, dry_run_allowed: bool = False) -> None:
    allowed = LOG_ENTRY.fullmatch(str(log_entry)) is not None or (dry_run_allowed and log_entry == DRY_RUN_ENTRY)
    require(allowed, f"Name the campaign log entry of the user's {what} go-ahead (--log-entry v3-NNN)")


def git_binding(root: Path = ROOT, files: tuple[str, ...] = BOUND_FILES) -> dict[str, Any]:
    """The HEAD commit and whether every bound file is committed there unchanged (read-only git)."""
    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "--no-optional-locks", "-C", str(root), *args], capture_output=True, text=True)

    uncommitted = []
    for name in files:
        head, work = git("rev-parse", "--verify", "--quiet", f"HEAD:{name}"), git("hash-object", "--", name)
        if head.returncode != 0 or work.returncode != 0 or head.stdout.strip() != work.stdout.strip():
            uncommitted.append(name)
    return {"commit": e2._git_commit(root), "bound_files_committed": not uncommitted, "uncommitted": uncommitted}


def placeholders(value: Any) -> bool:
    if isinstance(value, dict):
        return any(placeholders(item) for item in value.values())
    if isinstance(value, list):
        return any(placeholders(item) for item in value)
    return value == LEDGER_PLACEHOLDER


# ---------------------------------------------------------------------------
# F1: the selection record
# ---------------------------------------------------------------------------

def selection_record(campaign: Path = CAMPAIGN, *, log_entry: str, root: Path = ROOT) -> dict[str, Any]:
    """Bind the final E2 selection and the 10-epoch schedule decision; refused unless both verify now."""
    require_log_entry(log_entry, "F1", dry_run_allowed=True)
    spec = load_spec()
    selection = spec["selection"]
    e2_path, s10_path = campaign / selection["e2_assessment"], campaign / selection["schedule_assessment"]
    require(e2_path.is_file() and sha256_file(e2_path) == selection["e2_assessment_sha256"],
            f"{e2_path} is missing or not the E2 assessment the specification names")
    require(s10_path.is_file() and sha256_file(s10_path) == selection["schedule_assessment_sha256"],
            f"{s10_path} is missing or not the schedule assessment the specification names")
    e2_assessment, s10_assessment = read_json(e2_path), read_json(s10_path)
    problems = []
    final = ((e2_assessment.get("result") or {}).get("final_selection") or {}).get("selected")
    if e2_assessment.get("kind") != "E2" or (e2_assessment.get("result") or {}).get("status") != "complete":
        problems.append("the E2 assessment is not complete")
    if final != selection["cell"]:
        problems.append(f"the E2 final selection is {final!r}, not {selection['cell']}")
    decision = (s10_assessment.get("result") or {}).get("schedule_decision") or {}
    if (s10_assessment.get("result") or {}).get("status") != "complete" or decision.get("outcome") != "not_adopted" \
            or decision.get("adopted") is not False or decision.get("epochs") != selection["epochs"] \
            or decision.get("selection") != selection["cell"] or decision.get("checkpoint_rule") != "terminal":
        problems.append("the schedule assessment does not record the 10-epoch schedule as not adopted with E_sel = 50")
    if e2_assessment.get("held_out_test_accessed") is not False or s10_assessment.get("held_out_test_accessed") is not False:
        problems.append("an assessment does not record held_out_test_accessed false")
    require(not problems, "The selection does not verify: " + "; ".join(problems))
    binding = git_binding(root)
    return {"kind": "step_f_selection", "schema_version": 1, "campaign_id": spec["campaign_id"], "step": "F1",
            "log_entry": log_entry, "created_at": utc_now(), "git_commit": binding["commit"],
            "bound_files_committed": binding["bound_files_committed"],
            "spec": {"path": SPEC_REL, "sha256": sha256_file(SPEC_JSON)},
            "selection": {key: selection[key] for key in ("cell", "family", "target", "recipe", "epochs", "seed")},
            "e2_assessment": {"path": str(e2_path), "sha256": selection["e2_assessment_sha256"], "final_selection": final},
            "schedule_assessment": {"path": str(s10_path), "sha256": selection["schedule_assessment_sha256"],
                                    "outcome": decision["outcome"], "meaning": decision.get("meaning")},
            "refit": {"run_name": spec["refit"]["run_name"], "epochs": spec["refit"]["epochs"],
                      "checkpoint_rule": spec["refit"]["checkpoint_rule"], "seed": spec["refit"]["seed"],
                      "calibration": None, "ensemble": False},
            "not_adopted": "the 10-epoch schedule (frozen gates; S10 assessment)",
            "deferred_untested": "the other optimization hypotheses stay deferred and untested, not disproven",
            "held_out_access": False}


# ---------------------------------------------------------------------------
# F2: manifest, ceiling, authorization and launch gates
# ---------------------------------------------------------------------------

def build_manifest(campaign: Path = CAMPAIGN, *, log_entry: str, root: Path = ROOT) -> dict[str, Any]:
    require_log_entry(log_entry, "F2 manifest", dry_run_allowed=True)
    spec = load_spec()
    paths = record_paths(campaign)
    selection = d.read_record(paths["selection"])
    require(selection is not None, "Record the F1 selection first ('bind-selection')")
    require(selection["spec"]["sha256"] == sha256_file(SPEC_JSON), "The selection record binds another specification")
    extension_path = campaign / e2.EXTENSION_COPY
    extension = read_json(extension_path)
    runner = d.local_runner_sha256(root)
    require(extension.get("extension_id") == d.EXTENSION_ID and runner == extension["runner_sha256"],
            "This worktree's runner files differ from the extension record's")
    binding = git_binding(root)
    return {"schema_version": 1, "step": "F2", "campaign_id": spec["campaign_id"], "created_at": utc_now(),
            "log_entry": log_entry, "git_commit": binding["commit"],
            "bound_files_committed": binding["bound_files_committed"],
            "spec": {"path": SPEC_REL, "sha256": sha256_file(SPEC_JSON)},
            "selection_record": {"path": str(paths["selection"]), "sha256": sha256_file(paths["selection"])},
            "extension": {"id": extension["extension_id"], "sha256": sha256_file(extension_path),
                          "copy": str(extension_path)},
            "runner_sha256": runner, "tooling_sha256": {name: sha256_file(root / name) for name in TOOL_FILES},
            "vm_files": {name: {"remote_path": REMOTE_FILES[name], "sha256": sha256_file(root / name)}
                         for name in VM_FILES},
            "units": [spec["refit"]["run_name"]], "epochs": spec["refit"]["epochs"],
            "remote_code": d.REMOTE_CODE, "launch_folder": f"step_{STEP_KEY}/launch", "held_out_access": False}


def verify_manifest(manifest: dict[str, Any], *, campaign: Path = CAMPAIGN, root: Path = ROOT,
                    vm_extension: dict[str, Any] | None = None,
                    vm_files: dict[str, str | None] | None = None) -> list[str]:
    problems = []
    try:
        spec = load_spec()
    except (LauncherError, OSError, ValueError) as exc:
        return [f"specification: {exc}"]
    if (manifest.get("spec") or {}).get("sha256") != sha256_file(SPEC_JSON):
        problems.append("the specification differs from the manifest")
    bound = manifest.get("selection_record") or {}
    path = Path(bound.get("path", ""))
    if not path.is_file() or sha256_file(path) != bound.get("sha256"):
        problems.append("the F1 selection record is missing or differs from the manifest")
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
        problems.append("the step F tooling or a reused launcher module changed since the manifest was recorded "
                        "(record a new manifest and a new authorization)")
    for name in VM_FILES:
        expected = ((manifest.get("vm_files") or {}).get(name) or {}).get("sha256")
        if expected != sha256_file(root / name):
            problems.append(f"the bound {name} differs from this worktree's")
        if vm_files is not None and vm_files.get(name) != expected:
            problems.append(f"the VM copy of {name} is missing or not the bound one ('deploy')")
    if manifest.get("units") != [spec["refit"]["run_name"]]:
        problems.append("the manifest's unit is not the specified refit")
    return problems


def ceiling_record(*, usd: float, spent_usd: float, spent_at: str, spent_source: str, log_entry: str,
                   reserve_usd: float = DEFAULT_RESERVE_USD, reserve_basis: str = DEFAULT_RESERVE_BASIS) -> dict[str, Any]:
    record = e2.ceiling_record(usd=usd, spent_usd=spent_usd, spent_at=spent_at, spent_source=spent_source,
                               log_entry=log_entry, reserve_usd=reserve_usd, reserve_basis=reserve_basis)
    record["scope"] = "campaign total, gross, storage included; step F refit launches are refused above it"
    record["grants"] = "nothing: a GPU start needs the user's recorded step F refit authorization"
    return record


def authorization_record(*, manifest_path: Path, ceiling_path: Path, log_entry: str) -> dict[str, Any]:
    record = e2.authorization_record(manifest_path=manifest_path, ceiling_path=ceiling_path, log_entry=log_entry)
    record["decision"] = "the user's GPU-start authorization for the step F refit (training data only)"
    record["covers"] = "F2 only: no test-side read, test embedding, prediction or report"
    return record


def remote_file_sha256(remote: b.Remote) -> dict[str, str | None]:
    out = {}
    for name, path in REMOTE_FILES.items():
        text = remote.run(f"if [ -f {shlex.quote(path)} ]; then sha256sum {shlex.quote(path)}; fi").stdout
        out[name] = text.split()[0] if text.strip() else None
    return out


def f_blockers(names: list[str], state: dict[str, Any], *, campaign: Path, readiness: dict[str, Any] | None,
               runner: dict[str, str], vm_files: Callable[[], dict[str, str | None]], root: Path = ROOT,
               spend: Callable[[dict[str, Any]], dict[str, Any]] = e2.estimated_spend) -> list[str]:
    problems = d.readiness_blockers(state, readiness, runner)
    paths = record_paths(campaign)
    manifest = d.read_record(paths["manifest"])
    if manifest is None:
        problems.append("no manifest ('build-manifest'); the refit does not start without it")
    else:
        problems += [f"manifest: {p}" for p in verify_manifest(manifest, campaign=campaign, root=root,
                                                               vm_extension=state.get("extension"),
                                                               vm_files=vm_files())]
    ceiling = d.read_record(paths["ceiling"])
    if ceiling is None:
        problems.append("no ceiling record ('record-ceiling'); the ceiling must be in the execution controls")
    authorization = d.read_record(paths["authorization"])
    if authorization is None:
        problems.append("no recorded GPU-start authorization for the step F refit ('record-authorization' after the "
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
                            f"recorded {estimate['ceiling_usd']:.2f} USD ceiling")
    return problems


def build_step(campaign: Path = CAMPAIGN, *, vm_files: Callable[[], dict[str, str | None]],
               runner: dict[str, str] | None = None, root: Path = ROOT,
               spend: Callable[[dict[str, Any]], dict[str, Any]] = e2.estimated_spend) -> c.Step:
    units = (refit_name(),)
    readiness = d.latest_readiness(campaign)
    runner = d.local_runner_sha256(root) if runner is None else runner
    return c.Step(
        key=STEP_KEY, label="step F refit",
        outside="step F admits only its refit ('units'); the test-side actions never run through the lanes",
        second_failure="step F stops for diagnosis and the user's decision (a second refit failure)",
        done_hint="the refit has run: 'pull', 'evidence', vm-stop, then 'refit-receipt' on the workstation",
        units=lambda state: units,
        batch_blockers=lambda names, state: f_blockers(names, state, campaign=campaign, readiness=readiness,
                                                       runner=runner, vm_files=vm_files, root=root, spend=spend),
        forecast=lambda name, state: FIT_SECONDS,
        code=d.REMOTE_CODE, evidence=campaign / EVIDENCE_DIR, evidence_patterns=EVIDENCE_PATTERNS_F)


def unit_command(name: str, lane: int, session: dict[str, Any], *, fit_seconds: float,
                 selection_sha256: str) -> list[str]:
    return [b.REMOTE_PY, "-u", VM_RUNNER, "--action", "refit", "--campaign-dir", b.REMOTE_V3, "--unit", name,
            "--selection-sha256", selection_sha256, "--epochs", str(load_spec()["refit"]["epochs"]),
            "--train-dir", b.REMOTE_TRAIN, "--esm-dir", b.REMOTE_ESM, "--lane", str(lane),
            *b.allocation_args(session, fit_seconds=fit_seconds)]


def launch_script(batch: str, planned: list[dict[str, Any]], session: dict[str, Any], step: c.Step, *,
                  selection_sha256: str) -> str:
    """pmm_v3_step_c.launch_script, line for line, with the step F runner command."""
    directory = c.launch_dir(step)
    q = shlex.quote
    lines = ["set -eu", f"cd {q(c.remote_code(step))}", f"mkdir -p {q(directory)}"]
    for item in planned:
        base = f"{directory}/{item['unit']}"
        files = " ".join(q(f"{base}.{suffix}") for suffix in ("out", "rc", "pid", "boot", "launch.json"))
        lines.append(f"for f in {files}; do if [ -e \"$f\" ]; then n=1; while [ -e \"$f.$n\" ]; do n=$((n+1)); done; "
                     "mv \"$f\" \"$f.$n\"; fi; done")
    lines.append("boot=$(cat /proc/sys/kernel/random/boot_id)")
    members = []
    stamp = utc_now()
    for item in planned:
        base = f"{directory}/{item['unit']}"
        command = unit_command(item["unit"], item["lane"], session, fit_seconds=item["fit_seconds"],
                               selection_sha256=selection_sha256)
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
          selection_sha256: str, sleep=time.sleep) -> dict[str, Any]:
    b.admission_check(session, max(item["fit_seconds"] for item in planned))
    batch = f"{step.key}-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    try:
        out = remote.run(launch_script(batch, planned, session, step, selection_sha256=selection_sha256)).stdout
    except (LauncherError, subprocess.SubprocessError, OSError) as exc:
        raise LauncherError(f"The launch command failed ({exc}). The refit may have started: check with 'status' and "
                            "resume with 'wait'; never relaunch") from exc
    require("launched" in out, f"launch did not confirm: {out}; check with 'status' and 'wait' before anything else")
    result: dict[str, Any] = {"batch": batch, "launched": planned}
    if wait:
        result.update(c.cmd_wait(remote, session, names=[item["unit"] for item in planned], sleep=sleep, step=step))
    else:
        result["next"] = "wait"
    return result


def cmd_launch(remote: b.Remote, session: dict[str, Any], requests: list[tuple[str, int]], *, step: c.Step,
               campaign: Path = CAMPAIGN, fit_seconds: float | None = None, wait: bool = True) -> dict[str, Any]:
    planned = c.plan_units(requests, c.vm_state(remote, step=step), fit_seconds=fit_seconds, step=step)
    selection = record_paths(campaign)["selection"]
    return start(remote, session, planned, wait=wait, step=step, selection_sha256=sha256_file(selection))


def cmd_deploy(remote: b.Remote, campaign: Path = CAMPAIGN, *, root: Path = ROOT) -> dict[str, Any]:
    """Copy the three bound files into the extension code directory (never over a different file), then verify."""
    paths = record_paths(campaign)
    manifest = d.read_record(paths["manifest"])
    require(manifest is not None, "Record the manifest first ('build-manifest')")
    problems = verify_manifest(manifest, campaign=campaign, root=root)
    require(not problems, "The manifest does not verify: " + "; ".join(problems))
    authorization = d.read_record(paths["authorization"])
    require(authorization is not None and authorization.get("manifest_sha256") == sha256_file(paths["manifest"]),
            "No recorded GPU-start authorization for this manifest; nothing is copied to the VM before it")
    remote.run(f"test -f {shlex.quote(d.REMOTE_CODE + '/run_pmm_v3_campaign.py')}")
    current = remote_file_sha256(remote)
    copied = []
    for name in VM_FILES:
        bound = manifest["vm_files"][name]["sha256"]
        if current[name] == bound:
            continue
        require(current[name] is None, f"{REMOTE_FILES[name]} exists with another SHA-256 ({current[name]}); "
                                       "inspect it, never overwrite")
        remote.run(f"mkdir -p {shlex.quote(str(Path(REMOTE_FILES[name]).parent))}")
        remote.transport.put(root / name, REMOTE_FILES[name])
        copied.append(name)
    after = remote_file_sha256(remote)
    require(all(after[name] == manifest["vm_files"][name]["sha256"] for name in VM_FILES),
            f"The VM copies differ from the bound files: {after}")
    return {"remote_files": after, "copied": copied}


# ---------------------------------------------------------------------------
# The pulled refit (workstation, offline)
# ---------------------------------------------------------------------------

def refit_receipt(campaign: Path = CAMPAIGN) -> dict[str, Any]:
    """The pulled refit: completed, reload-checked, checkpoint intact, and bound to this selection, specification and
    VM runner. Raises unless every binding holds."""
    spec = load_spec()
    name = spec["refit"]["run_name"]
    paths = record_paths(campaign)
    found = sorted((campaign / "durable").glob(f"lane*/run_status_{name}.json"))
    require(len(found) == 1, f"Expected one pulled status record of {name}, found {len(found)}")
    status = read_json(found[0])
    lane_dir = found[0].parent
    run_dir = lane_dir / "runs" / name
    receipt = status.get("refit_receipt") or {}
    check = receipt.get("reload_check") or {}
    require(status.get("status") == "completed" and check.get("passed") is True
            and check.get("abs_difference", 1.0) <= RELOAD_TOLERANCE, "The refit did not complete its reload check")
    require(sha256_file(run_dir / receipt["checkpoint"]) == receipt["checkpoint_sha256"]
            == status["selected_checkpoint_sha256"], "The pulled checkpoint differs from its receipt")
    for key in ("run_config", "run_metadata"):
        require(sha256_file(run_dir / f"{key}.json") == receipt[f"{key}_sha256"], f"{key}.json changed after the refit")
    identity = status["identity"]
    selection = paths["selection"]
    manifest = d.read_record(paths["manifest"]) or {}
    require(selection.is_file() and identity.get("selection_record_sha256") == sha256_file(selection),
            "The refit identity does not bind the recorded F1 selection")
    require(identity.get("step_f_spec_sha256") == sha256_file(SPEC_JSON)
            and identity.get("step_f_runner_sha256") == ((manifest.get("vm_files") or {}).get(VM_RUNNER) or {}).get("sha256"),
            "The refit ran under another specification or VM runner than the manifest binds")
    require(identity.get("epochs") == spec["refit"]["epochs"] and identity.get("model_seed") == spec["refit"]["seed"]
            and identity.get("checkpoint_rule") == "terminal" and identity.get("validation") == "none"
            and receipt.get("selected_epoch") == spec["refit"]["epochs"],
            "The refit is not the 50-epoch terminal validation-free fit of seed 42")
    return {"run_name": name, "lane": lane_dir.name, "run_dir": str(run_dir), "status_path": str(found[0]),
            "status_sha256": sha256_file(found[0]), "checkpoint": receipt["checkpoint"],
            "checkpoint_sha256": receipt["checkpoint_sha256"], "run_config_sha256": receipt["run_config_sha256"],
            "run_metadata_sha256": receipt["run_metadata_sha256"], "reload_check": check, "identity": identity}


# ---------------------------------------------------------------------------
# Freeze 1 and Freeze 2 records and their DATASETS ledger text
# ---------------------------------------------------------------------------

def freeze1_record(campaign: Path = CAMPAIGN, *, log_entry: str | None, root: Path = ROOT, template: bool = False,
                   amends: str | None = None, prior_test_side_read: str | None = None) -> dict[str, Any]:
    """Freeze 1, before any test-side read. ``template`` fills what is known now and marks the rest. The bound code
    is the refit manifest's; only an amendment (``amends``: the SHA-256 of the superseded record, archived by hand as
    freeze1.superseded_<UTC>.json before Freeze 2 and the prediction claim) may bind other code, and it discloses
    which tools differ and whether a test-side read happened under the superseded record."""
    spec = load_spec()
    paths = record_paths(campaign)
    try:
        refit = refit_receipt(campaign)
    except (LauncherError, OSError, KeyError, ValueError) as exc:
        require(template, f"Freeze 1 needs the verified refit: {exc}")
        refit = {key: LEDGER_PLACEHOLDER for key in ("checkpoint_sha256", "run_config_sha256", "run_metadata_sha256",
                                                      "status_sha256")}
        refit["reload_check"] = LEDGER_PLACEHOLDER
        refit["identity"] = {"cohort_sha256": LEDGER_PLACEHOLDER, "step_f_runner_sha256": LEDGER_PLACEHOLDER}
    selection, manifest_path = paths["selection"], paths["manifest"]
    manifest = d.read_record(manifest_path) or {}
    tools = {name: sha256_file(root / name) for name in STEP_F_TOOL_FILES}
    differs = sorted(name for name in STEP_F_TOOL_FILES
                     if (manifest.get("tooling_sha256") or {}).get(name) != tools[name]
                     or (name == VM_RUNNER and refit["identity"].get("step_f_runner_sha256") != tools[name]))
    tests = spec["test_inputs"]
    binding = git_binding(root)
    record = {
        "kind": "step_f_freeze_1", "schema_version": 2, "campaign_id": spec["campaign_id"],
        "user_go_ahead_log_entry": log_entry or (LEDGER_PLACEHOLDER if template else None),
        "created_at": utc_now(), "git_commit": binding["commit"],
        "bound_files_committed": binding["bound_files_committed"],
        "step_f_spec_sha256": sha256_file(SPEC_JSON),
        "selection_record_sha256": sha256_file(selection) if selection.is_file() else LEDGER_PLACEHOLDER,
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else LEDGER_PLACEHOLDER,
        "selection": {key: spec["selection"][key] for key in ("cell", "epochs", "seed")},
        "refit": {"run_name": spec["refit"]["run_name"], "checkpoint_sha256": refit["checkpoint_sha256"],
                  "run_config_sha256": refit["run_config_sha256"], "run_metadata_sha256": refit["run_metadata_sha256"],
                  "status_sha256": refit["status_sha256"], "reload_check": refit["reload_check"],
                  "runner_sha256": refit["identity"].get("step_f_runner_sha256"),
                  "epochs": spec["refit"]["epochs"], "seed": spec["refit"]["seed"], "checkpoint_rule": "terminal",
                  "calibration": None, "ensemble": False},
        "checkpoint_sha256": refit["checkpoint_sha256"], "cohort_sha256": refit["identity"]["cohort_sha256"],
        "tool_sha256": tools, "tools_differ_from_refit_manifest": differs,
        "runner_sha256": d.local_runner_sha256(root),
        "test_source_csv_sha256": tests["source_csv_sha256"], "test_source_rows": spec["test_rows"]["source_rows"],
        "previously_opened_files": tests["previously_opened_structure_files"],
        "label": "both_results_secondary", "report": spec["report"], "clean_subset": spec["clean_subset"],
        "test_esm": spec["test_esm"], "disclosures": spec["disclosures"], "freeze_order": spec["freeze_order"],
        "amends": None, "test_side_read_before_this_record": False,
        "next": "F3b on the workstation CPU: run_pmm_v3_step_f.py --action prepare-test, then --action test-embeddings "
                "(label-blind)",
    }
    archived = sorted(paths["freeze1"].parent.glob(SUPERSEDED_FREEZE1))
    if amends is not None:
        match = [path for path in archived if sha256_file(path) == amends]
        require(bool(match), f"No archived Freeze 1 ({SUPERSEDED_FREEZE1}) has SHA-256 {amends}")
        require(prior_test_side_read in ("none", "yes"), "An amendment states --prior-test-side-read none or yes")
        record["amends"] = {"superseded_sha256": amends, "archived_path": str(match[0]),
                            "prior_test_side_read": prior_test_side_read}
        record["test_side_read_before_this_record"] = prior_test_side_read == "yes"
    if not template:
        require_log_entry(log_entry, "Freeze 1")
        require(not paths["freeze2"].exists() and not paths["prediction"].exists(),
                "Freeze 2 or the prediction claim exists; Freeze 1 precedes them and is not amended after them")
        require(amends is not None or not archived,
                f"An archived Freeze 1 exists ({archived[0].name if archived else ''}): record this one with --amends")
        require(amends is not None or not differs,
                f"The step F code differs from the refit manifest's ({differs}): Freeze 1 binds the refit's code, and "
                "only an amendment may bind other code (spec freeze1_amendment)")
    return record


def prepared_hashes(prepared_dir: Path) -> dict[str, str]:
    summary = read_json(Path(prepared_dir) / "preparation_summary.json")
    files = dict(summary["files_sha256"])
    for name, digest in files.items():
        require(sha256_file(Path(prepared_dir) / name) == digest, f"{name} differs from its preparation summary")
    files["preparation_summary.json"] = sha256_file(Path(prepared_dir) / "preparation_summary.json")
    return dict(sorted(files.items()))


def freeze2_record(campaign: Path = CAMPAIGN, *, log_entry: str | None, prepared_dir: Path | None,
                   test_esm_dir: Path | None, root: Path = ROOT, template: bool = False) -> dict[str, Any]:
    """Freeze 2, before any prediction or label read: the label-blind memberships and inventories by SHA-256."""
    spec = load_spec()
    paths = record_paths(campaign)
    freeze1 = d.read_record(paths["freeze1"])
    freeze1_sha = sha256_file(paths["freeze1"]) if paths["freeze1"].is_file() else LEDGER_PLACEHOLDER
    if not template:
        require(freeze1 is not None, "Freeze 1 is not recorded")
        require(freeze1.get("kind") == "step_f_freeze_1" and not placeholders(freeze1)
                and LOG_ENTRY.fullmatch(str(freeze1.get("user_go_ahead_log_entry"))) is not None,
                "The recorded Freeze 1 is not a complete record with the user's logged go-ahead")
        require_log_entry(log_entry, "Freeze 2")
        require(prepared_dir is not None and test_esm_dir is not None, "Freeze 2 needs --prepared-dir and --test-esm-dir")
        require(not paths["prediction"].exists(), "The prediction claim exists; Freeze 2 precedes it")
    frozen_prep = ((freeze1 or {}).get("tool_sha256") or {}).get("pmm_v3_test_prep.py")
    summary, hashes, inventory = {}, LEDGER_PLACEHOLDER, LEDGER_PLACEHOLDER
    if prepared_dir is not None and (Path(prepared_dir) / "preparation_summary.json").is_file():
        summary = read_json(Path(prepared_dir) / "preparation_summary.json")
        require(summary.get("labels_read") is False and summary.get("spec_sha256") == sha256_file(SPEC_JSON)
                and summary.get("source_csv_sha256") == spec["test_inputs"]["source_csv_sha256"]
                and summary.get("tool_sha256") == sha256_file(root / "pmm_v3_test_prep.py") == frozen_prep,
                "The preparation was not made label-blind under this specification, source file and Freeze 1's tool")
        require(summary.get("source_rows") == spec["test_rows"]["source_rows"] and summary.get("row_subset") is None,
                "The preparation does not cover every source row")
        require(summary.get("freeze1_sha256") == freeze1_sha, "The preparation was not made under the recorded Freeze 1")
        hashes = prepared_hashes(prepared_dir)
        require(set(hashes) == set(MEMBERSHIP_FILE_NAMES) | {"preparation_summary.json"},
                "The preparation does not hold exactly the specified membership files")
    elif not template:
        raise LauncherError(f"{prepared_dir}: no preparation summary")
    if test_esm_dir is not None and (Path(test_esm_dir) / "test_esm_receipt.json").is_file():
        receipt = read_json(Path(test_esm_dir) / "test_esm_receipt.json")
        for item in receipt["inventory"]:
            payload = Path(test_esm_dir) / item["file"]
            require(sha256_file(payload) == item["sha256"]
                    and sha256_file(payload.with_name(payload.name + ".json")) == item["sidecar_sha256"],
                    f"{item['file']} changed")
        recomputed = hashlib.sha256(json.dumps(receipt["inventory"], sort_keys=True).encode()).hexdigest()
        require(recomputed == receipt["inventory_sha256"], "The test embedding inventory hash does not match its list")
        require(receipt.get("freeze1_sha256") == freeze1_sha and receipt.get("tool_sha256") == frozen_prep,
                "The test embeddings were not made under the recorded Freeze 1 and its tool")
        test_esm = spec["test_esm"]
        require(receipt.get("device") == test_esm["device"] and receipt.get("model_dtype") == test_esm["model_dtype"]
                and receipt.get("storage_dtypes") == ["float32"],
                "The test embeddings were not made on the CPU route in the training model dtype (spec test_esm)")
        inventory = receipt["inventory_sha256"]
    elif not template:
        raise LauncherError(f"{test_esm_dir}: no test embedding receipt")
    record = {
        "kind": "step_f_freeze_2", "schema_version": 1, "campaign_id": spec["campaign_id"],
        "user_go_ahead_log_entry": log_entry or (LEDGER_PLACEHOLDER if template else None), "created_at": utc_now(),
        "freeze1_sha256": freeze1_sha,
        "checkpoint_sha256": (freeze1 or {}).get("checkpoint_sha256", LEDGER_PLACEHOLDER),
        "step_f_spec_sha256": sha256_file(SPEC_JSON),
        "prepared_files_sha256": hashes, "test_esm_inventory_sha256": inventory,
        "test_source_csv_sha256": spec["test_inputs"]["source_csv_sha256"],
        "test_source_rows": spec["test_rows"]["source_rows"],
        "preparation": {key: summary.get(key, LEDGER_PLACEHOLDER) for key in
                        ("dispositions", "unscorable_reasons", "physical_ions", "clean_subset", "esm_plan_chains",
                         "previously_opened_pdb_ids", "training_pdb_ids")},
        "labels_read": False, "predictions_made": False,
        "next": "F4 after the user's go-ahead: run_pmm_v3_step_f.py --action predict --campaign-dir C --log-entry "
                "v3-NNN (one campaign-wide claim), then pmm_v3_step_f_report.py --campaign-dir C",
    }
    if not template:
        require(not paths["freeze2"].exists(), "Freeze 2 is recorded once")
    return record


def ledger_text(record: dict[str, Any]) -> str:
    """The DATASETS test-use ledger paragraph for a freeze record (exact Zenodo PinMyMetal test)."""
    day = str(record.get("created_at", LEDGER_PLACEHOLDER))[:10]
    if record["kind"] == "step_f_freeze_1":
        tools = ", ".join(f"`{name}` `{digest[:12]}`" for name, digest in record["tool_sha256"].items())
        amends = record.get("amends")
        amended = "" if not amends else (
            f"Amended Freeze 1, superseding `{amends['superseded_sha256'][:12]}` (test-side read under it: "
            f"{amends['prior_test_side_read']}; tools differing from the refit manifest: "
            f"{', '.join(record.get('tools_differ_from_refit_manifest') or []) or 'none'}). ")
        return (f"**{day} v3 step F Freeze 1 (exact Zenodo PinMyMetal test, before any test-side read):** {amended}"
                f"specification `step_f_spec.json` `{record['step_f_spec_sha256'][:12]}`; selected model "
                f"`{record['selection']['cell']}` ({record['selection']['epochs']} epochs, terminal checkpoint, seed "
                f"{record['selection']['seed']}; refit checkpoint `{str(record['checkpoint_sha256'])[:12]}`); tools "
                f"{tools}; report specification, clean-subset rules and disclosures as in the specification (label "
                f"{record['label']}). User go-ahead: log {record['user_go_ahead_log_entry']}. Next: label-blind "
                "preparation and test-side ESMC generation on the workstation CPU (no label value used).")
    prep_info = record["preparation"]
    dispositions = prep_info.get("dispositions")
    clean = prep_info.get("clean_subset")
    clean_rows = clean.get("clean_subset_rows") if isinstance(clean, dict) else LEDGER_PLACEHOLDER
    return (f"**{day} v3 step F Freeze 2 (exact Zenodo PinMyMetal test, before any prediction or label read):** "
            f"label-blind preparation of {record['test_source_rows']} source rows: dispositions {dispositions}; "
            f"unscorable reasons {prep_info.get('unscorable_reasons')}; clean subset {clean_rows} rows "
            f"({(clean or {}).get('exclusion_counts') if isinstance(clean, dict) else LEDGER_PLACEHOLDER}); "
            f"memberships and the test ESMC inventory frozen by SHA-256 (inventory "
            f"`{str(record['test_esm_inventory_sha256'])[:12]}`; Freeze 1 `{str(record['freeze1_sha256'])[:12]}`). "
            f"No label value was used (the preparation parses and discards label-bearing cells unused). User go-ahead: "
            f"log {record['user_go_ahead_log_entry']}. Next, after its own go-ahead: the single prediction pass (one "
            "campaign-wide claim), "
            "then the report (the only component that uses a label value).")


# ---------------------------------------------------------------------------
# Offline views
# ---------------------------------------------------------------------------

def cmd_units() -> dict[str, Any]:
    spec = load_spec()
    return {"step": "F2", "units": [spec["refit"]["run_name"]], "lane": 0, "epochs": spec["refit"]["epochs"],
            "forecast_seconds": FIT_SECONDS, "selection": spec["selection"]["cell"],
            "note": "one validation-free refit on the full non-test cohort; the test side never runs through lanes"}


def cmd_gates(campaign: Path = CAMPAIGN) -> dict[str, Any]:
    paths = record_paths(campaign)
    readiness = d.latest_readiness(campaign)
    manifest, ceiling = d.read_record(paths["manifest"]), d.read_record(paths["ceiling"])
    out = {"spec_sha256": sha256_file(SPEC_JSON),
           "readiness_report": None if readiness is None else readiness["path"],
           "readiness_matches_worktree_runner": None if readiness is None
           else readiness.get("runner_sha256") == d.local_runner_sha256(),
           "records": {key: (sha256_file(path) if path.is_file() else None) for key, path in paths.items()},
           "manifest_problems": None if manifest is None else verify_manifest(manifest, campaign=campaign),
           "ceiling": ceiling, "authorization": d.read_record(paths["authorization"])}
    if ceiling is not None:
        out["spend"] = e2.estimated_spend(ceiling)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("units", "gates", "bind-selection", "build-manifest", "record-ceiling",
                                            "record-authorization", "spend", "deploy", "session", "status", "launch",
                                            "wait", "pull", "recover-lane", "archive-failed", "evidence",
                                            "refit-receipt", "freeze1-template", "record-freeze1", "freeze2-template",
                                            "record-freeze2"))
    parser.add_argument("items", nargs="*", help="launch: UNIT@LANE; wait: optional UNIT; archive-failed: UNIT; "
                                                 "recover-lane: K")
    parser.add_argument("--dry-run", action="store_true", help="bind-selection / build-manifest: print, never write")
    parser.add_argument("--fit-seconds", type=float)
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--any", action="store_true")
    parser.add_argument("--lanes", type=int, nargs="+", default=[0])
    parser.add_argument("--allow-running", action="store_true")
    parser.add_argument("--log-entry")
    parser.add_argument("--usd", type=float)
    parser.add_argument("--spent-usd", type=float)
    parser.add_argument("--spent-at")
    parser.add_argument("--spent-source")
    parser.add_argument("--reserve-usd", type=float, default=DEFAULT_RESERVE_USD)
    parser.add_argument("--reserve-basis", default=DEFAULT_RESERVE_BASIS)
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--test-esm-dir", type=Path)
    parser.add_argument("--amends", help="record-freeze1: SHA-256 of the superseded Freeze 1 (archived by hand)")
    parser.add_argument("--prior-test-side-read", choices=("none", "yes"),
                        help="record-freeze1 --amends: whether a test-side read happened under the superseded record")
    args = parser.parse_args(argv)
    paths = record_paths(CAMPAIGN)
    show = lambda payload: print(json.dumps(payload, indent=2, sort_keys=True, default=str))  # noqa: E731

    def require_committed() -> None:  # the record names the git commit, so it must hold the bound code
        binding = git_binding()
        require(binding["bound_files_committed"], "Commit the bound step F files first (the record names the git "
                                                  f"commit): {', '.join(binding['uncommitted'])}")
        require(args.log_entry != DRY_RUN_ENTRY, f"{DRY_RUN_ENTRY} is for printed dry runs only")
    if args.command in ("units", "gates", "spend", "refit-receipt"):
        if args.command == "spend":
            ceiling = d.read_record(paths["ceiling"])
            require(ceiling is not None, "No ceiling record ('record-ceiling')")
            show(e2.estimated_spend(ceiling))
        else:
            show({"units": cmd_units, "gates": cmd_gates, "refit-receipt": refit_receipt}[args.command]())
        return 0
    if args.command in ("bind-selection", "build-manifest"):
        if not args.dry_run:
            require_committed()
        build = selection_record if args.command == "bind-selection" else build_manifest
        record = build(log_entry=args.log_entry or (DRY_RUN_ENTRY if args.dry_run else ""))
        target = paths["selection"] if args.command == "bind-selection" else paths["manifest"]
        show(record if args.dry_run else write_json_once(target, record))
        return 0
    if args.command == "record-ceiling":
        require(None not in (args.usd, args.spent_usd, args.spent_at, args.spent_source),
                "record-ceiling needs --usd, --spent-usd, --spent-at and --spent-source")
        record = ceiling_record(usd=args.usd, spent_usd=args.spent_usd, spent_at=args.spent_at,
                                spent_source=args.spent_source, log_entry=args.log_entry,
                                reserve_usd=args.reserve_usd, reserve_basis=args.reserve_basis)
        show(write_json_once(paths["ceiling"], record))
        return 0
    if args.command == "record-authorization":
        manifest = d.read_record(paths["manifest"])
        require(manifest is not None, "Record the manifest first ('build-manifest')")
        problems = verify_manifest(manifest)
        require(not problems, "The manifest does not verify: " + "; ".join(problems))
        show(write_json_once(paths["authorization"], authorization_record(
            manifest_path=paths["manifest"], ceiling_path=paths["ceiling"], log_entry=args.log_entry)))
        return 0
    if args.command in ("freeze1-template", "record-freeze1"):
        template = args.command == "freeze1-template"
        if not template:
            require_committed()
        record = freeze1_record(log_entry=args.log_entry, template=template, amends=args.amends,
                                prior_test_side_read=args.prior_test_side_read)
        show({"record": record, "ledger_text": ledger_text(record)} if template
             else {"record": write_json_once(paths["freeze1"], record), "ledger_text": ledger_text(record)})
        return 0
    if args.command in ("freeze2-template", "record-freeze2"):
        template = args.command == "freeze2-template"
        record = freeze2_record(log_entry=args.log_entry, prepared_dir=args.prepared_dir,
                                test_esm_dir=args.test_esm_dir, template=template)
        show({"record": record, "ledger_text": ledger_text(record)} if template
             else {"record": write_json_once(paths["freeze2"], record), "ledger_text": ledger_text(record)})
        return 0
    session = b.read_session()
    if args.command == "session":
        show(session)
        return 0
    remote = b.Remote(b.check_ssh_config())
    step = build_step(vm_files=lambda: remote_file_sha256(remote))
    if args.command == "launch":
        require(bool(args.items), "launch needs UNIT@LANE")
    if args.command == "archive-failed":
        require(len(args.items) == 1, "archive-failed needs one unit")
    if args.command == "recover-lane":
        require(len(args.items) == 1 and args.items[0].isdigit(), "recover-lane needs one lane number")
    actions = {
        "deploy": lambda: cmd_deploy(remote),
        "status": lambda: {**c.cmd_status(remote, step), "gates": cmd_gates(), "vm_files": remote_file_sha256(remote)},
        "launch": lambda: cmd_launch(remote, session, [c.parse_request(item) for item in args.items], step=step,
                                     fit_seconds=args.fit_seconds, wait=not args.no_wait),
        "wait": lambda: c.cmd_wait(remote, session, names=args.items or None, any_unit=args.any, step=step),
        "pull": lambda: b.cmd_pull(remote, args.lanes),
        "recover-lane": lambda: c.cmd_recover_lane(remote, session, int(args.items[0]), step),
        "archive-failed": lambda: c.cmd_archive_failed(remote, args.items[0], step),
        "evidence": lambda: c.cmd_evidence(remote, allow_running=args.allow_running, step=step),
    }
    show(actions[args.command]())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (LauncherError, b.host_pull.HostPullError) as exc:
        print(f"step F launcher refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
