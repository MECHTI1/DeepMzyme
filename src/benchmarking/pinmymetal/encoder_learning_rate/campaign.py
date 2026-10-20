#!/usr/bin/env python3
"""Workstation launcher of the trunk-learning-rate comparison (campaign ``pmm_trunk_lr_2026-10``; standard library only).

Rules: docs/campaigns/pmm_trunk_lr_2026-10/trunklr_assessment_spec.json (this tool pins the file's SHA-256 and refuses
another). Five fits: the v3-selected cell ``gvp_late_fusion__four_class__baseline`` on folds 0-4, seed 42, with the
GVP-trunk learning rate 1e-4 instead of 3e-4, against the five verified fits of the same cell (reused, never rerun;
fold 0 is the reused step-B full-fp32 run). The fits run through run_pmm_trunklr.py, which this launcher copies next
to the hash-frozen v3 runner in the extension code directory ('deploy'), and through the step C lane, wait, pull,
recovery and evidence logic (pmm_v3_step_c) unchanged, as for the v3 10-epoch comparison (pmm_v3_sched10.py). Like
the other launchers it never starts, stops, restores or extends a VM: the external VM controller
($DEEPMZYME_VM_CONTROLLER/bin) owns billing.

Campaign identity: the records (manifest, ceiling, authorization), evidence copies and assessments live in their own
folder under DeepMzyme_Data/campaigns (RECORDS). The fits execute in the v3 execution root (VM campaign root and
local durable pulls) under their own run names, which no v3 unit uses; no v3 file is changed.

Commands:
  units                   offline: the five fits, their controls and the admission forecast
  gates                   offline: manifest, ceiling, authorization, spending and readiness records
  build-manifest --log-entry tlr-NNN
                          CPU, once: binds the frozen specification, the A4 and E2 assessments, the E2 manifest,
                          extension 1, the runner files, this tooling (VM runner included) and the evidence of the
                          five reused controls
  record-ceiling --usd X --baseline-at UTC --log-entry tlr-NNN [--reserve-usd R]
                          once: this campaign's compute ceiling (running hours of sessions after the baseline; the
                          active session counted to its hard stop; storage excluded)
  record-authorization --log-entry tlr-NNN
                          once, only after the user's GPU-start authorization for this comparison
  spend                   offline: estimated compute charges since the baseline and the ceiling check
  deploy                  copy run_pmm_trunklr.py into the extension code directory on the running VM (only with
                          a verified manifest and the recorded authorization; never overwrites a different file)
  session / status / launch UNIT@LANE ... [--fit-seconds S] [--no-wait] / wait [UNIT ...] [--any] /
  pull [--lanes K ...] / recover-lane K / archive-failed UNIT / evidence [--allow-running]
                          as for the 10-epoch comparison, with this comparison's launch folder (step_tlr/launch on
                          the VM) and evidence copies in RECORDS

Every launch is refused before anything starts unless: the CPU readiness report and the VM extension record agree
with this worktree; the manifest verifies; the VM copy of run_pmm_trunklr.py is the bound one; the recorded
authorization names this manifest and this ceiling record; and the estimated compute charges, with the running
session counted to its hard stop, plus the recorded reserve stay within the recorded ceiling. The assessment runs on
the workstation from the verified local copies (pmm_trunklr_assessment.py), after 'evidence'.
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

CAMPAIGN_ID = "pmm_trunk_lr_2026-10"
COMPARISON_ID = "pmm-trunk-lr-2026-10"
LOG_PREFIX = "tlr-"
CAMPAIGN = d.CAMPAIGN  # the v3 execution root under DeepMzyme_Data/ (durable pulls, A4/E2 assessments)
RECORDS = CAMPAIGN.parent / CAMPAIGN_ID  # this campaign's own records, evidence copies and assessments
SPEC_DIR = ROOT / "docs" / "campaigns" / CAMPAIGN_ID
SPEC_JSON = SPEC_DIR / "trunklr_assessment_spec.json"
# Frozen before any comparison fit (log tlr-001); a different file is refused.
FROZEN_SPEC_SHA256 = "61728a8f8b637537e00b0744cba468759cc08328cb553e5ca2e5bb511efa8a9c"
V3_SPEC_DIR = e2.SPEC_DIR
A4_SPEC_FILES = e2.A4_SPEC_FILES
VM_RUNNER = "run_pmm_trunklr.py"
COMPARISON_TOOL_FILES = ("pmm_trunklr.py", "pmm_trunklr_assessment.py", VM_RUNNER)
REUSED_FILES = ("pmm_v3_assessment.py", "pmm_v3_step_b.py", "pmm_v3_step_c.py", "pmm_v3_step_d.py",
                "pmm_v3_host_pull.py", "pmm_v3_step_e2.py", "pmm_v3_e2_assessment.py",
                "pmm_v3_sched10_assessment.py", "pmm_v3_sched10.py", "run_pmm_v3_sched10.py")
TOOL_FILES = COMPARISON_TOOL_FILES + REUSED_FILES
STEP_KEY = "tlr"
MANIFEST_NAME, CEILING_NAME, AUTHORIZATION_NAME = "tlr_manifest.json", "tlr_ceiling.json", "tlr_authorization.json"
E2_MANIFEST = Path("step_e2_evidence") / e2.MANIFEST_NAME
EVIDENCE_PATTERNS_TLR = tuple(f"step_{STEP_KEY}/**/*" if pattern == "step_c/**/*" else pattern
                              for pattern in c.EVIDENCE_PATTERNS) + ("campaign_extension.json",)
REMOTE_RUNNER = f"{d.REMOTE_CODE}/{VM_RUNNER}"
# Admission forecast of one 50-epoch late-fusion fit at three lanes with warm caches, launch to verified pull
# (pmm_v3_step_c.FIT_SECONDS, step B; measured 31-37 min wall per fit in steps E and E2). The learning rate does not
# change the work per epoch.
FIT_SECONDS = float(c.FIT_SECONDS["gvp_late_fusion"])
DEFAULT_RESERVE_USD = 0.0
DEFAULT_RESERVE_BASIS = "none: the hard stop already bounds the active session, and nothing else is planned"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def require_log_entry(log_entry: str | None) -> None:
    require(str(log_entry or "").startswith(LOG_PREFIX), f"Name the campaign log entry (--log-entry {LOG_PREFIX}NNN)")


# ---------------------------------------------------------------------------
# The frozen specification
# ---------------------------------------------------------------------------

def load_spec(path: Path | None = None, *, expected_sha256: str | None = None) -> dict[str, Any]:
    """The frozen specification (refused unless its SHA-256 is the one frozen in log tlr-001)."""
    path = SPEC_JSON if path is None else path
    expected = FROZEN_SPEC_SHA256 if expected_sha256 is None else expected_sha256
    require(sha256_file(path) == expected, f"{path} differs from the frozen trunk-LR comparison specification")
    spec = read_json(path)
    require(spec.get("step") == "TLR" and spec.get("id") == COMPARISON_ID and spec.get("campaign_id") == CAMPAIGN_ID
            and spec.get("target") == "four_class" and list(spec.get("folds", [])) == [0, 1, 2, 3, 4]
            and spec.get("seed") == 42 and spec.get("epochs") == 50
            and (spec.get("candidate_gvp_learning_rate"), spec.get("control_gvp_learning_rate"),
                 spec.get("base_learning_rate")) == ("1e-4", "3e-4", "3e-5")
            and spec.get("checkpoint_rule") == "terminal" and spec.get("lr_schedule") == "cosine"
            and list(spec["sensitivity"]["folds"]) == [1, 2, 3, 4] and spec.get("held_out_test_accessed") is False,
            "Not the trunk-LR comparison specification this launcher was built for")
    expected_units = [f"{spec['family']}__{spec['target']}__{spec['candidate_label']}__fold{k}__seed{spec['seed']}"
                      for k in spec["folds"]]
    expected_controls = [f"{spec['selection_cell']}__fold{k}__seed{spec['seed']}" for k in spec["folds"]]
    require(spec["units"] == expected_units and spec["controls"] == expected_controls
            and spec["selection_cell"] == f"{spec['family']}__{spec['target']}__{spec['recipe']}",
            "The specification's units or controls are not the selected cell's five folds")
    return spec


# ---------------------------------------------------------------------------
# The launch manifest (CPU, once)
# ---------------------------------------------------------------------------

def a4_problems(assessment: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    """The A4 assessment must hold the selected cell's five fold values with the recorded reference mean."""
    result = assessment.get("result") or {}
    summary = (result.get("summaries") or {}).get(spec["selection_cell"]) or {}
    problems = []
    if result.get("held_out_test_accessed") is not False or assessment.get("held_out_test_accessed", False) is not False:
        problems.append("held_out_test_accessed is not false")
    if summary.get("folds") != [0, 1, 2, 3, 4] or len(summary.get("fold_common4_ba") or []) != 5:
        problems.append(f"no five-fold summary of {spec['selection_cell']}")
    elif abs(float(summary["mean_common4_ba"]) - float(spec["control_source"]["reference_mean_common4_ba"])) > 1e-12:
        problems.append("its mean common-four BA is not the specification's reference")
    return problems


def e2_problems(assessment: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    """The final E2 assessment must be complete and keep the cell the specification names."""
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


def build_manifest(campaign: Path = CAMPAIGN, *, log_entry: str, root: Path = ROOT) -> dict[str, Any]:
    """The comparison's launch manifest: refused unless every binding holds now."""
    require_log_entry(log_entry)
    spec = load_spec()
    source = spec["control_source"]
    a4_spec = {name: sha256_file(V3_SPEC_DIR / name) for name in A4_SPEC_FILES}
    bound = {}
    for key in ("a4_assessment", "e2_assessment"):
        path = campaign / source[key]
        require(path.is_file() and sha256_file(path) == source[f"{key}_sha256"],
                f"{path} is missing or not the assessment the specification names")
        bound[key] = (path, read_json(path))
    a4_path, a4 = bound["a4_assessment"]
    e2_path, e2_assessment = bound["e2_assessment"]
    problems = [f"A4: {p}" for p in a4_problems(a4, spec)] + [f"E2: {p}" for p in e2_problems(e2_assessment, spec)]
    require(not problems, "; ".join(problems))
    require(e2_assessment.get("a4_spec_sha256") == a4_spec, "The E2 assessment was made under another A4 specification")
    e2_manifest_path = campaign / E2_MANIFEST
    e2_manifest = read_json(e2_manifest_path)
    require(e2_assessment["manifest"]["sha256"] == sha256_file(e2_manifest_path),
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
    keys = ("checkpoint_sha256", "predictions_sha256")
    a4_values = a4["result"]["summaries"][spec["selection_cell"]]["fold_common4_ba"]
    for fold, name in zip(spec["folds"], spec["controls"]):
        evidence = (a4.get("unit_evidence") or {}).get(name)
        others = [(e2_assessment.get("unit_evidence") or {}).get(name), (e2_manifest.get("reused_units") or {}).get(name)]
        require(evidence is not None and all(other is not None and all(other[k] == evidence[k] for k in keys)
                                             for other in others),
                f"{name}: the A4 assessment, the E2 assessment and the E2 manifest do not bind the same evidence")
        controls[name] = {**{k: evidence[k] for k in keys}, "fold": fold, "a4_common4_ba": a4_values[fold],
                          "source": "A4 unit evidence (= E2 unit evidence = E2 manifest reused unit)"}
    return {
        "schema_version": 1, "step": "TLR", "id": COMPARISON_ID, "campaign_id": CAMPAIGN_ID,
        "created_at": utc_now(), "log_entry": log_entry, "git_commit": e2._git_commit(root),
        "spec": {"path": str(SPEC_JSON), "sha256": sha256_file(SPEC_JSON)},
        "a4_spec_sha256": a4_spec,
        "a4_assessment": {"path": str(a4_path), "sha256": sha256_file(a4_path),
                          "reference_mean_common4_ba": source["reference_mean_common4_ba"]},
        "e2_assessment": {"path": str(e2_path), "sha256": sha256_file(e2_path),
                          "selection": e2_assessment["result"]["final_selection"]["selected"]},
        "e2_manifest": {"path": str(e2_manifest_path), "sha256": sha256_file(e2_manifest_path)},
        "extension": {"id": extension["extension_id"], "sha256": extension_sha, "copy": str(extension_path),
                      "parent_manifest_sha256": extension["parent"]["manifest_sha256"]},
        "runner_sha256": runner,
        "tooling_sha256": {name: sha256_file(root / name) for name in TOOL_FILES},
        "vm_runner": {"file": VM_RUNNER, "remote_path": REMOTE_RUNNER, "sha256": sha256_file(root / VM_RUNNER)},
        "units": list(spec["units"]), "controls": controls,
        "remote_code": d.REMOTE_CODE, "launch_folder": f"step_{STEP_KEY}/launch", "records": str(RECORDS),
        "held_out_access": False,
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
    if manifest.get("a4_spec_sha256") != {name: sha256_file(V3_SPEC_DIR / name) for name in A4_SPEC_FILES}:
        problems.append("the A4 specification differs from the manifest")
    for key in ("a4_assessment", "e2_assessment", "e2_manifest"):
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
        problems.append(f"the bound VM runner differs from this worktree's {VM_RUNNER}")
    if vm_runner_sha256 is not None and vm_runner_sha256 != (manifest.get("vm_runner") or {}).get("sha256"):
        problems.append(f"the VM copy of {VM_RUNNER} is missing or not the bound one ('deploy')")
    if manifest.get("units") != spec["units"] or sorted(manifest.get("controls") or {}) != sorted(spec["controls"]):
        problems.append("the manifest's units or controls are not the specification's")
    return problems


# ---------------------------------------------------------------------------
# Compute ceiling and authorization (execution controls; never permission by themselves)
# ---------------------------------------------------------------------------

def ceiling_record(*, usd: float, baseline_at: str, log_entry: str, reserve_usd: float = DEFAULT_RESERVE_USD,
                   reserve_basis: str = DEFAULT_RESERVE_BASIS) -> dict[str, Any]:
    require(usd > 0 and 0 <= reserve_usd < usd, "Implausible ceiling or reserve")
    e2.stamp_seconds(baseline_at)
    require_log_entry(log_entry)
    return {"ceiling_usd": float(usd), "log_entry": log_entry, "recorded_at": utc_now(),
            "baseline": {"at": baseline_at, "spent_usd": 0.0, "source": "campaign start; compute only"},
            "running_usd_per_hour": e2.RUNNING_USD_PER_HOUR, "reserve_usd": float(reserve_usd),
            "reserve_basis": reserve_basis,
            "scope": ("this campaign's compute: running hours of controller sessions that start after the baseline, "
                      "the active session counted to its hard stop; storage excluded (the kept disk and snapshots "
                      "bill whether or not this comparison runs); trunk-LR launches are refused above it"),
            "grants": "nothing: a GPU start needs the user's recorded authorization for this comparison"}


def estimated_spend(ceiling: dict[str, Any], *, ledger: Path = e2.LEDGER, session_file: Path = e2.CURRENT_SESSION,
                    now: float | None = None) -> dict[str, Any]:
    """Estimated compute charges since the baseline: closed sessions' running hours and the active session counted to
    its hard stop, plus the reserve, against the ceiling (pmm_v3_step_e2.estimated_spend without storage)."""
    now = time.time() if now is None else now
    since = e2.stamp_seconds(ceiling["baseline"]["at"])
    rate = float(ceiling["running_usd_per_hour"])
    closed, straddling = 0.0, []
    sessions = e2.ledger_sessions(ledger)
    for session_id, record in sorted(sessions.items()):
        if record["start"] < since < record["stop"]:
            straddling.append(session_id)
        elif record["start"] >= since:
            closed += record["running_seconds"]
    require(not straddling, f"Sessions {straddling} straddle the ceiling baseline time; record a new baseline")
    active = e2.active_session(session_file)
    active_seconds = 0.0
    if active is not None and active["session_id"] not in sessions:
        require(active["start"] >= since, "The active session started before the ceiling baseline")
        active_seconds = max(0.0, active["deadline"] - active["start"])
    spent = closed / 3600.0 * rate
    committed = spent + active_seconds / 3600.0 * rate
    projected = committed + float(ceiling["reserve_usd"])
    return {"baseline_at": ceiling["baseline"]["at"], "closed_running_hours_since": round(closed / 3600.0, 4),
            "spent_usd": round(spent, 2), "active_session": None if active is None else active["session_id"],
            "active_session_to_hard_stop_usd": round(active_seconds / 3600.0 * rate, 2),
            "committed_usd": round(committed, 2), "reserve_usd": float(ceiling["reserve_usd"]),
            "projected_usd": round(projected, 2), "ceiling_usd": float(ceiling["ceiling_usd"]),
            "within_ceiling": projected <= float(ceiling["ceiling_usd"]) + 1e-9,
            "basis": "estimated compute charges (running hours at the compute+IP rate); storage excluded"}


def authorization_record(*, manifest_path: Path, ceiling_path: Path, log_entry: str) -> dict[str, Any]:
    require(manifest_path.is_file() and ceiling_path.is_file(), "Record the manifest and the ceiling first")
    require_log_entry(log_entry)
    return {"decision": "the user's GPU-start authorization for the trunk-LR comparison", "log_entry": log_entry,
            "recorded_at": utc_now(), "manifest_sha256": sha256_file(manifest_path),
            "ceiling_sha256": sha256_file(ceiling_path)}


def record_paths(records: Path = RECORDS) -> dict[str, Path]:
    return {"manifest": records / MANIFEST_NAME, "ceiling": records / CEILING_NAME,
            "authorization": records / AUTHORIZATION_NAME}


# ---------------------------------------------------------------------------
# Launch gates
# ---------------------------------------------------------------------------

def remote_runner_sha256(remote: b.Remote) -> str | None:
    out = remote.run(f"if [ -f {shlex.quote(REMOTE_RUNNER)} ]; then sha256sum {shlex.quote(REMOTE_RUNNER)}; fi").stdout
    return out.split()[0] if out.strip() else None


def tlr_blockers(names: list[str], state: dict[str, Any], *, records: Path, readiness: dict[str, Any] | None,
                 runner: dict[str, str], vm_runner: Callable[[], str | None], root: Path = ROOT,
                 spend: Callable[[dict[str, Any]], dict[str, Any]] = estimated_spend) -> list[str]:
    problems = d.readiness_blockers(state, readiness, runner)
    paths = record_paths(records)
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
            problems.append(f"estimated compute charges {estimate['committed_usd']:.2f} USD with the running session "
                            f"to its hard stop, plus the {estimate['reserve_usd']:.2f} USD reserve, exceed the "
                            f"recorded {estimate['ceiling_usd']:.2f} USD ceiling: stop and close out, or ask the user "
                            "to record a larger ceiling")
    return problems


def build_step(campaign: Path = CAMPAIGN, *, records: Path = RECORDS, vm_runner: Callable[[], str | None],
               runner: dict[str, str] | None = None, root: Path = ROOT,
               spend: Callable[[dict[str, Any]], dict[str, Any]] = estimated_spend) -> c.Step:
    """The step C lane logic with this comparison's five fits, gates, forecast and folders."""
    units = tuple(load_spec()["units"])
    readiness = d.latest_readiness(campaign)
    runner = d.local_runner_sha256(root) if runner is None else runner
    return c.Step(
        key=STEP_KEY, label="trunk-LR comparison",
        outside="the comparison admits only its five fits ('units'); other work has its own launchers",
        second_failure="the comparison stops: a second failure blocks the decision (specification stopping rule)",
        done_hint="all five fits have run: 'evidence', vm-stop, then pmm_trunklr_assessment.py on the workstation",
        units=lambda state: units,
        batch_blockers=lambda names, state: tlr_blockers(names, state, records=records, readiness=readiness,
                                                         runner=runner, vm_runner=vm_runner, root=root, spend=spend),
        forecast=lambda name, state: FIT_SECONDS,
        code=d.REMOTE_CODE, evidence=records, evidence_patterns=EVIDENCE_PATTERNS_TLR)


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


def cmd_deploy(remote: b.Remote, records: Path = RECORDS, *, root: Path = ROOT) -> dict[str, Any]:
    """Copy the bound VM runner next to the frozen runner (only into the extension code directory, never over a
    different file), then verify its hash on the VM."""
    paths = record_paths(records)
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
    return {"step": "TLR", "units": spec["units"], "controls": spec["controls"],
            "single_change": spec["single_change"], "forecast_seconds_per_fit": FIT_SECONDS,
            "records": str(RECORDS)}


def cmd_gates(campaign: Path = CAMPAIGN, records: Path = RECORDS) -> dict[str, Any]:
    paths = record_paths(records)
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
        out["spend"] = estimated_spend(ceiling)
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
    parser.add_argument("--log-entry", help=f"build-manifest / record-*: the dated campaign log entry ({LOG_PREFIX}NNN)")
    parser.add_argument("--usd", type=float, help="record-ceiling: the compute ceiling (running charges only)")
    parser.add_argument("--baseline-at", help="record-ceiling: UTC time after which sessions count (YYYY-MM-DDTHH:MM:SSZ)")
    parser.add_argument("--reserve-usd", type=float, default=DEFAULT_RESERVE_USD,
                        help="record-ceiling: kept free after the last launch")
    parser.add_argument("--reserve-basis", default=DEFAULT_RESERVE_BASIS, help="record-ceiling: how the reserve is sized")
    args = parser.parse_args(argv)
    paths = record_paths()
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
        require(None not in (args.usd, args.baseline_at), "record-ceiling needs --usd and --baseline-at")
        record = ceiling_record(usd=args.usd, baseline_at=args.baseline_at, log_entry=args.log_entry,
                                reserve_usd=args.reserve_usd, reserve_basis=args.reserve_basis)
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
        print(json.dumps(estimated_spend(ceiling), indent=2, sort_keys=True))
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
        print(f"trunk-LR comparison launcher refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
