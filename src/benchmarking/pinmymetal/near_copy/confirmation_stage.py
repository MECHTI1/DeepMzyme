#!/usr/bin/env python3
"""Workstation launcher for v3 plan step E2 (standard library only).

Step E2 (plan step E2, log v3-025; rules frozen in docs/campaigns/pmm_ion_metal_v3/e2_assessment_spec.md and
.json, log v3-028) tests the eight remaining step D passes one at a time on folds 1-4 with seed 42: 32 candidate
fits and four Only-GVP ``sitenone`` geometry controls, 36 units. The fits run from the step D/E extension code
directory with its hash-frozen runner files, through the step C lane, launch, wait, pull, recovery and evidence
logic (pmm_v3_step_c.Step) unchanged. The VM runner checks fold, ESMC, extension and hashes but not step
membership, so this launcher is the admission check for E2 units; never call the runner directly. Like the other
launchers it never starts, stops, restores or extends a VM: the external VM controller
($DEEPMZYME_VM_CONTROLLER/bin) owns billing.

Commands (see the metal playbook, "v3 step E2 individual-confirmation amendment"):
  units                   offline: the 36 E2 units and the 17 declared comparisons
  gates                   offline: manifest, ceiling, authorization, spending and readiness records
  build-manifest --log-entry v3-NNN [--original-assessment PATH]
                          CPU, once: the E2 amendment manifest. It binds the frozen E2 and A4 specifications, the
                          complete original step E assessment and its selection by SHA-256 (never a projected
                          winner), the final step D assessment, extension 1, the runner files, the E2 tooling and
                          the evidence of every reused control
  record-ceiling --usd X --spent-usd S --spent-at UTC --spent-source TEXT --log-entry v3-NNN [--reserve-usd R]
                          once: the campaign ceiling every E2 launch must respect (estimated gross charges,
                          storage included)
  record-authorization --log-entry v3-NNN
                          once, only after the user's E2 GPU-start authorization: binds that decision to exactly
                          this manifest and this ceiling record
  spend                   offline: estimated campaign charges now and the ceiling check
  session / status / launch UNIT@LANE ... [--fit-seconds S] [--no-wait] / wait [UNIT ...] [--any] /
  pull [--lanes K ...] / recover-lane K / archive-failed UNIT / evidence [--allow-running]
                          as for steps C-E, with E2's own launch folder (step_e2/launch on the VM) and evidence
                          folder (step_e2_evidence under DeepMzyme_Data/)

Every launch is refused before anything starts unless: the CPU readiness report and the VM extension record agree
with this worktree (as for steps D and E); the manifest verifies (every bound file keeps its hash, the E2 tooling
is unchanged, the VM extension record is the bound one); the recorded authorization names this manifest and this
ceiling record; and the estimated campaign charges, with the running session counted to its hard stop, plus the
recorded reserve stay within the recorded ceiling. The E2 assessment runs on the workstation from the verified
local copies (pmm_v3_e2_assessment.py), after 'evidence'.
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
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import pmm_v3_step_b as b  # noqa: E402
import pmm_v3_step_c as c  # noqa: E402
import pmm_v3_step_d as d  # noqa: E402

LauncherError = b.LauncherError
require = b.require
sha256_file = d.sha256_file

CAMPAIGN = d.CAMPAIGN
SPEC_DIR = ROOT / "docs" / "campaigns" / "pmm_ion_metal_v3"
E2_SPEC_JSON = SPEC_DIR / "e2_assessment_spec.json"
E2_SPEC_MD = SPEC_DIR / "e2_assessment_spec.md"
# Frozen on 2026-10-08 (log v3-028); a different file is refused.
FROZEN_E2_SPEC_SHA256 = {"e2_assessment_spec.json": "ea2d71ca36848ee23a84ca4b1480a4aa7c231778a02ecde81f73e05740a1ef98",
                         "e2_assessment_spec.md": "71606fa62d251236e47dfc731d6ba13fadadf229e7fa747d7c3f3c6697ef2419"}
A4_SPEC_FILES = ("assessment_spec.json", "assessment_spec.md")
# The E2 tools and the frozen modules whose behaviour they reuse (admission, lanes, pulls, A4 statistics).
E2_TOOL_FILES = ("pmm_v3_step_e2.py", "pmm_v3_e2_assessment.py")
REUSED_FILES = ("pmm_v3_assessment.py", "pmm_v3_step_b.py", "pmm_v3_step_c.py", "pmm_v3_step_d.py",
                "pmm_v3_host_pull.py")
TOOL_FILES = E2_TOOL_FILES + REUSED_FILES
FOLDS = (1, 2, 3, 4)
SEED = 42
TARGET = "four_class"
DEVELOPMENT_FOLD = 0
STEP_E_KIND = "E"
EXTENSION_COPY = Path("step_d_evidence") / "extension" / "campaign_extension.json"
STEP_E_ASSESSMENTS = Path("step_e_evidence") / "assessments"
STEP_D_ASSESSMENTS = Path("step_d_evidence") / "assessments"
EVIDENCE_DIR = "step_e2_evidence"
MANIFEST_NAME, CEILING_NAME, AUTHORIZATION_NAME = "e2_manifest.json", "e2_ceiling.json", "e2_authorization.json"
EVIDENCE_PATTERNS_E2 = tuple("step_e2/**/*" if pattern == "step_c/**/*" else pattern
                             for pattern in c.EVIDENCE_PATTERNS) + ("campaign_extension.json",)
LEDGER = b.CONTROLLER / "state" / "usage_ledger.jsonl"
CURRENT_SESSION = b.CONTROLLER / "state" / "current_session.json"
# Estimated charges as in the campaign log (v3-029 onwards): running hours at the gross compute + IP rate without
# the disk share (the disk is counted once, in the daily storage rate: campaign disk $0.493 + archive snapshot
# $0.057 per calendar day).
RUNNING_USD_PER_HOUR = 0.858624
STORAGE_USD_PER_DAY = 0.55
# Kept free after the last E2 launch: step F (draft: up to 2.25 VM hours, $1.93) and seven more calendar days of
# storage until step F completes ($3.85), rounded up.
DEFAULT_RESERVE_USD = 6.0
DEFAULT_RESERVE_BASIS = ("step F up to 2.25 VM hours ($1.93) plus 7 calendar days of storage until step F completes "
                         "($3.85), rounded up")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stamp_seconds(value: str) -> float:
    return b._stamp(value)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_once(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Create a record file exactly once (O_EXCL); a change needs a new dated log entry and the user's decision."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        raise LauncherError(f"{path} exists; it is recorded once (a change needs a new dated log entry and the "
                            "user's decision; archive the old record by hand first)") from None
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return read_json(path)


# ---------------------------------------------------------------------------
# The frozen E2 specification: units and the declared comparison family
# ---------------------------------------------------------------------------

def load_e2_spec(path: Path = E2_SPEC_JSON, *, expected_sha256: str | None = None) -> dict[str, Any]:
    """The frozen E2 specification (refused unless its SHA-256 is the one frozen in log v3-028)."""
    expected = FROZEN_E2_SPEC_SHA256["e2_assessment_spec.json"] if expected_sha256 is None else expected_sha256
    require(sha256_file(path) == expected, f"{path} differs from the frozen E2 specification (log v3-028)")
    spec = read_json(path)
    require(spec.get("step") == "E2" and spec.get("target") == TARGET and list(spec.get("folds", [])) == list(FOLDS)
            and spec.get("epochs") == 50 and spec.get("checkpoint_rule") == "terminal"
            and spec.get("held_out_test_accessed") is False, "Not the E2 specification this launcher was built for")
    return spec


def cell_of(family: str, recipe: str, target: str = TARGET) -> str:
    return f"{family}__{target}__{recipe}"


def unit_of(cell: str, fold: int, seed: int = SEED) -> str:
    return f"{cell}__fold{fold}__seed{seed}"


def candidate_cells(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """The eight candidates in the declared order, as cells with their comparators."""
    out = []
    for item in spec["candidates"]:
        out.append({"cell": cell_of(item["family"], item["recipe"]), "family": item["family"],
                    "recipe": item["recipe"], "matched_control": item["matched_control"],
                    "additional_comparator": item.get("additional_comparator")})
    return out


def e2_units(spec: dict[str, Any]) -> tuple[str, ...]:
    """Every E2 fit: four folds per candidate in the declared order, then the new controls."""
    units = [unit_of(item["cell"], fold) for item in candidate_cells(spec) for fold in FOLDS]
    for cell, folds in spec["new_control_fits"].items():
        units += [unit_of(cell, int(fold)) for fold in folds]
    counts = spec["fit_counts"]
    require(len(units) == len(set(units)) == counts["total"]
            and len(units) - sum(len(f) for f in spec["new_control_fits"].values()) == counts["candidate_fits"],
            "The E2 unit list does not match the specification's fit counts")
    return tuple(units)


def declared_comparisons(spec: dict[str, Any], selection_cell: str) -> list[dict[str, Any]]:
    """C1-C8 candidate vs matched control, C9 the geometry candidate vs the Only-GVP baseline, C10-C17 candidate vs
    the original step E selection. A comparison repeating an earlier (candidate, comparator) pair is flagged
    coincident; the family and the divisor stay those of the specification."""
    candidates = candidate_cells(spec)
    out = [{"id": f"C{k}", "candidate": item["cell"], "comparator": item["matched_control"],
            "role": "matched control"} for k, item in enumerate(candidates, 1)]
    extra = [item for item in candidates if item["additional_comparator"]]
    require(len(extra) == 1, "The specification names one candidate with an additional comparator")
    out.append({"id": f"C{len(candidates) + 1}", "candidate": extra[0]["cell"],
                "comparator": extra[0]["additional_comparator"], "role": "additional comparator"})
    start = len(out) + 1
    out += [{"id": f"C{start + k}", "candidate": item["cell"], "comparator": selection_cell,
             "role": "original selection"} for k, item in enumerate(candidates)]
    require(len(out) == spec["comparison_family_size"], "The declared family differs from the specification")
    first: dict[tuple[str, str], str] = {}
    for item in out:
        key = (item["candidate"], item["comparator"])
        item["coincident_with"] = first.get(key)
        first.setdefault(key, item["id"])
    return out


# ---------------------------------------------------------------------------
# The E2 amendment manifest (CPU, once)
# ---------------------------------------------------------------------------

def original_assessment_problems(assessment: dict[str, Any]) -> list[str]:
    """E2 specification section 1: the original step E assessment must be complete."""
    result = assessment.get("result") or {}
    problems = []
    if assessment.get("kind") != STEP_E_KIND:
        problems.append(f"not a step E assessment (kind {assessment.get('kind')!r})")
    contrasts = (result.get("neutral_target_test") or {}).get("contrasts") or {}
    if len(contrasts) != 6 or any(item.get("status") != "complete" for item in contrasts.values()):
        problems.append("the six neutral contrasts are not all complete")
    improvements = result.get("improvement_check") or {}
    if not improvements or any(item.get("status") in (None, "incomplete") for item in improvements.values()):
        problems.append("the improvement checks are not complete")
    if not (result.get("stage6_selection") or {}).get("selected"):
        problems.append("no Stage 6 selection")
    if result.get("held_out_test_accessed") is not False:
        problems.append("held_out_test_accessed is not false")
    recipes = {family: item.get("recipe") for family, item in improvements.items()}
    cells, units = a4_expected(recipes)
    if set(assessment.get("unit_evidence") or {}) != set(units):
        problems.append(f"its unit evidence is not exactly the {len(units)} units of the step E assessment")
    if set(result.get("summaries") or {}) != set(cells) or len((result.get("stage6_selection") or {})
                                                               .get("candidates") or []) != len(cells):
        problems.append(f"its summaries or Stage 6 candidates are not exactly the {len(cells)} step E cells")
    return problems


def a4_expected(improvement_recipes: dict[str, str]) -> tuple[list[str], list[str]]:
    """The cells and units an original step E assessment covers (pmm_v3_assessment.main, step E): every
    family x target baseline and each improved recipe, on folds 0-4 with seed 42."""
    cells = [f"{family}__{target}__baseline" for family in c.FAMILIES for target in c.TARGETS]
    cells += [cell_of(family, recipe) for family, recipe in sorted(improvement_recipes.items())]
    return cells, [unit_of(cell, fold) for cell in cells for fold in range(5)]


def selection_cell_of(assessment: dict[str, Any]) -> tuple[str, str]:
    cell = assessment["result"]["stage6_selection"]["selected"]
    parts = cell.split("__")
    require(len(parts) == 3 and parts[1] in c.TARGETS and parts[0] in c.FAMILIES,
            f"The original selection {cell!r} is not a family__target__recipe cell")
    return cell, parts[1]


def latest_kind(directory: Path, kind: str) -> Path | None:
    found = [path for path in directory.glob(f"{kind}_*/assessment.json")]
    return max(found, key=lambda path: path.parent.name.rsplit("_", 1)[1]) if found else None


def reused_unit_names(spec: dict[str, Any], selection_cell: str) -> dict[str, list[str]]:
    """Completed units E2 reads but never runs: comparators on folds 1-4 (matched controls, the additional
    comparator, the original selection) and the fold-0 seed-42 development units."""
    candidates = candidate_cells(spec)
    comparator_cells = {item["matched_control"] for item in candidates} | {selection_cell}
    comparator_cells |= {item["additional_comparator"] for item in candidates if item["additional_comparator"]}
    new_controls = set(spec["new_control_fits"])
    folds_1_4 = sorted(unit_of(cell, fold) for cell in comparator_cells - new_controls for fold in FOLDS)
    development = sorted({unit_of(cell, DEVELOPMENT_FOLD)
                          for cell in comparator_cells | {item["cell"] for item in candidates}})
    return {"folds_1_4": folds_1_4, "development_fold0": development}


def build_manifest(campaign: Path = CAMPAIGN, *, log_entry: str, original: Path | None = None,
                   root: Path = ROOT) -> dict[str, Any]:
    """The E2 amendment manifest: refused unless every binding holds now."""
    require(str(log_entry).startswith("v3-"), "Name the campaign log entry (--log-entry v3-NNN)")
    spec = load_e2_spec()
    md = sha256_file(E2_SPEC_MD)
    require(md == FROZEN_E2_SPEC_SHA256["e2_assessment_spec.md"], "e2_assessment_spec.md differs from the frozen one")
    a4 = {name: sha256_file(SPEC_DIR / name) for name in A4_SPEC_FILES}
    require(a4 == spec["a4_spec_sha256"], "The A4 specification differs from the one the E2 specification binds")
    original = original or latest_kind(campaign / STEP_E_ASSESSMENTS, STEP_E_KIND)
    require(original is not None and original.is_file(), "No original step E assessment copy on the workstation")
    assessment = read_json(original)
    problems = original_assessment_problems(assessment)
    require(not problems, f"The original step E assessment {original} is not complete: {problems}")
    require(assessment.get("spec_sha256") == a4["assessment_spec.json"],
            "The original assessment was made under another A4 specification")
    final_d = latest_kind(campaign / STEP_D_ASSESSMENTS, "D-B")
    require(final_d is not None, "No final step D assessment copy on the workstation")
    d_assessment = read_json(final_d)
    finals = d.final_recipes({**d_assessment, "path": str(final_d)})
    require(finals is not None, f"{final_d} is not a final step D assessment")
    improved = {family: item["recipe"] for family, item in assessment["result"]["improvement_check"].items()}
    require(improved == {family: recipe for family, recipe in finals.items() if recipe != "baseline"},
            f"The original assessment checked improvement recipes {improved}, not the final step D recipes {finals}")
    extension_path = campaign / EXTENSION_COPY
    require(extension_path.is_file(), f"No workstation copy of the extension record ({extension_path})")
    extension = read_json(extension_path)
    extension_sha = sha256_file(extension_path)
    require(extension.get("extension_id") == d.EXTENSION_ID, "Unknown extension record")
    require((assessment.get("campaign_extension") or {}).get("sha256") == extension_sha
            and (d_assessment.get("campaign_extension") or {}).get("sha256") == extension_sha,
            "The original assessments belong to another extension record")
    runner = d.local_runner_sha256(root)
    require(runner == extension["runner_sha256"], "This worktree's runner files differ from the extension record's")
    a4_assessor = sha256_file(root / "pmm_v3_assessment.py")
    require(a4_assessor == assessment.get("assessor_sha256") == extension.get("assessor_sha256"),
            "This worktree's pmm_v3_assessment.py is not the assessor of the original step E assessment and of the "
            "extension record")
    selection, selection_target = selection_cell_of(assessment)
    reused = reused_unit_names(spec, selection)
    bound = {}
    for name in reused["folds_1_4"] + reused["development_fold0"]:
        evidence = assessment["unit_evidence"].get(name) or d_assessment.get("unit_evidence", {}).get(name)
        require(evidence is not None and evidence.get("checkpoint_sha256") and evidence.get("predictions_sha256"),
                f"{name}: no verified evidence in the original assessments; E2 cannot reuse it")
        source = "original step E assessment" if name in assessment["unit_evidence"] else "final step D assessment"
        bound[name] = {**evidence, "source": source}
    units = e2_units(spec)
    require(not set(units) & set(bound), "An E2 unit is also a reused unit")
    manifest = {
        "schema_version": 1, "step": "E2", "campaign_id": "pmm_ion_metal_v3", "created_at": utc_now(),
        "log_entry": log_entry, "git_commit": _git_commit(root),
        "e2_spec_sha256": {"e2_assessment_spec.json": sha256_file(E2_SPEC_JSON), "e2_assessment_spec.md": md},
        "a4_spec_sha256": a4,
        "original_assessment": {"path": str(original), "sha256": sha256_file(original),
                                "assessor_sha256": assessment.get("assessor_sha256"),
                                "campaign_manifest_sha256": assessment.get("manifest_sha256"),
                                "selection": selection, "selection_target": selection_target,
                                "improvement_recipes": improved},
        "final_step_d_assessment": {"path": str(final_d), "sha256": sha256_file(final_d), "final_recipes": finals},
        "extension": {"id": extension["extension_id"], "sha256": extension_sha, "copy": str(extension_path),
                      "parent_manifest_sha256": extension["parent"]["manifest_sha256"]},
        "runner_sha256": runner,
        "tooling_sha256": {name: sha256_file(root / name) for name in TOOL_FILES},
        "units": list(units),
        "new_control_units": [unit_of(cell, int(fold)) for cell, folds in spec["new_control_fits"].items()
                              for fold in folds],
        "comparisons": declared_comparisons(spec, selection),
        "reused_units": bound,
        "remote_code": d.REMOTE_CODE, "launch_folder": "step_e2/launch",
        "held_out_access": False,
    }
    return manifest


def _git_commit(root: Path) -> str | None:
    import subprocess

    try:
        return subprocess.run(["git", "--no-optional-locks", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def verify_manifest(manifest: dict[str, Any], *, root: Path = ROOT,
                    vm_extension: dict[str, Any] | None = None) -> list[str]:
    """Every binding of a recorded manifest still holds (empty list) or the reasons it does not."""
    problems = []
    try:
        spec = load_e2_spec()
    except (LauncherError, OSError, ValueError) as exc:
        return [f"E2 specification: {exc}"]
    current = {"e2_assessment_spec.json": sha256_file(E2_SPEC_JSON), "e2_assessment_spec.md": sha256_file(E2_SPEC_MD)}
    if manifest.get("e2_spec_sha256") != current or current != FROZEN_E2_SPEC_SHA256:
        problems.append("the E2 specification files differ from the manifest or from the frozen values")
    if manifest.get("a4_spec_sha256") != {name: sha256_file(SPEC_DIR / name) for name in A4_SPEC_FILES}:
        problems.append("the A4 specification differs from the manifest")
    for key in ("original_assessment", "final_step_d_assessment"):
        bound = manifest.get(key) or {}
        path = Path(bound.get("path", ""))
        if not path.is_file() or sha256_file(path) != bound.get("sha256"):
            problems.append(f"{key}: {path} is missing or differs from its bound SHA-256")
    original = manifest.get("original_assessment") or {}
    if Path(original.get("path", "")).is_file():
        assessment = read_json(Path(original["path"]))
        problems += [f"original assessment: {p}" for p in original_assessment_problems(assessment)]
        if not original_assessment_problems(assessment) and selection_cell_of(assessment)[0] != original.get("selection"):
            problems.append("the manifest's selection is not the original assessment's")
    extension = manifest.get("extension") or {}
    copy = Path(extension.get("copy", ""))
    if not copy.is_file() or sha256_file(copy) != extension.get("sha256"):
        problems.append("the workstation copy of the extension record differs from the manifest")
    elif read_json(copy).get("assessor_sha256") != sha256_file(root / "pmm_v3_assessment.py"):
        problems.append("pmm_v3_assessment.py is not the assessor the extension record binds")
    if vm_extension is not None and (vm_extension.get("sha256") != extension.get("sha256")
                                     or vm_extension.get("runner_sha256") != manifest.get("runner_sha256")):
        problems.append("the VM extension record is not the one the manifest binds")
    if d.local_runner_sha256(root) != manifest.get("runner_sha256"):
        problems.append("this worktree's runner files differ from the manifest")
    if {name: sha256_file(root / name) for name in TOOL_FILES} != manifest.get("tooling_sha256"):
        problems.append("the E2 tooling or a reused launcher/assessor module changed since the manifest was recorded "
                        "(record a new manifest and a new authorization)")
    if manifest.get("units") != list(e2_units(spec)):
        problems.append("the manifest's units are not the specification's 36")
    if original.get("selection") and manifest.get("comparisons") != declared_comparisons(spec, original["selection"]):
        problems.append("the manifest's comparisons are not the declared family")
    return problems


# ---------------------------------------------------------------------------
# Ceiling and spending (an execution control: estimated charges, never a bill)
# ---------------------------------------------------------------------------

def ledger_sessions(path: Path = LEDGER) -> dict[str, dict[str, Any]]:
    """Stopped controller sessions: id -> start, stop, running seconds (one consistent record per session)."""
    out: dict[str, dict[str, Any]] = {}
    if not Path(path).is_file():
        return out
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") != "STOPPED" or not event.get("session_id"):
            continue
        record = {"start": stamp_seconds(event["session_start"]), "stop": stamp_seconds(event["session_stop"]),
                  "running_seconds": float(event["elapsed_running_s"])}
        previous = out.setdefault(event["session_id"], record)
        require(previous == record, f"The ledger records session {event['session_id']} twice with different values")
    return out


def active_session(path: Path = CURRENT_SESSION) -> dict[str, Any] | None:
    if not Path(path).is_file():
        return None
    state = read_json(path)
    if state.get("active") is not True:
        return None
    return {"session_id": str(state["session_id"]), "start": stamp_seconds(state["session_start"]),
            "deadline": stamp_seconds(state["termination_ts"])}


def estimated_spend(ceiling: dict[str, Any], *, ledger: Path = LEDGER, session_file: Path = CURRENT_SESSION,
                    now: float | None = None) -> dict[str, Any]:
    """Estimated charges since the recorded baseline: closed sessions' running hours, the active session counted
    to its hard stop, storage per calendar day; plus the reserve, against the ceiling."""
    now = time.time() if now is None else now
    base = ceiling["baseline"]
    since = stamp_seconds(base["at"])
    rate, storage_rate = float(ceiling["running_usd_per_hour"]), float(ceiling["storage_usd_per_day"])
    closed, straddling = 0.0, []
    sessions = ledger_sessions(ledger)
    for session_id, record in sorted(sessions.items()):
        if record["start"] < since < record["stop"]:
            straddling.append(session_id)
        elif record["start"] >= since:
            closed += record["running_seconds"]
    require(not straddling, f"Sessions {straddling} straddle the ceiling baseline time; record a new baseline")
    active = active_session(session_file)
    active_seconds, horizon = 0.0, now
    if active is not None and active["session_id"] not in sessions:
        require(active["start"] >= since, "The active session started before the ceiling baseline")
        active_seconds = max(0.0, active["deadline"] - active["start"])
        horizon = max(now, active["deadline"])
    storage_days = max(0.0, (now - since) / 86400.0)
    spent = float(base["spent_usd"]) + closed / 3600.0 * rate + storage_days * storage_rate
    # Committed: the running session to its hard stop, and storage over the same span.
    committed = (spent + active_seconds / 3600.0 * rate
                 + max(0.0, (horizon - now) / 86400.0) * storage_rate)
    projected = committed + float(ceiling["reserve_usd"])
    return {"baseline_spent_usd": float(base["spent_usd"]), "baseline_at": base["at"],
            "closed_running_hours_since": round(closed / 3600.0, 4), "storage_days_since": round(storage_days, 4),
            "spent_usd": round(spent, 2), "active_session": None if active is None else active["session_id"],
            "active_session_to_hard_stop_usd": round(active_seconds / 3600.0 * rate, 2),
            "committed_usd": round(committed, 2), "reserve_usd": float(ceiling["reserve_usd"]),
            "projected_usd": round(projected, 2), "ceiling_usd": float(ceiling["ceiling_usd"]),
            "within_ceiling": projected <= float(ceiling["ceiling_usd"]) + 1e-9,
            "basis": "estimated gross charges (running hours at the compute+IP rate, storage per calendar day)"}


def ceiling_record(*, usd: float, spent_usd: float, spent_at: str, spent_source: str, log_entry: str,
                   reserve_usd: float = DEFAULT_RESERVE_USD,
                   reserve_basis: str = DEFAULT_RESERVE_BASIS) -> dict[str, Any]:
    require(usd > 0 and 0 <= spent_usd < usd and reserve_usd >= 0, "Implausible ceiling, spending or reserve")
    stamp_seconds(spent_at)
    require(str(log_entry).startswith("v3-"), "Name the campaign log entry (--log-entry v3-NNN)")
    return {"ceiling_usd": float(usd), "log_entry": log_entry, "recorded_at": utc_now(),
            "baseline": {"spent_usd": float(spent_usd), "at": spent_at, "source": spent_source},
            "running_usd_per_hour": RUNNING_USD_PER_HOUR, "storage_usd_per_day": STORAGE_USD_PER_DAY,
            "reserve_usd": float(reserve_usd),
            "reserve_basis": reserve_basis,
            "scope": "campaign total, gross, storage included; E2 launches are refused above it",
            "grants": "nothing: a GPU start needs the user's recorded E2 authorization"}


def authorization_record(*, manifest_path: Path, ceiling_path: Path, log_entry: str) -> dict[str, Any]:
    require(manifest_path.is_file() and ceiling_path.is_file(), "Record the manifest and the ceiling first")
    require(str(log_entry).startswith("v3-"), "Name the campaign log entry of the user's decision (--log-entry)")
    return {"decision": "the user's E2 GPU-start authorization", "log_entry": log_entry, "recorded_at": utc_now(),
            "manifest_sha256": sha256_file(manifest_path), "ceiling_sha256": sha256_file(ceiling_path)}


# ---------------------------------------------------------------------------
# Launch gates (pure apart from reading the workstation records)
# ---------------------------------------------------------------------------

def record_paths(campaign: Path) -> dict[str, Path]:
    evidence = campaign / EVIDENCE_DIR
    return {"manifest": evidence / MANIFEST_NAME, "ceiling": evidence / CEILING_NAME,
            "authorization": evidence / AUTHORIZATION_NAME}


def e2_blockers(names: list[str], state: dict[str, Any], *, campaign: Path, readiness: dict[str, Any] | None,
                runner: dict[str, str], root: Path = ROOT,
                spend: Callable[[dict[str, Any]], dict[str, Any]] = estimated_spend) -> list[str]:
    problems = d.readiness_blockers(state, readiness, runner)
    paths = record_paths(campaign)
    manifest = d.read_record(paths["manifest"])
    if manifest is None:
        problems.append("no E2 manifest ('build-manifest'); nothing of E2 starts without it")
    else:
        problems += [f"E2 manifest: {p}" for p in verify_manifest(manifest, root=root,
                                                                   vm_extension=state.get("extension"))]
    ceiling = d.read_record(paths["ceiling"])
    if ceiling is None:
        problems.append("no E2 ceiling record ('record-ceiling'); the E2 ceiling must be in the execution controls")
    authorization = d.read_record(paths["authorization"])
    if authorization is None:
        problems.append("no recorded E2 GPU-start authorization: E2 fits need the user's separate authorization "
                        "('record-authorization' after the decision is logged)")
    elif manifest is not None and ceiling is not None:
        if authorization.get("manifest_sha256") != sha256_file(paths["manifest"]):
            problems.append("the recorded authorization names another E2 manifest")
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


def build_step(campaign: Path = CAMPAIGN, *, runner: dict[str, str] | None = None, root: Path = ROOT,
               spend: Callable[[dict[str, Any]], dict[str, Any]] = estimated_spend) -> c.Step:
    """The step C launcher logic with E2's units, gates and folders; the fits run from the extension code."""
    spec = load_e2_spec()
    units = e2_units(spec)
    readiness = d.latest_readiness(campaign)
    runner = d.local_runner_sha256(root) if runner is None else runner
    return c.Step(
        key="e2", label="step E2",
        outside="E2 admits only the 36 units of its specification and manifest ('units'); other steps have their "
                "own launchers",
        second_failure="E2 stops for diagnosis: a second failure blocks the final E2 assessment (E2 specification "
                       "section 8)",
        done_hint="every E2 unit has run: 'evidence', vm-stop, then the E2 assessment on the workstation",
        units=lambda state: units,
        batch_blockers=lambda names, state: e2_blockers(names, state, campaign=campaign, readiness=readiness,
                                                        runner=runner, root=root, spend=spend),
        code=d.REMOTE_CODE, evidence=campaign / EVIDENCE_DIR, evidence_patterns=EVIDENCE_PATTERNS_E2)


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def cmd_units() -> dict[str, Any]:
    spec = load_e2_spec()
    selection = "<the original step E selection, bound by the manifest>"
    return {"step": "E2", "units": list(e2_units(spec)), "counts": spec["fit_counts"],
            "candidates": candidate_cells(spec), "comparisons": declared_comparisons(spec, selection),
            "forecast_seconds": {"warm": c.FIT_SECONDS, "note": "the site-geometry graph caches were built in step D"}}


def cmd_gates(campaign: Path = CAMPAIGN) -> dict[str, Any]:
    paths = record_paths(campaign)
    manifest, ceiling = d.read_record(paths["manifest"]), d.read_record(paths["ceiling"])
    readiness = d.latest_readiness(campaign)
    out = {"readiness_report": None if readiness is None else readiness["path"],
           "readiness_matches_worktree_runner": None if readiness is None
           else readiness.get("runner_sha256") == d.local_runner_sha256(),
           "manifest": None if manifest is None else {"path": str(paths["manifest"]),
                                                      "sha256": sha256_file(paths["manifest"]),
                                                      "problems": verify_manifest(manifest),
                                                      "selection": manifest["original_assessment"]["selection"]},
           "ceiling": ceiling, "authorization": d.read_record(paths["authorization"])}
    if ceiling is not None:
        out["spend"] = estimated_spend(ceiling)
    return out


def cmd_status(remote: b.Remote, step: c.Step, campaign: Path = CAMPAIGN) -> dict[str, Any]:
    status = c.cmd_status(remote, step)
    state = c.vm_state(remote, step=step)
    status["readiness_blockers"] = d.readiness_blockers(state, d.latest_readiness(campaign), d.local_runner_sha256())
    status["gates"] = cmd_gates(campaign)
    return status


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("units", "gates", "build-manifest", "record-ceiling",
                                            "record-authorization", "spend", "session", "status", "launch", "wait",
                                            "pull", "recover-lane", "archive-failed", "evidence"))
    parser.add_argument("items", nargs="*", help="launch: UNIT@LANE ...; wait: optional UNIT ...; archive-failed: "
                                                 "UNIT; recover-lane: K")
    parser.add_argument("--fit-seconds", type=float, help="launch: admission forecast for every unit")
    parser.add_argument("--no-wait", action="store_true", help="launch: return after the launch")
    parser.add_argument("--any", action="store_true", help="wait: return after the first unit ends")
    parser.add_argument("--lanes", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--allow-running", action="store_true", help="evidence: emergency copy while units run")
    parser.add_argument("--log-entry", help="build-manifest / record-*: the dated campaign log entry")
    parser.add_argument("--original-assessment", type=Path, help="build-manifest: the original step E assessment "
                                                                 "copy (default: the newest kind-E copy)")
    parser.add_argument("--usd", type=float, help="record-ceiling: the campaign ceiling (gross, storage included)")
    parser.add_argument("--spent-usd", type=float, help="record-ceiling: estimated charges at --spent-at")
    parser.add_argument("--spent-at", help="record-ceiling: UTC time of that calculation (YYYY-MM-DDTHH:MM:SSZ)")
    parser.add_argument("--spent-source", help="record-ceiling: where that calculation is recorded (log entry)")
    parser.add_argument("--reserve-usd", type=float, default=DEFAULT_RESERVE_USD,
                        help="record-ceiling: kept free after the last E2 launch (step F and its storage days)")
    parser.add_argument("--reserve-basis", default=DEFAULT_RESERVE_BASIS, help="record-ceiling: how the reserve is sized")
    args = parser.parse_args(argv)
    paths = record_paths(CAMPAIGN)
    if args.command in ("units", "gates"):
        print(json.dumps(cmd_units() if args.command == "units" else cmd_gates(), indent=2, sort_keys=True))
        return 0
    if args.command == "build-manifest":
        manifest = build_manifest(log_entry=args.log_entry, original=args.original_assessment)
        written = write_json_once(paths["manifest"], manifest)
        print(json.dumps({"manifest": str(paths["manifest"]), "sha256": sha256_file(paths["manifest"]),
                          "selection": written["original_assessment"]["selection"],
                          "original_assessment_sha256": written["original_assessment"]["sha256"],
                          "units": len(written["units"])}, indent=2))
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
        print(json.dumps(estimated_spend(ceiling), indent=2, sort_keys=True))
        return 0
    session = b.read_session()
    if args.command == "session":
        print(json.dumps(session, indent=2, sort_keys=True))
        return 0
    remote = b.Remote(b.check_ssh_config())
    step = build_step()
    if args.command == "launch":
        require(bool(args.items), "launch needs UNIT@LANE ...")
    if args.command == "archive-failed":
        require(len(args.items) == 1, "archive-failed needs one unit")
    if args.command == "recover-lane":
        require(len(args.items) == 1 and args.items[0].isdigit(), "recover-lane needs one lane number")
    actions = {
        "status": lambda: cmd_status(remote, step),
        "launch": lambda: c.cmd_launch(remote, session, [c.parse_request(item) for item in args.items],
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
        print(f"step-E2 launcher refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
