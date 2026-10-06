#!/usr/bin/env python3
"""Workstation launcher for v3 plan steps D and E (standard library only).

It reuses the step-C launcher's lane, launch, wait, pull, recovery and evidence logic (pmm_v3_step_c.Step) and
adds the gates of the amended step D (extension 1, log v3-017) and of step E. Like the other launchers it never
starts, stops, restores or extends a VM: the controller (~/deepmzyme-vm/bin) owns billing.

Commands (see the metal playbook, "v3 step D regularization amendment"):
  units [--step D|E]      offline: the frozen unit lists per round and the admission forecasts
  gates [--step D|E]      offline: the CPU readiness report, the runner files and the latest assessment gates
  session                 print the controller session (refuses if inactive, expired or mismatched)
  push-bundle --bundle-dir DIR   copy the extension bundle and apply it into its own new code directory on the VM
                          (the first code directory stays untouched; the fold set there is verified, not replaced)
  extend                  record extension 1 on the VM campaign root once, before the first step D fit, and keep a
                          checked copy of the record on the workstation
  status                  read-only: lanes, launches, claims, unit statuses, the extension record and the gates
  launch UNIT@LANE ... [--fit-seconds S] [--no-wait]
                          start 1-3 units together, one per lane, each detached; refused before readiness, for a
                          round whose gate is not met and without enough session time
  wait [UNIT ...] [--any] / pull [--lanes K ...] / recover-lane K / archive-failed UNIT   as in step C
  assess --round D-A|D-R|D-B [--not-tested-cost RECIPE ...]   (step D)   or   assess   (step E)
                          run the assessor on the VM and keep a checked copy; the copy is the gate for Round B,
                          for the one combination per family and for the step E improvement units
  record-cost-gate --recipes RECIPE ... --log-entry v3-NNN    record the user's OK for cost-gated candidates
  record-e-gate --label LABEL --log-entry v3-NNN              record the user's final-test label decision
  evidence [--allow-running]   copy the step evidence with SHA-256 checked on both ends; run before every vm-stop

Step D gates (every refusal happens before anything starts):
  readiness   a CPU readiness report (audit_v3_step_d_readiness.py) with ready = true, written for exactly the
              runner files of this worktree, and a VM extension record that froze the same runner files;
  Round A     the 16 fits of the prepared plan (two seed-43 controls included);
  Round R     admitted once every Round A unit has started (A is completed first; R runs whatever A shows);
  Round B     only after an assessment of rounds A and R that shows a pass in either family; the cost-gated
              augmentations also need the recorded user OK and an explicit --fit-seconds;
  combination only the one recipe per family that the assessment of all completed rounds requires.
Step E gates: the recorded final-test label decision, a final step D assessment, and for improvement units the
final recipe of that assessment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pmm_v3_step_b as b  # noqa: E402
import pmm_v3_step_c as c  # noqa: E402

LauncherError = b.LauncherError
require = b.require

CAMPAIGN = Path("/media/mechti/Data1/DeepMzyme_Data/campaigns/pmm_ion_metal_v3")
REMOTE_CODE = f"{b.REMOTE_HOME}/projects/DeepMzyme_v3_ext1"  # the extension bundle's own code directory
EXTENSION_ID = "v3-ext1-round-r"  # pmm_v3_campaign.EXTENSION_ID
RUNNER_FILES = ("pmm_v3_campaign.py", "run_pmm_v3_campaign.py", "pmm_v3_probes.py", "pmm_v3_speed_report.py")
GVP_FAMILIES = ("only_gvp", "gvp_late_fusion")
FUSION = ("gvp_late_fusion",)
SEEDS = (42, 43)
# pmm_v3_campaign.step_units("D-A" | "D-R" | "D-B") (checked by tests; importing the campaign module would import torch).
ROUNDS: dict[str, dict[str, tuple[str, ...]]] = {
    "D-A": {"meanagg": GVP_FAMILIES, "resdrop01": GVP_FAMILIES, "structlr": FUSION, "gvpaux03": FUSION,
            "esmdrop02": FUSION},
    "D-R": {"wd001": GVP_FAMILIES, "wd01": GVP_FAMILIES, "wd10": GVP_FAMILIES, "headdrop01": GVP_FAMILIES,
            "headdrop03": GVP_FAMILIES, "resdrop02": GVP_FAMILIES, "esmdrop04": FUSION},
    "D-B": {"sitenone": GVP_FAMILIES, "sitecountsangles": GVP_FAMILIES, "posnoise01": ("only_gvp",),
            "outerdrop01": ("only_gvp",), "vecnorm": ("only_gvp",), "invsqrtw": GVP_FAMILIES},
}
ROUND_ORDER = ("D-A", "D-R", "D-B")
COST_GATED = ("posnoise01", "outerdrop01")
FINAL_TEST_LABELS = ("clean_subset_primary_not_pristine", "both_results_secondary")
EVIDENCE_PATTERNS_D = tuple("step_d/**/*" if pattern == "step_c/**/*" else pattern
                            for pattern in c.EVIDENCE_PATTERNS) + ("campaign_extension.json",)
EVIDENCE_PATTERNS_E = tuple("step_e/**/*" if pattern == "step_c/**/*" else pattern
                            for pattern in c.EVIDENCE_PATTERNS) + ("campaign_extension.json",)


def round_units(name: str) -> tuple[str, ...]:
    units = [f"{family}__four_class__baseline__fold0__seed43" for family in GVP_FAMILIES] if name == "D-A" else []
    for recipe, families in ROUNDS[name].items():
        units += [f"{family}__four_class__{recipe}__fold0__seed{seed}" for family in families for seed in SEEDS]
    return tuple(units)


def recipe_of(name: str) -> str:
    parts = name.split("__")
    require(len(parts) == 5, f"Not a v3 unit name: {name!r}")
    return parts[2]


def components(recipe: str) -> list[str]:
    return recipe[len("combo-"):].split("+") if recipe.startswith("combo-") else [recipe]


def round_of(name: str) -> str | None:
    """The step D round a unit belongs to (None: not a step D unit)."""
    for round_name in ROUND_ORDER:
        if name in round_units(round_name):
            return round_name
    parts = name.split("__")
    if (len(parts) == 5 and parts[0] in GVP_FAMILIES and parts[1] == "four_class" and parts[2].startswith("combo-")
            and parts[3] == "fold0" and parts[4] in ("seed42", "seed43")):
        return "D-combo"
    return None


def e_neutral_units() -> tuple[str, ...]:
    return tuple(f"{family}__{target}__baseline__fold{fold}__seed42"
                 for family in c.FAMILIES for target in c.TARGETS for fold in (1, 2, 3, 4))


def e_improvement_units(finals: dict[str, str]) -> tuple[str, ...]:
    return tuple(f"{family}__four_class__{recipe}__fold{fold}__seed42"
                 for family, recipe in sorted(finals.items()) if recipe != "baseline" for fold in (1, 2, 3, 4))


# ---------------------------------------------------------------------------
# Workstation gate records (all read-only here except the two 'record-*' commands)
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local_runner_sha256(root: Path = ROOT) -> dict[str, str]:
    return {name: sha256_file(root / name) for name in RUNNER_FILES}


def latest_readiness(campaign: Path) -> dict[str, Any] | None:
    reports = sorted(campaign.glob("audits/d_readiness_*/readiness_report.json"))
    return {**json.loads(reports[-1].read_text()), "path": str(reports[-1])} if reports else None


def latest_assessment(evidence: Path, kinds: tuple[str, ...]) -> dict[str, Any] | None:
    """The newest checked assessment copy of the given kinds (directory names end with a UTC timestamp)."""
    found = [path for path in evidence.glob("assessments/*/assessment.json")
             if path.parent.name.rsplit("_", 1)[0] in kinds]
    if not found:
        return None
    newest = max(found, key=lambda path: path.parent.name.rsplit("_", 1)[1])
    return {**json.loads(newest.read_text()), "path": str(newest)}


def read_record(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.is_file() else None


def write_record_once(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    require(not path.exists(), f"{path} exists; a decision is recorded once (a change needs a new dated log entry "
                               "and the user's decision)")
    require(str(record.get("log_entry", "")).startswith("v3-"), "Name the campaign log entry (--log-entry v3-NNN)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**record, "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                               indent=2, sort_keys=True) + "\n")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Gates (pure: decided from the VM snapshot and the workstation records)
# ---------------------------------------------------------------------------

def readiness_blockers(state: dict[str, Any], readiness: dict[str, Any] | None, runner: dict[str, str]) -> list[str]:
    """No step D or E fit before the CPU readiness report and the VM extension record agree with this worktree."""
    if readiness is None:
        return ["no CPU readiness report (run audit_v3_step_d_readiness.py); nothing starts before readiness"]
    problems = []
    if readiness.get("ready") is not True:
        failed = sorted(name for name, check in (readiness.get("checks") or {}).items() if not check.get("passed"))
        problems.append(f"the CPU readiness report is not ready (failed checks: {failed}): {readiness.get('path')}")
    if readiness.get("runner_sha256") != runner:
        problems.append("the runner files of this worktree differ from the ones the readiness report checked; "
                        "rerun the readiness audit")
    extension = state.get("extension")
    if extension is None:
        problems.append("the VM campaign has no extension record ('push-bundle', then 'extend')")
    elif extension.get("id") != EXTENSION_ID or extension.get("runner_sha256") != runner:
        problems.append("the VM extension record froze other runner files than this worktree's")
    return problems


def gate_assessment_blockers(assessment: dict[str, Any] | None, state: dict[str, Any], what: str) -> list[str]:
    if assessment is None:
        return [f"{what}: no checked assessment copy on the workstation yet ('assess')"]
    recorded = (assessment.get("campaign_extension") or {}).get("sha256")
    if recorded != (state.get("extension") or {}).get("sha256"):
        return [f"{what}: the assessment copy {assessment.get('path')} belongs to another extension record"]
    return []


def round_b_blockers(assessment: dict[str, Any] | None, state: dict[str, Any]) -> list[str]:
    problems = gate_assessment_blockers(assessment, state, "Round B")
    if problems:
        return problems
    rounds = assessment["result"].get("rounds", {})
    if not (rounds.get("D-A", {}).get("complete") and rounds.get("D-R", {}).get("complete")):
        return ["Round B: rounds A and R are not both complete in the latest assessment ('assess --round D-R')"]
    if not (rounds["D-A"]["passed"] or rounds["D-R"]["passed"]):
        return ["Round B: no candidate of rounds A or R passed, so step D ends after Round R and the baselines "
                "are kept (plan stopping rule)"]
    return []


def required_combinations(assessment: dict[str, Any] | None) -> dict[str, str]:
    """family -> the one combination recipe an assessment of every completed round requires."""
    if assessment is None or assessment.get("kind") != "D-B":
        return {}
    result = assessment["result"]
    if not str(result.get("status", "")).startswith("blocked: a combination"):
        return {}
    return {family: item["recipe"] for family, item in result.get("families", {}).items()
            if item.get("decision") == "run combination"}


def final_recipes(assessment: dict[str, Any] | None) -> dict[str, str] | None:
    """family -> final step D recipe, once the latest step D assessment is final (None before that)."""
    if assessment is None or assessment["result"].get("final") is not True:
        return None
    return {family: item["recipe"] for family, item in assessment["result"]["families"].items()}


def started_units(state: dict[str, Any]) -> set[str]:
    return set(state["statuses"]) | set(state["claims"]) | set(state["launches"]) | set(state["archived"])


def step_d_units(state: dict[str, Any], gate: dict[str, Any] | None) -> tuple[str, ...]:
    combos = [f"{family}__four_class__{recipe}__fold0__seed{seed}"
              for family, recipe in sorted(required_combinations(gate).items()) for seed in SEEDS]
    combos += sorted(name for name in started_units(state) if round_of(name) == "D-combo" and name not in combos)
    return tuple(unit for name in ROUND_ORDER for unit in round_units(name)) + tuple(combos)


def step_d_blockers(names: list[str], state: dict[str, Any], *, readiness: dict[str, Any] | None,
                    runner: dict[str, str], gate: dict[str, Any] | None,
                    cost_gate: dict[str, Any] | None, fit_seconds: float | None) -> list[str]:
    problems = readiness_blockers(state, readiness, runner)
    started = started_units(state) | set(names)
    for name in names:
        round_name = round_of(name)
        if round_name == "D-R":
            waiting = [unit for unit in round_units("D-A") if unit not in started]
            if waiting:
                problems.append(f"{name}: Round R follows Round A; {len(waiting)} Round A units have not started "
                                f"(first: {waiting[0]})")
        elif round_name == "D-B":
            problems += [f"{name}: {problem}" for problem in round_b_blockers(gate, state)]
            if recipe_of(name) in COST_GATED:
                approved = (cost_gate or {}).get("recipes", [])
                if recipe_of(name) not in approved:
                    problems.append(f"{name}: cost-gated candidate; it needs the user's recorded OK "
                                    "('record-cost-gate') or stays 'not tested (cost)'")
                if fit_seconds is None:
                    problems.append(f"{name}: an augmented fit rebuilds its graphs every epoch; pass the measured "
                                    "--fit-seconds")
        elif round_name == "D-combo":
            family = name.split("__")[0]
            wanted = required_combinations(gate).get(family)
            if wanted != recipe_of(name):
                problems.append(f"{name}: not the combination the assessment of every completed round requires "
                                f"for {family} ({wanted or 'none'}); 'assess --round D-B' decides it")
    return problems


def step_e_units(state: dict[str, Any], gate: dict[str, Any] | None) -> tuple[str, ...]:
    return e_neutral_units() + e_improvement_units(final_recipes(gate) or {})


def step_e_blockers(names: list[str], state: dict[str, Any], *, readiness: dict[str, Any] | None,
                    runner: dict[str, str], gate: dict[str, Any] | None,
                    label: dict[str, Any] | None) -> list[str]:
    problems = readiness_blockers(state, readiness, runner)
    if label is None or label.get("final_test_label") not in FINAL_TEST_LABELS:
        problems.append("step E waits for the user's final-test label decision ('record-e-gate' after it is logged)")
    problems += gate_assessment_blockers(gate, state, "step E")
    if gate is not None and final_recipes(gate) is None:
        problems.append("step E: step D is not final in the latest assessment (rounds or a combination still open)")
    return problems


def build_step(key: str, campaign: Path = CAMPAIGN, *, fit_seconds: float | None = None,
               runner: dict[str, str] | None = None) -> c.Step:
    """The step C launcher logic with the units, gates and folders of step D or E."""
    require(key in ("D", "E"), "step must be D or E")
    evidence = campaign / f"step_{key.lower()}_evidence"
    readiness = latest_readiness(campaign)
    runner = local_runner_sha256() if runner is None else runner
    d_gate = latest_assessment(campaign / "step_d_evidence", ("D-A", "D-R", "D-B"))
    if key == "D":
        cost_gate = read_record(evidence / "cost_gate_approval.json")
        return c.Step(
            key="d", label="step D", outside="see 'units'; step E has its own gate",
            second_failure="the candidate 'did not pass the one-fold screen' (plan step D); no further rerun",
            done_hint="every listed step D unit has run: 'assess --round ...', then 'evidence' and vm-stop",
            units=lambda state: step_d_units(state, d_gate),
            batch_blockers=lambda names, state: step_d_blockers(
                names, state, readiness=readiness, runner=runner, gate=d_gate, cost_gate=cost_gate,
                fit_seconds=fit_seconds),
            code=REMOTE_CODE, evidence=evidence, evidence_patterns=EVIDENCE_PATTERNS_D)
    label = read_record(evidence / "step_e_gate.json")
    return c.Step(
        key="e", label="step E", outside="see 'units --step E'",
        second_failure="step E stops for diagnosis and the user's decision",
        done_hint="every step E unit has run: 'assess', then 'evidence' and vm-stop",
        units=lambda state: step_e_units(state, d_gate),
        batch_blockers=lambda names, state: step_e_blockers(names, state, readiness=readiness, runner=runner,
                                                            gate=d_gate, label=label),
        code=REMOTE_CODE, evidence=evidence, evidence_patterns=EVIDENCE_PATTERNS_E)


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def cmd_units(key: str) -> dict[str, Any]:
    forecasts = {"warm": c.FIT_SECONDS, "cold_cache_extra": c.COLD_CACHE_SECONDS,
                 "note": "site-geometry recipes build their own graph caches (cold once, with and without ESMC); "
                         "cost-gated augmentations need an explicit --fit-seconds"}
    if key == "D":
        rounds = {name: list(round_units(name)) for name in ROUND_ORDER}
        return {"step": "D", "rounds": rounds, "counts": {name: len(units) for name, units in rounds.items()},
                "combination": "at most one recipe per family, both seeds, named by 'assess --round D-B'",
                "cost_gated": list(COST_GATED), "forecast_seconds": forecasts}
    return {"step": "E", "neutral": list(e_neutral_units()),
            "improvement": "four folds per improved family, named by the final step D assessment",
            "forecast_seconds": forecasts}


def cmd_gates(key: str, campaign: Path = CAMPAIGN) -> dict[str, Any]:
    """Offline view of the workstation gate records (the VM extension record is checked by 'status' and 'launch')."""
    readiness = latest_readiness(campaign)
    runner = local_runner_sha256()
    gate = latest_assessment(campaign / "step_d_evidence", ("D-A", "D-R", "D-B"))
    out = {"readiness_report": None if readiness is None else readiness["path"],
           "readiness_ready": None if readiness is None else readiness.get("ready"),
           "readiness_matches_worktree_runner": None if readiness is None else readiness.get("runner_sha256") == runner,
           "latest_step_d_assessment": None if gate is None else {
               "path": gate["path"], "kind": gate.get("kind"), "status": gate["result"].get("status"),
               "passed": {name: item.get("passed") for name, item in gate["result"].get("rounds", {}).items()}},
           "required_combinations": required_combinations(gate), "final_recipes": final_recipes(gate)}
    if key == "D":
        out["cost_gate_approval"] = read_record(campaign / "step_d_evidence" / "cost_gate_approval.json")
    else:
        out["final_test_label"] = read_record(campaign / "step_e_evidence" / "step_e_gate.json")
    return out


def cmd_push_bundle(remote: b.Remote, bundle_dir: Path) -> dict[str, Any]:
    """Apply the extension bundle into its own new code directory; the first one and the fold set stay as they are."""
    import tarfile

    manifest_path = bundle_dir / "v3_bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("parent_bundle") is not None, "Not an extension bundle: build it with --parent-manifest")
    require(manifest["runner_sha256"] == local_runner_sha256(),
            "The bundle's runner files differ from this worktree's; rebuild the bundle from the current commit")
    with tarfile.open(bundle_dir / manifest["code"]["path"], "r:gz") as tar:
        script = tar.extractfile(f"{manifest['code_root']}/pmm_v3_bundle.py").read()
    staged = bundle_dir / "pmm_v3_bundle.py"
    if not staged.exists():
        staged.write_bytes(script)
    require(staged.read_bytes() == script, "pmm_v3_bundle.py next to the bundle differs from the bundled copy")
    remote_dir = f"{b.REMOTE_BUNDLES}/{manifest['git_commit'][:12]}"
    remote.run(f"mkdir -p {shlex.quote(remote_dir)} && test ! -e {shlex.quote(REMOTE_CODE)}")
    for name in ("v3_bundle_manifest.json", manifest["code"]["path"], manifest["folds"]["path"], "pmm_v3_bundle.py"):
        remote.transport.put(bundle_dir / name, f"{remote_dir}/{name}")
    applied = remote.run(f"cd {shlex.quote(remote_dir)} && {b.REMOTE_PY} pmm_v3_bundle.py apply --bundle-dir . "
                         f"--code-dest {REMOTE_CODE} --folds-parent {b.REMOTE_FOLDS_PARENT} "
                         "--reuse-existing-folds").stdout
    return {"remote_bundle": remote_dir, "apply": json.loads(applied)}


def cmd_extend(remote: b.Remote, campaign: Path = CAMPAIGN) -> dict[str, Any]:
    """Record extension 1 on the VM (CPU, once) and keep a checked workstation copy of the record."""
    readiness, runner = latest_readiness(campaign), local_runner_sha256()
    problems = [p for p in readiness_blockers({"extension": {"id": EXTENSION_ID, "runner_sha256": runner}},
                                              readiness, runner)]
    require(not problems, "The extension is not recorded:\n  " + "\n  ".join(problems))
    out = remote.run(f"cd {shlex.quote(REMOTE_CODE)} && {b.REMOTE_PY} run_pmm_v3_campaign.py --action extend "
                     f"--campaign-dir {b.REMOTE_V3}").stdout
    printed = json.loads(out)
    require(printed["runner_sha256"] == runner, "The VM recorded other runner files than this worktree's")
    data = remote.transport.read_bytes(f"{b.REMOTE_V3}/campaign_extension.json")
    require(data is not None and hashlib.sha256(data).hexdigest() == printed["extension_sha256"],
            "the copied extension record differs from the VM file")
    target = campaign / "step_d_evidence" / "extension" / "campaign_extension.json"
    target.parent.mkdir(parents=True, exist_ok=False)
    target.write_bytes(data)
    return {**printed, "copy": str(target)}


def cmd_status(remote: b.Remote, step: c.Step, key: str, campaign: Path = CAMPAIGN) -> dict[str, Any]:
    status = c.cmd_status(remote, step)
    state = c.vm_state(remote, step=step)
    status["readiness_blockers"] = readiness_blockers(state, latest_readiness(campaign), local_runner_sha256())
    status["gates"] = cmd_gates(key, campaign)
    if key == "D":
        started = started_units(state)
        status["rounds"] = {name: {"units": len(round_units(name)),
                                   "completed": sum(state["statuses"].get(u, {}).get("status") == "completed"
                                                    for u in round_units(name)),
                                   "not_started": sum(u not in started for u in round_units(name))}
                            for name in ROUND_ORDER}
    return status


def assess_args(key: str, *, round_name: str | None, not_tested: list[str], campaign: Path = CAMPAIGN) -> tuple[str, ...]:
    if key == "D":
        require(round_name in ROUND_ORDER, f"assess needs --round, one of {ROUND_ORDER}")
        require(set(not_tested) <= set(COST_GATED), f"only {COST_GATED} can be recorded as not tested (cost)")
        require(not not_tested or round_name == "D-B", "--not-tested-cost applies to Round B only")
        return ("--step", round_name, *(("--not-tested-cost", *not_tested) if not_tested else ()))
    finals = final_recipes(latest_assessment(campaign / "step_d_evidence", ("D-A", "D-R", "D-B")))
    require(finals is not None, "step E is assessed with the final step D recipes; step D is not final yet")
    args = ["--step", "E"]
    for family, recipe in sorted(finals.items()):
        if recipe != "baseline":
            args += ["--final-recipe", f"{family}={recipe}"]
    return tuple(args)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("units", "gates", "session", "push-bundle", "extend", "status", "launch",
                                            "wait", "pull", "recover-lane", "archive-failed", "assess",
                                            "record-cost-gate", "record-e-gate", "evidence"))
    parser.add_argument("items", nargs="*", help="launch: UNIT@LANE ...; wait: optional UNIT ...; archive-failed: "
                                                 "UNIT; recover-lane: K")
    parser.add_argument("--step", choices=("D", "E"), default="D")
    parser.add_argument("--fit-seconds", type=float, help="launch: admission forecast for every unit")
    parser.add_argument("--no-wait", action="store_true", help="launch: return after the launch")
    parser.add_argument("--any", action="store_true", help="wait: return after the first unit ends")
    parser.add_argument("--lanes", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--allow-running", action="store_true", help="evidence: emergency copy while units run")
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--round", choices=ROUND_ORDER, help="assess (step D): the last round to assess")
    parser.add_argument("--not-tested-cost", nargs="*", default=[], help="assess D-B: cost-gated candidates never run")
    parser.add_argument("--recipes", nargs="+", choices=COST_GATED, help="record-cost-gate: approved candidates")
    parser.add_argument("--label", choices=FINAL_TEST_LABELS, help="record-e-gate: the user's final-test label")
    parser.add_argument("--log-entry", help="record-*: the dated campaign log entry of the user's decision")
    args = parser.parse_args(argv)
    if args.command in ("units", "gates"):
        print(json.dumps(cmd_units(args.step) if args.command == "units" else cmd_gates(args.step), indent=2))
        return 0
    if args.command == "record-cost-gate":
        require(bool(args.recipes), "record-cost-gate needs --recipes")
        print(json.dumps(write_record_once(CAMPAIGN / "step_d_evidence" / "cost_gate_approval.json",
                                           {"recipes": sorted(args.recipes), "log_entry": args.log_entry}), indent=2))
        return 0
    if args.command == "record-e-gate":
        require(args.label is not None, "record-e-gate needs --label")
        print(json.dumps(write_record_once(CAMPAIGN / "step_e_evidence" / "step_e_gate.json",
                                           {"final_test_label": args.label, "log_entry": args.log_entry}), indent=2))
        return 0
    session = b.read_session()
    if args.command == "session":
        print(json.dumps(session, indent=2, sort_keys=True))
        return 0
    remote = b.Remote(b.check_ssh_config())
    step = build_step(args.step, fit_seconds=args.fit_seconds)
    if args.command == "launch":
        require(bool(args.items), "launch needs UNIT@LANE ...")
    if args.command == "archive-failed":
        require(len(args.items) == 1, "archive-failed needs one unit")
    if args.command == "recover-lane":
        require(len(args.items) == 1 and args.items[0].isdigit(), "recover-lane needs one lane number")
    if args.command == "push-bundle":
        require(args.bundle_dir is not None, "push-bundle needs --bundle-dir")
    actions = {
        "push-bundle": lambda: cmd_push_bundle(remote, args.bundle_dir),
        "extend": lambda: cmd_extend(remote),
        "status": lambda: cmd_status(remote, step, args.step),
        "launch": lambda: c.cmd_launch(remote, session, [c.parse_request(item) for item in args.items],
                                       fit_seconds=args.fit_seconds, wait=not args.no_wait, step=step),
        "wait": lambda: c.cmd_wait(remote, session, names=args.items or None, any_unit=args.any, step=step),
        "pull": lambda: b.cmd_pull(remote, args.lanes),
        "recover-lane": lambda: c.cmd_recover_lane(remote, session, int(args.items[0]), step),
        "archive-failed": lambda: c.cmd_archive_failed(remote, args.items[0], step),
        "assess": lambda: c.cmd_assess(remote, step=step, assess_args=assess_args(
            args.step, round_name=args.round, not_tested=args.not_tested_cost)),
        "evidence": lambda: c.cmd_evidence(remote, allow_running=args.allow_running, step=step),
    }
    print(json.dumps(actions[args.command](), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (LauncherError, b.host_pull.HostPullError) as exc:
        print(f"step-D/E launcher refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
