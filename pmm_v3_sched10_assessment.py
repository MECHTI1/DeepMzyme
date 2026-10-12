#!/usr/bin/env python3
"""Validation-only assessment of the v3 10-epoch schedule comparison under its frozen specification (workstation).

Rules: docs/campaigns/pmm_ion_metal_v3/sched10_assessment_spec.json (handoff section 11, log v3-039; pinned by
pmm_v3_sched10.FROZEN_SPEC_SHA256). The A4 and E2 specifications, assessors and assessments are only read.

Data path as for step E2 (pmm_v3_e2_assessment): the host-pulled copies on Data1, never the VM. A temporary read-only
view of the campaign root links the newest evidence copy's root files and status records to the durable lane copies;
every unit used must match an acknowledged host-pull receipt file by file. The four 50-epoch controls are loaded by the
frozen A4 loader and must equal the evidence the launch manifest binds; the four 10-epoch fits are loaded with the same
artifact checks (recorded identity, terminal checkpoint at epoch 10, reconciled predictions, independent replay) and
must carry the bound ``schedule_extension`` in their identity. As a self-check the controls' fold values must equal
the final E2 assessment's.

One comparison (S1, candidate minus control, folds 1-4, seed 42); the adoption rule (a)-(d) of the specification;
Bonferroni over 18 as sensitivity only; the development fold 0 reported separately; a machine-readable schedule
decision for step F. The tool never starts a fit and never opens held-out data.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import pmm_v3_assessment as pa  # noqa: E402
import pmm_v3_campaign as v3  # noqa: E402
import pmm_v3_e2_assessment as e2a  # noqa: E402
import pmm_v3_sched10 as s10  # noqa: E402
import run_pmm_v3_sched10 as vm  # noqa: E402

require = e2a.require
sha256 = e2a.sha256
COHORT_SOURCE = e2a.COHORT_SOURCE
PARITY_TOLERANCE = 1e-9
# The values this code implements; the frozen specification must state exactly these.
TESTED = {"confidence": 0.95, "bootstrap_resamples": 10000, "bootstrap_seed": 42, "bootstrap_interval": "percentile",
          "t_degrees_of_freedom": 3, "candidate_epochs": 10, "control_epochs": 50}
TESTED_RULE = {"c_min_mean_gain": 0.002, "c_min_mean_gain_strict": True, "d_recall_drop_limit": 0.03,
               "d_no_missing_or_zero_common4_recall_in_any_fold": True}
TESTED_SENSITIVITY = {"method": "bonferroni", "n_contrasts": 18}


def load_spec() -> dict[str, Any]:
    try:
        spec = s10.load_spec()
    except s10.LauncherError as exc:
        raise ValueError(str(exc)) from None
    differing = sorted(key for key, value in TESTED.items() if spec.get(key) != value)
    differing += sorted(key for key, value in TESTED_RULE.items() if spec["adoption_rule"].get(key) != value)
    differing += sorted(key for key, value in TESTED_SENSITIVITY.items() if spec["sensitivity"].get(key) != value)
    require(not differing, f"The specification differs from the tested rules in {differing}")
    return spec


def interval_spec(spec: dict[str, Any], a4: dict[str, Any]) -> dict[str, Any]:
    """The A4 interval settings; the comparison keeps them unchanged (recall limit = the A4 limit)."""
    require(all(spec[key] == a4[key] for key in ("confidence", "bootstrap_resamples", "bootstrap_seed"))
            and spec["adoption_rule"]["d_recall_drop_limit"] == a4["recall_drop_limit"],
            "The comparison's interval settings differ from A4's")
    return {key: a4[key] for key in e2a.INTERVAL_KEYS}


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

def schedule_binding(manifest: dict[str, Any]) -> dict[str, Any]:
    return {"id": s10.SCHEDULE_ID, "runner_file": s10.VM_RUNNER, "runner_sha256": manifest["vm_runner"]["sha256"],
            "epochs": vm.EPOCHS, "control_epochs": int(v3.PROFILE["epochs"])}


def not_completed(name: str, record: dict[str, Any] | None, reruns: int) -> dict[str, Any]:
    """pa.load_unit's record for a fit that did not complete."""
    status = "missing" if record is None else record["status"]
    final = status != "missing" and reruns >= v3.MAX_RERUNS
    return {"name": name, "status": status, "archived_attempts": reruns, "final_failure": final,
            "note": ("failed after its one rerun" if final else "rerun once (archive-failed, then run)"
                     if status != "missing" else "not run")}


def verified_rows(run_dir: Path, recorded: dict[str, Any], membership, *, fold: int, name: str) -> dict[str, Any]:
    """The frozen artifact checks: receipt, independent replay, prediction rows, BA reconciliation."""
    receipt = v3.verify_completed_unit(run_dir, recorded, v3.resolve_recipe(vm.RECIPE))
    v3.verify_independent_replay(run_dir, receipt)
    rows = pa._read_rows(run_dir / receipt["validation_predictions"]["path"])
    pa.validate_rows(rows, membership, fold=fold, target=vm.TARGET, seed=vm.SEED,
                     checkpoint_sha256=receipt["selected_checkpoint_sha256"], selected_epoch=receipt["selected_epoch"])
    metrics = pa.prediction_metrics(rows, vm.TARGET)
    for key, value in (("val_metal_collapsed4_balanced_acc", metrics["common4"]["balanced_accuracy"]),
                       ("val_metal_balanced_acc", metrics["native"]["balanced_accuracy"])):
        require(abs(float(receipt["metrics"][key]) - value) <= pa.RECONCILIATION_TOLERANCE,
                f"{name}: {key} from predictions differs from the receipt")
    return {"rows": rows, "metrics": metrics, "selected_epoch": receipt["selected_epoch"],
            "selected_checkpoint": receipt["selected_checkpoint"],
            "evidence": {"checkpoint_sha256": receipt["selected_checkpoint_sha256"],
                         "predictions_sha256": receipt["validation_predictions"]["sha256"]}}


def load_comparison_fit(view: v3.V3Paths, fold: int, statuses, membership, contract, *,
                        binding: dict[str, Any]) -> dict[str, Any]:
    """A verified 10-epoch fit, or a record saying why it does not count."""
    name, unit = vm.run_name(fold), vm.control_unit(fold)
    record = statuses.get(name)
    if record is None or record["status"] != "completed":
        return not_completed(name, record, len(v3.archived_attempts(view, name)))
    recorded = record["identity"]
    expected = v3.recorded_identity_expectation(view, contract, name, recorded,
                                                pa.expected_identity(view, unit, epochs=vm.EPOCHS))
    expected = {**expected, "schedule_extension": binding}
    differing = sorted(k for k in set(expected) | set(recorded) if expected.get(k) != recorded.get(k))
    require(not differing, f"{name}: recorded identity differs from the specified 10-epoch fit in {differing}")
    run_dir = Path(record["status_path"]).parent / "runs" / name
    checked = verified_rows(run_dir, recorded, membership, fold=fold, name=name)
    require(checked["selected_epoch"] == vm.EPOCHS, f"{name}: the terminal checkpoint is not epoch {vm.EPOCHS}")
    return {"name": name, "run_name": name, "status": "completed", "unit": unit, **checked}


def load_development_run(view: v3.V3Paths, run_name: str, statuses, membership, *, bound: dict[str, Any],
                         epochs: int) -> dict[str, Any]:
    """A fold-0 development run bound by hash (step-B probes predate extension 1; their identity is not re-derived)."""
    record = statuses.get(run_name)
    require(record is not None and record["status"] == "completed", f"{run_name}: not completed")
    run_dir = Path(record["status_path"]).parent / "runs" / run_name
    checked = verified_rows(run_dir, record["identity"], membership, fold=0, name=run_name)
    require(checked["selected_epoch"] == epochs, f"{run_name}: not a terminal epoch-{epochs} checkpoint")
    require(all(checked["evidence"][key] == bound[key] for key in ("checkpoint_sha256", "predictions_sha256")),
            f"{run_name}: differs from the evidence the launch manifest binds")
    return {"name": run_name, "run_name": run_name, "status": "completed", **checked}


def collect(view: v3.V3Paths, manifest: dict[str, Any], *, data: Path, evidence_dir: Path) -> dict[str, Any]:
    contract, membership = pa.read_contract(view)
    statuses = v3.completed_units(view)
    binding = schedule_binding(manifest)
    out: dict[str, Any] = {}
    jobs = [(name, lambda name=name: pa.load_unit(view, v3.Unit.parse(name), statuses, membership, contract,
                                                  epochs=int(v3.PROFILE["epochs"])), manifest["controls"][name])
            for name in manifest["controls"]]
    jobs += [(vm.run_name(fold), lambda fold=fold: load_comparison_fit(view, fold, statuses, membership, contract,
                                                                       binding=binding), None) for fold in vm.FOLDS]
    for name, load, bound in jobs:
        try:
            record = load()
            if record["status"] == "completed":
                run_name = record["run_name"]
                record["durable"] = e2a.verify_durable_unit(data, e2a.lane_of(statuses[run_name]), run_name,
                                                            evidence_dir=evidence_dir)
                if bound is not None and any(record["evidence"][k] != bound[k]
                                             for k in ("checkpoint_sha256", "predictions_sha256")):
                    raise ValueError("differs from the evidence the launch manifest binds")
        except (ValueError, KeyError, OSError) as exc:
            record = {"name": name, "status": "invalid evidence", "note": f"{type(exc).__name__}: {exc}",
                      "final_failure": False}
        out[name] = record
    dev = manifest["development_fold"]
    development = {}
    for role, epochs in (("candidate", vm.EPOCHS), ("control", int(v3.PROFILE["epochs"]))):
        try:
            development[role] = load_development_run(view, dev[role]["run"], statuses, membership, bound=dev[role],
                                                     epochs=epochs)
        except (ValueError, KeyError, OSError) as exc:
            development[role] = {"name": dev[role]["run"], "status": "invalid evidence",
                                 "note": f"{type(exc).__name__}: {exc}"}
    return {"units": out, "development": development}


# ---------------------------------------------------------------------------
# The comparison and the adoption rule
# ---------------------------------------------------------------------------

def per_fold_recall_failures(challenger: dict[int, Any]) -> list[str]:
    """Adoption rule (d): no common-four recall missing or zero in any fold of the candidate."""
    out = []
    for fold, unit in sorted(challenger.items()):
        for label in pa.COMMON4:
            value = unit["metrics"]["common4"]["recall"][label]
            if value is None:
                out.append(f"(d) fold {fold}: {label} recall missing")
            elif value == 0.0:
                out.append(f"(d) fold {fold}: {label} recall is zero")
    return out


def breakdown(control: dict[int, Any], challenger: dict[int, Any], membership) -> dict[str, Any]:
    """Descriptive, not gating (log v3-038 style): per fold, the examples each run gets right and the other wrong,
    in BA points, the largest net protein group, and the mean confidence and NLL of the true class."""
    per_fold = {}
    for fold in sorted(control):
        a = {row["source_uid"]: row for row in control[fold]["rows"]}
        b = {row["source_uid"]: row for row in challenger[fold]["rows"]}
        require(set(a) == set(b), f"fold {fold}: the two runs cover different examples")
        n = defaultdict(int)
        for row in a.values():
            n[int(row["y_common4"])] += 1
        corr = rev = 0.0
        counts = [0, 0]
        groups: dict[str, float] = defaultdict(float)
        for uid, row in a.items():
            y = int(row["y_common4"])
            w = 100.0 / (4 * n[y])
            right_a, right_b = int(row["pred_common4"]) == y, int(b[uid]["pred_common4"]) == y
            if right_b and not right_a:
                corr += w
                counts[0] += 1
                groups[membership[uid]["group_id"]] += w
            elif right_a and not right_b:
                rev += w
                counts[1] += 1
                groups[membership[uid]["group_id"]] -= w
        top = max(groups.items(), key=lambda kv: kv[1]) if groups else (None, 0.0)

        def confidence(rows):
            probs = [[float(r[f"p_common4_{c.replace(' ', '_')}"]) for c in pa.COMMON4] for r in rows]
            ys = [int(r["y_common4"]) for r in rows]
            return (float(np.mean([max(p) for p in probs])),
                    float(np.mean([-math.log(max(p[y], 1e-12)) for p, y in zip(probs, ys)])))

        per_fold[fold] = {"candidate_right_control_wrong": counts[0], "candidate_right_control_wrong_points": corr,
                          "control_right_candidate_wrong": counts[1], "control_right_candidate_wrong_points": rev,
                          "net_points": corr - rev, "top_net_group": top[0], "top_net_group_points": top[1],
                          "net_points_without_top_group": corr - rev - max(top[1], 0.0),
                          "mean_max_probability": {"control": confidence(control[fold]["rows"])[0],
                                                   "candidate": confidence(challenger[fold]["rows"])[0]},
                          "mean_true_class_nll": {"control": confidence(control[fold]["rows"])[1],
                                                  "candidate": confidence(challenger[fold]["rows"])[1]}}
    return {"per_fold": per_fold, "note": "descriptive only; never part of the adoption rule"}


def assess(collected: dict[str, Any], manifest: dict[str, Any], spec: dict[str, Any], istats: dict[str, Any],
           membership) -> dict[str, Any]:
    units = collected["units"]
    control = {fold: units[f"{spec['selection_cell']}__fold{fold}__seed{spec['seed']}"] for fold in vm.FOLDS}
    challenger = {fold: units[vm.run_name(fold)] for fold in vm.FOLDS}
    result = pa.compare(control, challenger, istats, challenger_target=vm.TARGET, n_contrasts=1)
    rule = spec["adoption_rule"]
    final_failures = sorted(unit["name"] for unit in challenger.values() if unit.get("final_failure"))
    missing = pa.incomplete(control) + pa.incomplete(challenger)
    reasons: list[str] = []
    comparison: dict[str, Any] = {"id": spec["comparison"]["id"], **result}
    if final_failures:
        status = f"blocked: {final_failures} failed after their one rerun; no decision (stopping rule)"
    elif missing or result["status"] != "complete":
        status = f"incomplete: {missing}"
        reasons.append(f"(a) incomplete evidence: {missing}")
    else:
        status = "complete"
        comparison["sd_difference"] = float(np.std(result["fold_differences"], ddof=1))
        comparison["sensitivity_bonferroni"] = pa.paired_intervals(
            result["fold_differences"], istats, n_contrasts=spec["sensitivity"]["n_contrasts"])["adjusted"]
        lower = result["unadjusted"]
        if not (lower["bootstrap"][0] > 0 and lower["t"][0] > 0):
            reasons.append(f"(b) not interval-supported: bootstrap lower bound {100 * lower['bootstrap'][0]:+.2f}, "
                           f"t lower bound {100 * lower['t'][0]:+.2f} points")
        gain = result["mean_difference"]
        if not (gain > rule["c_min_mean_gain"] if rule["c_min_mean_gain_strict"] else gain >= rule["c_min_mean_gain"]):
            reasons.append(f"(c) mean gain {100 * gain:+.2f} points, not above {100 * rule['c_min_mean_gain']:.2f}")
        reasons += [f"(d) {failure}" for failure in e2a.gate_failures(result, challenger, spec["selection_cell"],
                                                                      spec["comparison"]["id"],
                                                                      rule["d_recall_drop_limit"])]
        reasons += per_fold_recall_failures(challenger)
        comparison["fold_common4_ba"] = {
            role: {str(k): folds[k]["metrics"]["common4"]["balanced_accuracy"] for k in sorted(folds)}
            for role, folds in (("candidate", challenger), ("control", control))}
        comparison["summaries"] = {"candidate": pa.summarize(challenger, vm.TARGET),
                                   "control": pa.summarize(control, vm.TARGET)}
        comparison["breakdown"] = breakdown(control, challenger, membership)
    adopted = status == "complete" and not reasons
    dev = collected["development"]
    development = {"note": spec["development_fold"]["role"]}
    if all(item.get("status") == "completed" for item in dev.values()):
        development.update({role: {"run": item["run_name"], "common4_ba": item["metrics"]["common4"]["balanced_accuracy"],
                                   "common4_recall": item["metrics"]["common4"]["recall"]}
                            for role, item in dev.items()})
        development["difference"] = development["candidate"]["common4_ba"] - development["control"]["common4_ba"]
    else:
        development["problems"] = {role: item.get("note") for role, item in dev.items() if item.get("status") != "completed"}
    if status == "complete":
        outcome = "adopted" if adopted else "not_adopted"
    elif status.startswith("blocked"):
        outcome = "blocked"
    else:
        outcome = None  # incomplete: no decision yet
    decision = None if outcome is None else {
        "selection": spec["selection_cell"], "outcome": outcome, "adopted": adopted,
        "epochs": spec["candidate_epochs"] if adopted else spec["control_epochs"],
        "meaning": spec["outcomes"][outcome], "reasons": reasons,
        "checkpoint_rule": spec["checkpoint_rule"], "lr_schedule": spec["lr_schedule"], "seed": spec["seed"],
        "calibration": None, "ensemble": False}
    return {"step": "S10", "status": status, "comparison": comparison, "reasons": reasons,
            "final_failures": final_failures, "development_fold0": development, "schedule_decision": decision,
            "held_out_test_accessed": False}


def control_parity(collected: dict[str, Any], manifest: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """Self-check: the four controls' fold values equal those of the final E2 assessment (comparison C10's comparator
    = the original selection on folds 1-4), and their evidence equals its unit evidence."""
    e2_assessment = e2a.e2.read_json(Path(manifest["e2_assessment"]["path"]))
    reference = None
    for comparison in e2_assessment["result"]["comparisons"].values():
        if comparison["comparator"] == spec["selection_cell"] and comparison["role"] == "original selection":
            reference = comparison["fold_common4_ba"]["comparator"]
            break
    require(reference is not None, "The E2 assessment has no comparison against the original selection")
    problems = []
    for fold in vm.FOLDS:
        name = f"{spec['selection_cell']}__fold{fold}__seed{spec['seed']}"
        unit = collected["units"][name]
        if unit.get("status") != "completed":
            problems.append(f"{name}: {unit.get('status')}")
            continue
        value = unit["metrics"]["common4"]["balanced_accuracy"]
        if not math.isclose(value, reference[str(fold)], rel_tol=0.0, abs_tol=PARITY_TOLERANCE):
            problems.append(f"{name}: {value!r} != E2 {reference[str(fold)]!r}")
        if unit["evidence"] != {k: e2_assessment["unit_evidence"][name][k] for k in unit["evidence"]}:
            problems.append(f"{name}: evidence differs from the E2 assessment")
    return {"equal": not problems, "problems": problems, "tolerance": PARITY_TOLERANCE}


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def run(*, data: Path, manifest_path: Path, evidence_dir: Path, out_dir: Path, cohort: Path = COHORT_SOURCE,
        purpose: str = "assessment") -> Path:
    spec = load_spec()
    a4 = pa.load_spec()
    istats = interval_spec(spec, a4)
    manifest = e2a.e2.read_json(manifest_path)
    problems = s10.verify_manifest(manifest)
    require(not problems, "The launch manifest does not verify: " + "; ".join(problems))
    evidence = e2a.check_evidence_copy(evidence_dir)
    with tempfile.TemporaryDirectory(prefix="v3-s10-view-") as scratch:
        view_dir = Path(scratch) / "campaign"
        view = e2a.build_view(view_dir, data=data, evidence_dir=evidence_dir, manifest=manifest, cohort=cohort)
        paths = v3.V3Paths(view_dir)
        pa.check_spec_against_campaign(a4, e2a.e2.read_json(paths.manifest))
        collected = collect(paths, manifest, data=data, evidence_dir=evidence_dir)
        _, membership = pa.read_contract(paths)
        result = assess(collected, manifest, spec, istats, membership)
    parity = control_parity(collected, manifest, spec)
    if not parity["equal"] and result["schedule_decision"] is not None:
        result["status"] = "blocked: the controls do not reproduce the final E2 assessment from these copies"
        result["schedule_decision"] = None
    payload = {
        "campaign_id": v3.CAMPAIGN_ID, "kind": "S10", "purpose": purpose,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "assessor_sha256": sha256(Path(__file__)), "launcher_sha256": sha256(ROOT / "pmm_v3_sched10.py"),
        "vm_runner_sha256": sha256(ROOT / s10.VM_RUNNER), "a4_assessor_sha256": sha256(ROOT / "pmm_v3_assessment.py"),
        "spec_sha256": s10.FROZEN_SPEC_SHA256, "a4_spec_sha256": manifest["a4_spec_sha256"],
        "manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        "e2_assessment": manifest["e2_assessment"], "evidence_copy": evidence, "view": view, "control_parity": parity,
        "unit_evidence": {name: record.get("evidence") for name, record in sorted(collected["units"].items())},
        "unit_status": {name: {"status": record.get("status"), "note": record.get("note"),
                               "durable": record.get("durable")} for name, record in sorted(collected["units"].items())},
        "interpretation": spec["interpretation"],
        "result": pa._jsonable(result), "held_out_test_accessed": False,
    }
    if result["schedule_decision"] is not None:
        payload["result"]["schedule_decision"].update({
            "spec_sha256": s10.FROZEN_SPEC_SHA256, "manifest_sha256": payload["manifest"]["sha256"],
            "evidence": payload["unit_evidence"], "held_out_test_accessed": False})
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "assessment.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_dir / "assessment.json"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=s10.CAMPAIGN)
    parser.add_argument("--manifest", type=Path, help="default: the recorded launch manifest")
    parser.add_argument("--evidence", type=Path, help="evidence copy (default: the newest comparison copy)")
    parser.add_argument("--cohort", type=Path, default=COHORT_SOURCE)
    parser.add_argument("--out-dir", type=Path, help="default: step_s10_evidence/assessments/S10_<UTC>")
    parser.add_argument("--readiness-dry-run", action="store_true",
                        help="CPU readiness check before any fit: allows another step's evidence copy and needs "
                             "--out-dir outside the comparison's assessments")
    args = parser.parse_args(argv)
    data = args.data_root
    evidence_root = data / s10.EVIDENCE_DIR
    manifest_path = args.manifest or evidence_root / s10.MANIFEST_NAME
    assessments = evidence_root / "assessments"
    if args.readiness_dry_run:
        require(args.evidence is not None and args.out_dir is not None
                and assessments.resolve() not in args.out_dir.resolve().parents,
                "A readiness dry run names --evidence and writes outside the comparison's assessments (--out-dir)")
        evidence_dir, purpose = args.evidence, "readiness dry run (not an assessment)"
    else:
        evidence_dir = args.evidence or e2a.newest_evidence(evidence_root)
        require(evidence_dir.resolve().parent == evidence_root.resolve(),
                "An assessment reads a comparison evidence copy ('evidence' after the session)")
        purpose = "assessment"
    out_dir = args.out_dir or assessments / f"S10_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    path = run(data=data, manifest_path=manifest_path, evidence_dir=evidence_dir, out_dir=out_dir,
               cohort=args.cohort, purpose=purpose)
    written = json.loads(path.read_text())
    print(json.dumps({"assessment": str(path), "sha256": sha256(path), "status": written["result"]["status"],
                      "schedule_decision": written["result"]["schedule_decision"],
                      "control_parity_equal": written["control_parity"]["equal"]}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"v3 10-epoch comparison assessment refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
