#!/usr/bin/env python3
"""Validation-only assessment of the trunk-learning-rate comparison under its frozen specification (workstation).

Rules: docs/campaigns/pmm_trunk_lr_2026-10/trunklr_assessment_spec.json (pinned by pmm_trunklr.FROZEN_SPEC_SHA256).
The A4 and E2 specifications, assessors and assessments are only read.

Data path as for the v3 10-epoch comparison (pmm_v3_sched10_assessment): the local host-pulled copies, never the
VM. A temporary read-only view of the v3 execution root links the evidence copy's root files and status records to the
durable lane copies; every unit used must match an acknowledged host-pull receipt file by file. The five controls are
loaded by the frozen A4 loader (fold 0 resolves to the reused step-B run) and must equal the evidence the launch
manifest binds and the A4 fold values; the five candidate fits are loaded with the same artifact checks (recorded
identity rebuilt from the frozen builder, terminal checkpoint at epoch 50, reconciled predictions, independent replay)
and their optimizer groups are read from the selected checkpoints: the control's groups with the trunk group at
initial learning rate 1e-4.

One comparison (T1, candidate minus control, folds 0-4, seed 42) under the adoption rule (a)-(e) of the specification;
folds 1-4 reported as sensitivity (its mean is condition (e)). The tool never starts a fit and never opens held-out
data. A readiness dry run (before any candidate fit) exercises the control loading, parity and optimizer groups and the
configuration difference on the real frozen campaign files.
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
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import pmm_trunklr as tl  # noqa: E402
import pmm_v3_assessment as pa  # noqa: E402
import pmm_v3_campaign as v3  # noqa: E402
import pmm_v3_e2_assessment as e2a  # noqa: E402
import pmm_v3_sched10_assessment as s10a  # noqa: E402  (frozen artifact checks of the same cell)
import run_pmm_trunklr as vm  # noqa: E402

require = e2a.require
sha256 = e2a.sha256
COHORT_SOURCE = e2a.COHORT_SOURCE
PARITY_TOLERANCE = 1e-9
EVIDENCE_KEYS = ("checkpoint_sha256", "predictions_sha256")
# The values this code implements; the frozen specification must state exactly these.
TESTED = {"confidence": 0.95, "bootstrap_resamples": 10000, "bootstrap_seed": 42, "bootstrap_interval": "percentile",
          "t_degrees_of_freedom": 4, "epochs": 50}
TESTED_RULE = {"c_min_mean_gain": 0.002, "c_min_mean_gain_strict": True, "d_recall_drop_limit": 0.03,
               "d_no_missing_or_zero_common4_recall_in_any_fold": True, "e_folds_1_4_mean_gain_positive": True}


def load_spec() -> dict[str, Any]:
    try:
        spec = tl.load_spec()
    except tl.LauncherError as exc:
        raise ValueError(str(exc)) from None
    differing = sorted(key for key, value in TESTED.items() if spec.get(key) != value)
    differing += sorted(key for key, value in TESTED_RULE.items() if spec["adoption_rule"].get(key) != value)
    require(not differing, f"The specification differs from the tested rules in {differing}")
    require(len(spec["folds"]) - 1 == spec["t_degrees_of_freedom"], "t degrees of freedom are not folds - 1")
    return spec


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

def optimizer_groups(run_dir: Path, checkpoint_name: str) -> list[dict[str, Any]]:
    """Parameter-group sizes and initial learning rates / weight decays saved with the selected checkpoint."""
    import torch

    checkpoint = torch.load(run_dir / checkpoint_name, map_location="cpu", weights_only=False)
    return [{"n_params": len(group["params"]), "initial_lr": float(group["initial_lr"]),
             "weight_decay": float(group["weight_decay"])}
            for group in checkpoint["optimizer_state_dict"]["param_groups"]]


def expected_candidate_groups(control: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The control's two groups (GVP trunk first, then base) with the trunk group starting at the candidate rate."""
    require(len(control) == 2 and math.isclose(control[0]["initial_lr"], float(vm.control_gvp_lr()), rel_tol=1e-12)
            and math.isclose(control[1]["initial_lr"], 3e-5, rel_tol=1e-12),
            f"The control's optimizer groups are not the frozen trunk/base groups: {control}")
    return [{**control[0], "initial_lr": float(vm.CANDIDATE_GVP_LR)}, dict(control[1])]


def same_groups(found: list[dict[str, Any]], expected: list[dict[str, Any]]) -> bool:
    return len(found) == len(expected) and all(
        a["n_params"] == b["n_params"] and math.isclose(a["initial_lr"], b["initial_lr"], rel_tol=1e-12)
        and math.isclose(a["weight_decay"], b["weight_decay"], rel_tol=1e-12) for a, b in zip(found, expected))


def load_comparison_fit(view: v3.V3Paths, fold: int, statuses, membership, contract, *, amp: bool) -> dict[str, Any]:
    """A verified candidate fit, or a record saying why it does not count."""
    name, unit = vm.run_name(fold), vm.control_unit(fold)
    record = statuses.get(name)
    if record is None or record["status"] != "completed":
        return s10a.not_completed(name, record, len(v3.archived_attempts(view, name)))
    recorded = record["identity"]
    planned = vm.build(view, fold, python_bin="python", train_dir=view.root / "train", esm_dir=view.root / "esm",
                       device="cuda", lane=0, amp=amp)[2]
    expected = v3.recorded_identity_expectation(view, contract, name, recorded, planned)
    differing = sorted(k for k in set(expected) | set(recorded) if expected.get(k) != recorded.get(k))
    require(not differing, f"{name}: recorded identity differs from the specified candidate fit in {differing}")
    run_dir = Path(record["status_path"]).parent / "runs" / name
    checked = s10a.verified_rows(run_dir, recorded, membership, fold=fold, name=name)
    require(checked["selected_epoch"] == int(v3.PROFILE["epochs"]),
            f"{name}: the terminal checkpoint is not epoch {v3.PROFILE['epochs']}")
    return {"name": name, "run_name": name, "status": "completed", "unit": unit, **checked}


def collect(view: v3.V3Paths, manifest: dict[str, Any], *, data: Path, evidence_dir: Path, amp: bool) -> dict[str, Any]:
    contract, membership = pa.read_contract(view)
    statuses = v3.completed_units(view)
    out: dict[str, Any] = {}
    jobs = [(name, lambda name=name: pa.load_unit(view, v3.Unit.parse(name), statuses, membership, contract,
                                                  epochs=int(v3.PROFILE["epochs"])), manifest["controls"][name])
            for name in manifest["controls"]]
    jobs += [(vm.run_name(fold), lambda fold=fold: load_comparison_fit(view, fold, statuses, membership, contract,
                                                                       amp=amp), None) for fold in vm.FOLDS]
    for name, load, bound in jobs:
        try:
            record = load()
            if record["status"] == "completed":
                run_name = record["run_name"]
                record["durable"] = e2a.verify_durable_unit(data, e2a.lane_of(statuses[run_name]), run_name,
                                                            evidence_dir=evidence_dir)
                if bound is not None and any(record["evidence"][k] != bound[k] for k in EVIDENCE_KEYS):
                    raise ValueError("differs from the evidence the launch manifest binds")
                run_dir = Path(statuses[run_name]["status_path"]).parent / "runs" / run_name
                record["optimizer_groups"] = optimizer_groups(run_dir, record["selected_checkpoint"])
        except (ValueError, KeyError, OSError) as exc:
            record = {"name": name, "status": "invalid evidence", "note": f"{type(exc).__name__}: {exc}",
                      "final_failure": False}
        out[name] = record
    return {"units": out}


# ---------------------------------------------------------------------------
# The comparison and the adoption rule
# ---------------------------------------------------------------------------

def optimizer_failures(control: dict[int, Any], challenger: dict[int, Any]) -> list[str]:
    """Adoption rule (a), optimizer part: each candidate has its control's groups with the trunk group at 1e-4."""
    out = []
    for fold in sorted(challenger):
        try:
            expected = expected_candidate_groups(control[fold]["optimizer_groups"])
        except ValueError as exc:
            out.append(f"(a) fold {fold}: {exc}")
            continue
        if not same_groups(challenger[fold]["optimizer_groups"], expected):
            out.append(f"(a) fold {fold}: optimizer groups {challenger[fold]['optimizer_groups']} are not {expected}")
    return out


def assess(collected: dict[str, Any], manifest: dict[str, Any], spec: dict[str, Any], istats: dict[str, Any],
           membership) -> dict[str, Any]:
    units = collected["units"]
    folds = list(spec["folds"])
    control = {k: units[f"{spec['selection_cell']}__fold{k}__seed{spec['seed']}"] for k in folds}
    challenger = {k: units[vm.run_name(k)] for k in folds}
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
        reasons += optimizer_failures(control, challenger)
        comparison["sd_difference"] = float(np.std(result["fold_differences"], ddof=1))
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
        reasons += s10a.per_fold_recall_failures(challenger)
        sens_folds = list(spec["sensitivity"]["folds"])
        sensitivity = pa.compare({k: control[k] for k in sens_folds}, {k: challenger[k] for k in sens_folds}, istats,
                                 challenger_target=vm.TARGET, n_contrasts=1)
        comparison["sensitivity_folds_1_4"] = {**sensitivity, "role": spec["sensitivity"]["role"]}
        if not sensitivity["mean_difference"] > 0:
            reasons.append(f"(e) folds 1-4 mean gain {100 * sensitivity['mean_difference']:+.2f} points, not positive")
        comparison["fold_common4_ba"] = {
            role: {str(k): units_by_fold[k]["metrics"]["common4"]["balanced_accuracy"] for k in sorted(units_by_fold)}
            for role, units_by_fold in (("candidate", challenger), ("control", control))}
        comparison["optimizer_groups"] = {str(k): {"control": control[k]["optimizer_groups"],
                                                   "candidate": challenger[k]["optimizer_groups"]} for k in folds}
        comparison["summaries"] = {"candidate": pa.summarize(challenger, vm.TARGET),
                                   "control": pa.summarize(control, vm.TARGET)}
        comparison["breakdown"] = s10a.breakdown(control, challenger, membership)
    promising = status == "complete" and not reasons
    if status == "complete":
        outcome = "promising_candidate" if promising else "reference_retained"
    elif status.startswith("blocked"):
        outcome = "blocked"
    else:
        outcome = None  # incomplete: no decision yet
    decision = None if outcome is None else {
        "selection": spec["selection_cell"], "outcome": outcome, "promising_candidate": promising,
        "gvp_learning_rate": spec["candidate_gvp_learning_rate"] if promising else spec["control_gvp_learning_rate"],
        "meaning": spec["outcomes"][outcome], "reasons": reasons, "checkpoint_rule": spec["checkpoint_rule"],
        "lr_schedule": spec["lr_schedule"], "seed": spec["seed"]}
    return {"step": "TLR", "status": status, "comparison": comparison, "reasons": reasons,
            "final_failures": final_failures, "decision": decision, "held_out_test_accessed": False}


def control_parity(collected: dict[str, Any], manifest: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """Self-check: the five controls' fold values equal the A4 assessment's and their evidence the bound evidence."""
    problems = []
    for fold in spec["folds"]:
        name = f"{spec['selection_cell']}__fold{fold}__seed{spec['seed']}"
        unit, bound = collected["units"][name], manifest["controls"][name]
        if unit.get("status") != "completed":
            problems.append(f"{name}: {unit.get('status')} ({unit.get('note')})")
            continue
        value = unit["metrics"]["common4"]["balanced_accuracy"]
        if not math.isclose(value, bound["a4_common4_ba"], rel_tol=0.0, abs_tol=PARITY_TOLERANCE):
            problems.append(f"{name}: {value!r} != A4 {bound['a4_common4_ba']!r}")
        if any(unit["evidence"][k] != bound[k] for k in EVIDENCE_KEYS):
            problems.append(f"{name}: evidence differs from the manifest")
        try:
            expected_candidate_groups(unit["optimizer_groups"])
        except ValueError as exc:
            problems.append(f"{name}: {exc}")
    values = [collected["units"][f"{spec['selection_cell']}__fold{k}__seed{spec['seed']}"].get("metrics", {})
              .get("common4", {}).get("balanced_accuracy") for k in spec["folds"]]
    mean = float(np.mean(values)) if all(v is not None for v in values) else None
    if mean is not None and not math.isclose(mean, spec["control_source"]["reference_mean_common4_ba"], abs_tol=1e-12):
        problems.append(f"control mean {mean!r} != reference {spec['control_source']['reference_mean_common4_ba']!r}")
    return {"equal": not problems, "problems": problems, "tolerance": PARITY_TOLERANCE, "control_mean_common4_ba": mean}


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def run(*, data: Path, manifest: dict[str, Any], manifest_path: Path | None, evidence_dir: Path, out_dir: Path,
        cohort: Path = COHORT_SOURCE, purpose: str = "assessment") -> Path:
    spec = load_spec()
    a4 = pa.load_spec()
    istats = s10a.interval_spec(spec, a4)
    problems = tl.verify_manifest(manifest)
    require(not problems, "The launch manifest does not verify: " + "; ".join(problems))
    evidence = e2a.check_evidence_copy(evidence_dir)
    with tempfile.TemporaryDirectory(prefix="tlr-view-") as scratch:
        view_dir = Path(scratch) / "campaign"
        view = e2a.build_view(view_dir, data=data, evidence_dir=evidence_dir, manifest=manifest, cohort=cohort)
        paths = v3.V3Paths(view_dir)
        pa.check_spec_against_campaign(a4, e2a.e2.read_json(paths.manifest))
        amp = bool((v3.read_execution_settings(paths) or {}).get("amp", False))
        plan = [vm.differences_from_control(paths, fold, python_bin="python", train_dir=paths.root / "train",
                                            esm_dir=paths.root / "esm", device="cuda", lane=0, amp=amp)
                for fold in vm.FOLDS]
        collected = collect(paths, manifest, data=data, evidence_dir=evidence_dir, amp=amp)
        _, membership = pa.read_contract(paths)
        result = assess(collected, manifest, spec, istats, membership)
    configuration = {"all_expected": all(vm.expected_lr_difference(row) for row in plan), "fits": plan}
    parity = control_parity(collected, manifest, spec)
    if result["decision"] is not None and not (parity["equal"] and configuration["all_expected"]):
        result["status"] = "blocked: the controls or the configuration difference do not verify from these copies"
        result["decision"] = None
    payload = {
        "campaign_id": tl.CAMPAIGN_ID, "kind": "TLR", "purpose": purpose,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "assessor_sha256": sha256(Path(__file__)), "launcher_sha256": sha256(ROOT / "pmm_trunklr.py"),
        "vm_runner_sha256": sha256(ROOT / tl.VM_RUNNER), "a4_assessor_sha256": sha256(ROOT / "pmm_v3_assessment.py"),
        "spec_sha256": tl.FROZEN_SPEC_SHA256, "a4_spec_sha256": manifest["a4_spec_sha256"],
        "manifest": {"path": None if manifest_path is None else str(manifest_path),
                     "sha256": None if manifest_path is None else sha256(manifest_path)},
        "a4_assessment": manifest["a4_assessment"], "evidence_copy": evidence, "view": view,
        "configuration_difference": configuration, "control_parity": parity,
        "unit_evidence": {name: record.get("evidence") for name, record in sorted(collected["units"].items())},
        "unit_status": {name: {"status": record.get("status"), "note": record.get("note"),
                               "durable": record.get("durable")} for name, record in sorted(collected["units"].items())},
        "interpretation": spec["interpretation"],
        "result": pa._jsonable(result), "held_out_test_accessed": False,
    }
    if result["decision"] is not None:
        payload["result"]["decision"].update({"spec_sha256": tl.FROZEN_SPEC_SHA256,
                                              "manifest_sha256": payload["manifest"]["sha256"],
                                              "evidence": payload["unit_evidence"], "held_out_test_accessed": False})
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "assessment.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_dir / "assessment.json"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=tl.CAMPAIGN,
                        help="the v3 execution root (DeepMzyme_Data/campaigns/pmm_ion_metal_v3)")
    parser.add_argument("--records", type=Path, default=tl.RECORDS, help="this campaign's records folder")
    parser.add_argument("--manifest", type=Path, help="default: the recorded launch manifest")
    parser.add_argument("--evidence", type=Path, help="evidence copy (default: the newest comparison copy)")
    parser.add_argument("--cohort", type=Path, default=COHORT_SOURCE)
    parser.add_argument("--out-dir", type=Path, help="default: <records>/assessments/TLR_<UTC>")
    parser.add_argument("--readiness-dry-run", action="store_true",
                        help="CPU readiness check before any fit: reads another step's evidence copy, builds the "
                             "manifest in memory if none is recorded, and writes outside the comparison's assessments")
    args = parser.parse_args(argv)
    assessments = args.records / "assessments"
    manifest_path = args.manifest or tl.record_paths(args.records)["manifest"]
    if args.readiness_dry_run:
        require(args.evidence is not None and args.out_dir is not None
                and assessments.resolve() not in args.out_dir.resolve().parents,
                "A readiness dry run names --evidence and writes outside the comparison's assessments (--out-dir)")
        evidence_dir, purpose = args.evidence, "readiness dry run (not an assessment)"
        if manifest_path.is_file():
            manifest = e2a.e2.read_json(manifest_path)
        else:
            manifest, manifest_path = tl.build_manifest(args.data_root, log_entry="tlr-000-dry-run"), None
    else:
        evidence_dir = args.evidence or e2a.newest_evidence(args.records)
        require(evidence_dir.resolve().parent == args.records.resolve(),
                "An assessment reads a comparison evidence copy ('evidence' after the session)")
        manifest, purpose = e2a.e2.read_json(manifest_path), "assessment"
    out_dir = args.out_dir or assessments / f"TLR_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    path = run(data=args.data_root, manifest=manifest, manifest_path=manifest_path, evidence_dir=evidence_dir,
               out_dir=out_dir, cohort=args.cohort, purpose=purpose)
    written = json.loads(path.read_text())
    print(json.dumps({"assessment": str(path), "sha256": sha256(path), "status": written["result"]["status"],
                      "decision": written["result"]["decision"],
                      "control_parity": written["control_parity"],
                      "configuration_all_expected": written["configuration_difference"]["all_expected"]}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"trunk-LR comparison assessment refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
