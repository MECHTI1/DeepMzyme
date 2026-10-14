#!/usr/bin/env python3
"""Assessment v2 of the trunk-learning-rate comparison: the final-worker completion gate of
pmm_trunklr_safeguards_v2 before the frozen assessor (pmm_trunklr_assessment.py, unchanged; validation data only).

Added 2026-10-13 at the campaign's closeout maintenance; the original assessment did not use it. A scientific
decision exists only if every candidate fit's final attempt verifies from the evidence copy (exit code 0, hashes of
every file, unit/run/lane/session/identity agreement, no earlier attempt standing in for the final one); otherwise the
status is blocked and the frozen assessor is not run. Outputs always go to a fresh directory; an existing assessment is
never overwritten.

Commands:
  assess --out-dir DIR [--evidence E] [--sessions-from-ledger]
                          the gate, then the frozen assessor into DIR/frozen_assessment
  retrospective --historical PATH --historical-sha256 SHA --out-dir DIR
                          once, after the fact: the gate against the completed evidence with each launch checked
                          against the controller-recorded allocation of its session, the session against the
                          comparison's intended limits (one session, 9,000 s), the historical v1 authorization against
                          the v2 policy schema, then the frozen assessor re-run into DIR/frozen_assessor_rerun and its
                          numbers compared with the historical assessment, which stays the record of the decision
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import pmm_trunklr as tl  # noqa: E402
import pmm_trunklr_safeguards_v2 as sg  # noqa: E402

e2 = tl.e2
require = tl.require
sha256 = tl.sha256_file
RETROSPECTIVE_LABEL = "retrospective verification; these safeguards were not active during the original run"
# The comparison's intended session limits (log tlr-002: one L4 session of at most 2.5 hours).
INTENDED_MAX_SESSIONS = 1
INTENDED_MAX_SESSION_SECONDS = 9000.0
COMPARED_KEYS = ("status", "final_failures", "reasons")


def fresh_dir(path: Path, records: Path) -> Path:
    require(not path.exists(), f"{path} exists; outputs go to a fresh directory")
    require((records / "assessments").resolve() not in path.resolve().parents,
            "v2 outputs never go into the comparison's assessments folder")
    path.mkdir(parents=True)
    return path


def ledger_sessions_for(evidence_dir: Path, spec: dict[str, Any], ledger: Path = e2.LEDGER) -> dict[str, Any]:
    """The controller-recorded allocation of every session the candidate launch records name."""
    sessions = {}
    for unit in spec["units"]:
        try:
            launch = e2.read_json(Path(evidence_dir) / "campaign/step_tlr/launch" / f"{unit}.launch.json")
        except (OSError, ValueError):
            continue
        session_id = launch.get("session_id")
        if isinstance(session_id, str) and session_id not in sessions:
            recorded = sg.ledger_session(session_id, ledger)
            if recorded is not None:
                sessions[session_id] = recorded
    return sessions


def gate(evidence_dir: Path, manifest: dict[str, Any], *, sessions: dict[str, Any] | None,
         durable: Path | None) -> dict[str, Any]:
    spec = tl.load_spec()
    return sg.completion_report(evidence_dir, spec=spec, runner_sha256=manifest["vm_runner"]["sha256"],
                                sessions=sessions, durable=durable)


def run_frozen(*, data: Path, manifest_path: Path, evidence_dir: Path, out_dir: Path, cohort: Path,
               purpose: str) -> Path:
    import pmm_trunklr_assessment as tla  # the frozen assessor, unchanged (imports numpy/torch lazily)

    return tla.run(data=data, manifest=e2.read_json(manifest_path), manifest_path=manifest_path,
                   evidence_dir=evidence_dir, out_dir=out_dir, cohort=cohort, purpose=purpose)


def assess(*, data: Path, records: Path, manifest_path: Path, evidence_dir: Path, out_dir: Path, cohort: Path,
           sessions: dict[str, Any] | None) -> dict[str, Any]:
    """The gate; only a verified gate runs the frozen assessor and carries its decision."""
    out_dir = fresh_dir(out_dir, records)
    manifest = e2.read_json(manifest_path)
    completion = gate(evidence_dir, manifest, sessions=sessions, durable=data / "durable")
    receipt: dict[str, Any] = {"kind": "TLR-v2", "created_at": tl.utc_now(), "evidence_copy": str(evidence_dir),
                               "manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
                               "safety_code_sha256": sg.safety_code_sha256(), "completion": completion,
                               "held_out_test_accessed": False}
    if not completion["verified"]:
        receipt.update({"status": "blocked: final worker completion not verified; no scientific decision",
                        "decision": None})
    else:
        path = run_frozen(data=data, manifest_path=manifest_path, evidence_dir=evidence_dir,
                          out_dir=out_dir / "frozen_assessment", cohort=cohort, purpose="assessment (v2 gate passed)")
        frozen = e2.read_json(path)
        receipt.update({"status": frozen["result"]["status"], "decision": frozen["result"]["decision"],
                        "frozen_assessment": {"path": str(path), "sha256": sha256(path)}})
    (out_dir / "assessment_v2.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def scientific_view(assessment: dict[str, Any]) -> dict[str, Any]:
    """The numbers and decision of an assessment, without paths, times or the purpose."""
    result = assessment["result"]
    comparison = result.get("comparison") or {}
    decision = dict(result.get("decision") or {})
    return {"result": {key: result.get(key) for key in COMPARED_KEYS},
            "decision_outcome": decision.get("outcome"), "decision_reasons": decision.get("reasons"),
            "decision_gvp_learning_rate": decision.get("gvp_learning_rate"),
            "fold_common4_ba": comparison.get("fold_common4_ba"), "fold_differences": comparison.get("fold_differences"),
            "mean_difference": comparison.get("mean_difference"), "unadjusted": comparison.get("unadjusted"),
            "sd_difference": comparison.get("sd_difference"),
            "sensitivity_mean_difference": (comparison.get("sensitivity_folds_1_4") or {}).get("mean_difference"),
            "sensitivity_unadjusted": (comparison.get("sensitivity_folds_1_4") or {}).get("unadjusted"),
            "optimizer_groups": comparison.get("optimizer_groups"), "unit_evidence": assessment.get("unit_evidence"),
            "control_parity": {k: v for k, v in (assessment.get("control_parity") or {}).items()},
            "configuration_all_expected": (assessment.get("configuration_difference") or {}).get("all_expected"),
            "spec_sha256": assessment.get("spec_sha256"), "held_out_test_accessed": assessment.get("held_out_test_accessed")}


def differences(a: Any, b: Any, prefix: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        return [d for key in sorted(set(a) | set(b)) for d in differences(a.get(key), b.get(key), f"{prefix}/{key}")]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, f"{prefix}[{i}]")]
    return [] if a == b else [f"{prefix}: {a!r} != {b!r}"]


def retrospective(*, data: Path, records: Path, historical: Path, historical_sha256: str, out_dir: Path,
                  cohort: Path, ledger: Path = e2.LEDGER, rerun: bool = True) -> dict[str, Any]:
    out_dir = fresh_dir(out_dir, records)
    started = time.time()
    paths = sg.policy_paths(records)
    require(sha256(historical) == historical_sha256, f"{historical} is not the recorded historical assessment")
    history = e2.read_json(historical)
    manifest = e2.read_json(paths["manifest"])
    manifest_problems = tl.verify_manifest(manifest)
    evidence_dir = Path(history["evidence_copy"]["path"])
    spec = tl.load_spec()
    sessions = ledger_sessions_for(evidence_dir, spec, ledger)
    completion = sg.completion_report(evidence_dir, spec=spec, runner_sha256=manifest["vm_runner"]["sha256"],
                                      sessions=sessions, durable=data / "durable")
    # What the v2 session rule would have concluded, evaluated in memory (no policy or binding is written).
    intended = {"authorized_units": list(spec["units"]), "max_sessions": INTENDED_MAX_SESSIONS,
                "max_session_seconds": INTENDED_MAX_SESSION_SECONDS}
    session_checks, bindings = [], []
    for session_id in completion["sessions_used"]:
        recorded = sessions.get(session_id)
        problems, binding = sg.session_gate(intended, "retrospective-in-memory", bindings,
                                            None if recorded is None else {k: recorded[k] for k in sg.SESSION_KEYS},
                                            list(spec["units"]))
        if binding is not None:
            bindings.append(binding)
        session_checks.append({"session_id": session_id, "controller_record": recorded, "problems": problems})
    session_verified = bool(session_checks) and len(completion["sessions_used"]) <= INTENDED_MAX_SESSIONS and all(
        not check["problems"] for check in session_checks)
    authorization = e2.read_json(paths["authorization"])
    as_v2 = sg.policy_problems(authorization, manifest_sha256=sha256(paths["manifest"]),
                               ceiling_sha256=sha256(paths["ceiling"]), authorization_sha256=sha256(paths["authorization"]),
                               code_sha256=sg.safety_code_sha256(), now=time.time())
    receipt: dict[str, Any] = {
        "kind": "TLR-v2-retrospective", "label": RETROSPECTIVE_LABEL, "created_at": tl.utc_now(),
        "historical_assessment": {"path": str(historical), "sha256": historical_sha256,
                                  "outcome": (history["result"].get("decision") or {}).get("outcome"),
                                  "status": history["result"]["status"]},
        "manifest": {"path": str(paths["manifest"]), "sha256": sha256(paths["manifest"]),
                     "historical_assessment_binds_it": history["manifest"]["sha256"] == sha256(paths["manifest"]),
                     "verify_problems": manifest_problems},
        "authorization": {"path": str(paths["authorization"]), "sha256": sha256(paths["authorization"]),
                          "names_manifest_and_ceiling": authorization.get("manifest_sha256") == sha256(paths["manifest"])
                          and authorization.get("ceiling_sha256") == sha256(paths["ceiling"]),
                          "accepted_as_v2_launch_policy": not as_v2, "v2_policy_problems": as_v2},
        "sessions": {"intended_limits": intended, "checks": session_checks, "verified": session_verified},
        "completion": completion, "safety_code_sha256": sg.safety_code_sha256(), "held_out_test_accessed": False,
    }
    verified = completion["verified"] and session_verified and not manifest_problems \
        and receipt["manifest"]["historical_assessment_binds_it"]
    receipt["gate_verified"] = verified
    if verified and rerun:
        path = run_frozen(data=data, manifest_path=paths["manifest"], evidence_dir=evidence_dir,
                          out_dir=out_dir / "frozen_assessor_rerun", cohort=cohort,
                          purpose="retrospective verification re-run (maintenance; not a new assessment)")
        rerun_view, history_view = scientific_view(e2.read_json(path)), scientific_view(history)
        diff = differences(history_view, rerun_view)
        receipt["frozen_assessor_rerun"] = {"path": str(path), "sha256": sha256(path), "differences": diff,
                                            "scientific_result_equal": not diff}
        receipt["decision"] = {"outcome": history_view["decision_outcome"] if not diff else None,
                               "source": "the historical assessment (unchanged); the re-run reproduces it"
                               if not diff else "blocked: the re-run differs from the historical assessment"}
    else:
        receipt["decision"] = {"outcome": None, "source": "blocked: the retrospective gate did not verify"
                               if not verified else "not re-run"}
    receipt["elapsed_seconds"] = round(time.time() - started, 1)
    target = out_dir / "retrospective_verification.json"
    target.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n")
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("assess", "retrospective"))
    parser.add_argument("--data-root", type=Path, default=tl.CAMPAIGN)
    parser.add_argument("--records", type=Path, default=tl.RECORDS)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--cohort", type=Path, help="default: the frozen cohort of the E2 assessor")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sessions-from-ledger", action="store_true",
                        help="assess: check each launch against the controller-recorded allocation of its session")
    parser.add_argument("--historical", type=Path)
    parser.add_argument("--historical-sha256")
    args = parser.parse_args(argv)
    import pmm_v3_e2_assessment as e2a  # frozen; its cohort constant is the one the frozen assessor uses

    args.cohort = args.cohort or e2a.COHORT_SOURCE
    if args.command == "retrospective":
        require(args.historical is not None and args.historical_sha256, "retrospective needs --historical and "
                                                                        "--historical-sha256")
        receipt = retrospective(data=args.data_root, records=args.records, historical=args.historical,
                                historical_sha256=args.historical_sha256, out_dir=args.out_dir, cohort=args.cohort)
        print(json.dumps({"gate_verified": receipt["gate_verified"], "decision": receipt["decision"],
                          "rerun": {k: v for k, v in (receipt.get("frozen_assessor_rerun") or {}).items()
                                    if k != "differences"},
                          "elapsed_seconds": receipt["elapsed_seconds"]}, indent=2))
        return 0
    evidence_dir = args.evidence or e2a.newest_evidence(args.records)
    manifest_path = tl.record_paths(args.records)["manifest"]
    sessions = ledger_sessions_for(evidence_dir, tl.load_spec()) if args.sessions_from_ledger else None
    receipt = assess(data=args.data_root, records=args.records, manifest_path=manifest_path,
                     evidence_dir=evidence_dir, out_dir=args.out_dir, cohort=args.cohort, sessions=sessions)
    print(json.dumps({"status": receipt["status"], "decision": receipt["decision"]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, tl.LauncherError) as exc:
        print(f"trunk-LR comparison assessment v2 refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
