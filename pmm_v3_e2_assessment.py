#!/usr/bin/env python3
"""Validation-only assessment of v3 step E2 under its frozen specification (workstation, CPU).

Rules: docs/campaigns/pmm_ion_metal_v3/e2_assessment_spec.md and .json (frozen 2026-10-08, log v3-028; this tool
pins the JSON hash and refuses another). The A4 specification, the A4 assessor and the original step E assessment
are unchanged and only read.

Data path: the assessment reads the host-pulled copies on Data1, never the VM. From the newest evidence copy (the
VM's campaign files and lane status records at that time) it builds a temporary read-only view of the campaign
root whose lanes point at the durable lane copies; every unit it uses must match an acknowledged host-pull
receipt file by file (SHA-256 and size), and the frozen artifact checks of pmm_v3_assessment.load_unit (recorded
identity, terminal checkpoint, reconciled predictions, independent replay) then apply unchanged. As a self-check
the original A4 step E result is re-derived from the same view and must equal the saved assessment.

Per E2 specification: 17 declared comparisons on folds 1-4 (seed 42), unadjusted 95% fold-bootstrap and t
intervals decide, Bonferroni over 17 is sensitivity evidence; two conclusions per candidate (own model against
its matched control; replacement of the original selection); the fold-0 development difference is reported
separately; five-fold summaries are descriptive; a machine-readable final selection. A comparison against a
five- or six-class original selection uses common-four balanced accuracy from collapsed predictions and keeps
the selection's native metrics. The tool never starts a fit and never opens held-out data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import pmm_v3_assessment as pa  # noqa: E402
import pmm_v3_campaign as v3  # noqa: E402
import pmm_v3_step_e2 as e2  # noqa: E402

COHORT_SOURCE = Path("/media/mechti/Data1/DeepMzyme_Data/campaigns/pmm_ion_metal_v2_context/train_cohort.csv")
ROOT_FILES = ("campaign_manifest.json", "execution_settings.json", "fold_class_weights.json", "campaign_extension.json")
FAMILY_ORDER = ("only_gvp", "gvp_late_fusion")  # E2 tie-breaker: simpler family first
PARITY_TOLERANCE = 1e-9
INTERPRETATION_HEADING = "## 0. Interpretation (reported verbatim with every E2 result)"

# The values this code implements; the frozen specification must state exactly these.
TESTED = {
    "comparison_family_size": 17, "confidence": 0.95, "bootstrap_resamples": 10000, "bootstrap_seed": 42,
    "bootstrap_interval": "percentile", "t_degrees_of_freedom": 3, "multiplicity": "bonferroni",
    "recall_drop_limit": 0.03, "tie_band": 0.002, "assessment_seed": 42, "folds": [1, 2, 3, 4],
    "target": "four_class", "primary_estimand": "mean over folds 1-4",
    "development_fold": {"fold": 0, "role": "reported separately; never inside an interval", "seed": 42},
    "own_model_conclusion": ["improvement interval-supported on folds 1-4",
                             "positive mean gain on folds 1-4, not interval-supported",
                             "no positive mean gain on folds 1-4",
                             "blocked by a recall gate against the matched control"],
    "replacement_labels": ["selected", "eligible, not best",
                           "not eligible: <failed conditions by comparator, class, magnitude>"],
    "tie_breakers": ["mean minimum common-four recall (higher)", "worst-fold BA (higher)", "SD over folds 1-4 (lower)",
                     "simpler family (only_gvp before gvp_late_fusion)", "stable recipe name"],
}
TESTED_REPLACEMENT = {"c_min_mean_gain_over_original_selection": 0.002, "c_min_mean_gain_strict": True}
INTERVAL_KEYS = ("confidence", "bootstrap_resamples", "bootstrap_seed", "recall_drop_limit")
REFIT = {"seed": 42, "checkpoint_rule": "terminal", "calibration": None, "ensemble": False}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Specification
# ---------------------------------------------------------------------------

def load_spec(path: Path = e2.E2_SPEC_JSON, *, expected_sha256: str | None = None) -> dict[str, Any]:
    """The frozen E2 specification; refused if its hash or a tested value differs."""
    try:
        spec = e2.load_e2_spec(path, expected_sha256=expected_sha256)
    except e2.LauncherError as exc:
        raise ValueError(str(exc)) from None
    differing = sorted(key for key, value in TESTED.items() if spec.get(key) != value)
    rule = spec.get("replacement_rule") or {}
    differing += sorted(key for key, value in TESTED_REPLACEMENT.items() if rule.get(key) != value)
    require(not differing, f"E2 specification differs from the tested rules in {differing}")
    return spec


def interval_spec(spec: dict[str, Any], a4: dict[str, Any]) -> dict[str, Any]:
    """The A4 interval and gate settings, which the E2 specification keeps unchanged."""
    differing = [key for key in INTERVAL_KEYS if spec[key] != a4[key]]
    require(not differing, f"The E2 and A4 interval settings differ in {differing}")
    return {key: a4[key] for key in INTERVAL_KEYS}


def interpretation(path: Path = e2.E2_SPEC_MD) -> str:
    """Section 0 of the frozen E2 page, verbatim."""
    require(sha256(path) == e2.FROZEN_E2_SPEC_SHA256["e2_assessment_spec.md"], f"{path} is not the frozen page")
    text = Path(path).read_text(encoding="utf-8")
    start = text.index(INTERPRETATION_HEADING) + len(INTERPRETATION_HEADING)
    return text[start:text.index("\n## 1.", start)].strip()


# ---------------------------------------------------------------------------
# The verified read-only view of the Data1 copies
# ---------------------------------------------------------------------------

def newest_evidence(evidence_root: Path) -> Path:
    found = sorted(Path(evidence_root).glob("evidence_*/evidence_manifest.json"))
    require(bool(found), f"No evidence copy under {evidence_root} ('evidence' first)")
    return found[-1].parent


def check_evidence_copy(evidence_dir: Path) -> dict[str, Any]:
    """The evidence copy's own manifest: every file present with its recorded SHA-256, nothing mismatched."""
    manifest = e2.read_json(evidence_dir / "evidence_manifest.json")
    require(not manifest.get("mismatched"), f"{evidence_dir}: the copy recorded mismatches")
    for name, digest in manifest["files"].items():
        if name.startswith("campaign/") and (name.endswith(".json") or name.endswith(".jsonl")):
            require(sha256(evidence_dir / name) == digest, f"{evidence_dir}/{name} differs from its evidence manifest")
    return {"path": str(evidence_dir), "copied_at": manifest.get("copied_at"), "files": len(manifest["files"])}


def build_view(view: Path, *, data: Path, evidence_dir: Path, manifest: dict[str, Any],
               cohort: Path = COHORT_SOURCE) -> dict[str, Any]:
    """A campaign root of symlinks: the immutable root files from the evidence copy (hash-checked against the
    bound extension record), the frozen fold file and cohort, and per lane the status records the VM held at the
    copy (each identical to its durable copy) with the durable run directories."""
    campaign = evidence_dir / "campaign"
    view.mkdir(parents=True, exist_ok=False)
    for name in ROOT_FILES:
        require((campaign / name).is_file(), f"{campaign / name} is missing")
        (view / name).symlink_to(campaign / name)
    extension = e2.read_json(view / "campaign_extension.json")
    require(sha256(view / "campaign_extension.json") == manifest["extension"]["sha256"],
            "The evidence copy's extension record is not the one the E2 manifest binds")
    require(sha256(view / "campaign_manifest.json") == extension["parent"]["manifest_sha256"],
            "The evidence copy's campaign manifest is not the extension's parent")
    require(sha256(view / "execution_settings.json") == extension["parent"]["execution_settings_sha256"],
            "The evidence copy's execution setting is not the extension's")
    campaign_manifest = e2.read_json(view / "campaign_manifest.json")
    require(sha256(view / "fold_class_weights.json") == campaign_manifest["fold_class_weights_sha256"],
            "The evidence copy's class weights are not the prepared ones")
    folds = data / "folds" / campaign_manifest["fold_set"]["id"] / "fold_membership.csv"
    require(sha256(folds) == campaign_manifest["fold_set"]["fold_membership_sha256"], f"{folds} is not the frozen fold file")
    (view / "fold_membership.csv").symlink_to(folds)
    require(sha256(cohort) == campaign_manifest["cohort"]["sha256"], f"{cohort} is not the frozen cohort")
    (view / "frozen_inputs").mkdir()
    (view / "frozen_inputs" / "train_cohort.csv").symlink_to(cohort)
    if (campaign / "failed_attempts").is_dir():
        (view / "failed_attempts").symlink_to(campaign / "failed_attempts")
    statuses = 0
    for lane_dir in sorted((campaign / "lanes").glob("lane*")):
        durable = data / "durable" / lane_dir.name
        target = view / "lanes" / lane_dir.name
        target.mkdir(parents=True)
        (target / "runs").symlink_to(durable / "runs")
        for status in sorted(lane_dir.glob("run_status_*.json")):
            copy = durable / status.name
            require(copy.is_file() and copy.read_bytes() == status.read_bytes(),
                    f"{lane_dir.name}/{status.name}: the durable copy differs from the evidence copy (pull first)")
            (target / status.name).symlink_to(copy)
            statuses += 1
    return {"evidence_copy": str(evidence_dir), "cohort": str(cohort), "fold_membership": str(folds),
            "status_records": statuses}


def verify_durable_unit(data: Path, lane: int, run_name: str, *, evidence_dir: Path) -> dict[str, Any]:
    """The unit's durable copy equals an acknowledged host-pull receipt file by file, nothing else is there, and the
    acknowledgment was uploaded to the lane (the evidence copy of the VM lane holds the same acknowledgment)."""
    lane_root = data / "durable" / f"lane{lane}"
    status_rel = f"run_status_{run_name}.json"
    status_sha = sha256(lane_root / status_rel)
    prefix = f"runs/{run_name}/"
    matched = []
    for receipt in sorted((lane_root / "persistence_receipts").glob("*.json")):
        if receipt.name.endswith((".ack.json", ".state.json")):
            continue
        entries = {entry["path"]: entry for entry in e2.read_json(receipt)["files"]}
        if entries.get(status_rel, {}).get("sha256") == status_sha:
            matched.append((receipt, entries))
    require(bool(matched), f"{run_name}: no host-pull receipt lists its current status record in lane{lane}")
    checked = []
    for receipt, entries in matched:
        manifest = e2.read_json(receipt)
        ack_path = lane_root / manifest["ack_path"]
        require(ack_path.is_file(), f"{run_name}: receipt {receipt.name} was never acknowledged")
        ack = e2.read_json(ack_path)
        require(ack.get("manifest_sha256") == sha256(receipt) and ack.get("file_count") == len(entries),
                f"{run_name}: the acknowledgment does not match receipt {receipt.name}")
        uploaded = Path(evidence_dir) / "campaign" / "lanes" / f"lane{lane}" / manifest["ack_path"]
        require(uploaded.is_file() and uploaded.read_bytes() == ack_path.read_bytes(),
                f"{run_name}: the acknowledgment of receipt {receipt.name} is not on the VM lane in the evidence copy "
                "(never uploaded: 'pull' again, then 'evidence')")
        members = {path: entry for path, entry in entries.items() if path.startswith(prefix)}
        require(bool(members), f"{run_name}: receipt {receipt.name} lists no run files")
        for path, entry in sorted(members.items()):
            target = lane_root / path
            require(target.is_file() and not target.is_symlink() and target.stat().st_size == entry["size_bytes"]
                    and sha256(target) == entry["sha256"], f"{run_name}: {path} differs from receipt {receipt.name}")
        present = {str(p.relative_to(lane_root)) for p in (lane_root / "runs" / run_name).rglob("*") if p.is_file()}
        require(present == set(members), f"{run_name}: the durable run directory holds files outside receipt "
                                         f"{receipt.name}: {sorted(present ^ set(members))[:3]}")
        checked.append({"receipt": receipt.name, "ack_verified_unix": ack.get("verified_unix"),
                        "run_files": len(members)})
    return {"lane": lane, "receipts": checked}


def lane_of(record: dict[str, Any]) -> int:
    name = Path(record["status_path"]).parent.name
    require(name.startswith("lane") and name[4:].isdigit(), f"Unexpected status location {record['status_path']}")
    return int(name[4:])


def collect_verified(view: v3.V3Paths, names: list[str], *, data: Path, bound: dict[str, dict[str, Any]],
                     evidence_dir: Path) -> dict[str, Any]:
    """Unit records as pmm_v3_assessment.collect makes them, after the durable-copy check; evidence that is
    missing, failed, mismatched, unreplayed or unverified is recorded as such (never silently dropped)."""
    contract, membership = pa.read_contract(view)
    statuses = v3.completed_units(view)
    out = {}
    for name in dict.fromkeys(names):
        unit = v3.Unit.parse(name)
        try:
            record = pa.load_unit(view, unit, statuses, membership, contract, epochs=v3.PROFILE["epochs"])
            if record["status"] == "completed":
                record["durable"] = verify_durable_unit(data, lane_of(statuses[record["run_name"]]),
                                                        record["run_name"], evidence_dir=evidence_dir)
                expected = bound.get(name)
                if expected is not None and any(record["evidence"][key] != expected[key]
                                                for key in ("checkpoint_sha256", "predictions_sha256")):
                    raise ValueError("differs from the evidence the E2 manifest binds")
        except (ValueError, KeyError, OSError) as exc:
            record = {"name": name, "status": "invalid evidence", "note": f"{type(exc).__name__}: {exc}",
                      "final_failure": False}
        out[name] = record
    return out


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------

def cell_parts(cell: str) -> tuple[str, str, str]:
    family, target, recipe = cell.split("__")
    return family, target, recipe


def cell_folds(collected: dict[str, Any], cell: str, folds) -> dict[int, Any]:
    return pa.cell_folds(collected, *cell_parts(cell), folds, e2.SEED)


def compare_cells(control: dict[int, Any], control_target: str, challenger: dict[int, Any], challenger_target: str,
                  spec: dict[str, Any], *, n_contrasts: int) -> dict[str, Any]:
    """pmm_v3_assessment.compare for a four-class control; for a five- or six-class control the same rule with
    the control summarized under its own scheme (common-four BA and recalls from collapsed predictions)."""
    if control_target == "four_class":
        return pa.compare(control, challenger, spec, challenger_target=challenger_target, n_contrasts=n_contrasts)
    return compare_any_target(control, control_target, challenger, challenger_target, spec, n_contrasts=n_contrasts)


def compare_any_target(control: dict[int, Any], control_target: str, challenger: dict[int, Any],
                       challenger_target: str, spec: dict[str, Any], *, n_contrasts: int) -> dict[str, Any]:
    """pmm_v3_assessment.compare with the control's own target scheme (tested equal to it for four_class)."""
    missing = pa.incomplete(control) + pa.incomplete(challenger)
    if missing or sorted(control) != sorted(challenger):
        return {"status": "incomplete", "missing": missing, "verdict": "incomplete"}
    order = sorted(control)
    a, b = pa.summarize(control, control_target), pa.summarize(challenger, challenger_target)
    differences = [challenger[k]["metrics"]["common4"]["balanced_accuracy"]
                   - control[k]["metrics"]["common4"]["balanced_accuracy"] for k in order]
    intervals = pa.paired_intervals(differences, spec, n_contrasts=n_contrasts)
    drops = {label: (None if a["mean_common4_recall"][label] is None or b["mean_common4_recall"][label] is None
                     else a["mean_common4_recall"][label] - b["mean_common4_recall"][label]) for label in pa.COMMON4}
    gates = {"no_missing_common4_recall": all(v is not None for v in b["mean_common4_recall"].values()),
             "no_zero_common4_recall": all(v not in (None, 0.0) for v in b["mean_common4_recall"].values()),
             "no_common4_recall_drop_over_limit": all(v is not None and v <= spec["recall_drop_limit"] + 1e-12
                                                      for v in drops.values())}
    if challenger_target != "four_class":
        gates["no_zero_native_recall"] = all(v not in (None, 0.0) for v in b["mean_native_recall"].values())
    outcome = intervals["unadjusted"]["outcome"]
    verdict = ("better" if outcome == "better" and all(gates.values())
               else "blocked by a recall gate" if outcome == "better" else outcome)
    return {"status": "complete", "folds": order, **intervals, "class_recall_drop": drops, "gates": gates,
            "verdict": verdict,
            "adjusted_claim_supported": verdict == "better" and intervals["adjusted"]["outcome"] == "better"}


def gate_failures(result: dict[str, Any], challenger: dict[int, Any], comparator: str, ids: str,
                  limit: float) -> list[str]:
    """Every failed recall gate, named by comparator, class and magnitude."""
    if result.get("status") != "complete":
        return []
    means = pa.summarize(challenger, "four_class")["mean_common4_recall"]
    out = []
    for label in pa.COMMON4:
        if means[label] is None:
            out.append(f"recall gate against {comparator} ({ids}): {label} mean recall missing")
        elif means[label] == 0.0:
            out.append(f"recall gate against {comparator} ({ids}): {label} mean recall is zero")
        drop = result["class_recall_drop"][label]
        if drop is not None and drop > limit + 1e-12:
            out.append(f"recall gate against {comparator} ({ids}): {label} mean recall {-100 * drop:+.2f} points "
                       f"(limit {-100 * limit:.2f})")
    return out


def own_model_conclusion(result: dict[str, Any], failures: list[str], vocabulary: list[str]) -> dict[str, Any]:
    """E2 specification section 5 (the A4 improvement-check logic against the matched control)."""
    if result.get("status") != "complete":
        return {"status": "incomplete", "missing": result.get("missing", [])}
    mean = result["mean_difference"]
    if result["verdict"] == "better":
        status = vocabulary[0]
    elif mean > 0 and all(result["gates"].values()):
        status = vocabulary[1]
    elif mean <= 0:
        status = vocabulary[2]
    else:
        status = vocabulary[3]
    return {"status": status, "mean_difference": mean, "verdict": result["verdict"], "failed_gates": failures}


def run_comparisons(collected: dict[str, Any], manifest: dict[str, Any], spec: dict[str, Any],
                    istats: dict[str, Any]) -> dict[str, Any]:
    out = {}
    n = spec["comparison_family_size"]
    for item in manifest["comparisons"]:
        candidate = cell_folds(collected, item["candidate"], e2.FOLDS)
        comparator = cell_folds(collected, item["comparator"], e2.FOLDS)
        result = compare_cells(comparator, cell_parts(item["comparator"])[1], candidate, "four_class", istats,
                               n_contrasts=n)
        values = {}
        for role, folds in (("candidate", candidate), ("comparator", comparator)):
            values[role] = {str(k): (unit["metrics"]["common4"]["balanced_accuracy"]
                                     if unit.get("status") == "completed" else None) for k, unit in folds.items()}
        if result["status"] == "complete":
            result["sd_difference"] = float(np.std(result["fold_differences"], ddof=1))
        out[item["id"]] = {**item, **result, "fold_common4_ba": values}
    return out


def summary_or_missing(collected: dict[str, Any], cell: str, folds) -> dict[str, Any]:
    cells = cell_folds(collected, cell, folds)
    missing = pa.incomplete(cells)
    return {"incomplete": missing} if missing else pa.summarize(cells, cell_parts(cell)[1])


def assess_e2(collected: dict[str, Any], manifest: dict[str, Any], spec: dict[str, Any],
              istats: dict[str, Any]) -> dict[str, Any]:
    """Comparisons, both conclusions per candidate, the replacement rule and the final selection."""
    comparisons = run_comparisons(collected, manifest, spec, istats)
    selection = manifest["original_assessment"]["selection"]
    limit = istats["recall_drop_limit"]
    min_gain = spec["replacement_rule"]["c_min_mean_gain_over_original_selection"]
    candidates, eligible = {}, []
    for item in e2.candidate_cells(spec):
        cell = item["cell"]
        mine = [comp for comp in comparisons.values() if comp["candidate"] == cell]
        by_comparator: dict[str, list[dict[str, Any]]] = {}
        for comp in mine:
            by_comparator.setdefault(comp["comparator"], []).append(comp)
        challenger = cell_folds(collected, cell, e2.FOLDS)
        failures = {comparator: gate_failures(comps[0], challenger, comparator,
                                              " = ".join(c["id"] for c in comps), limit)
                    for comparator, comps in by_comparator.items()}
        matched = by_comparator[item["matched_control"]][0]
        conclusions = {"own_model": {"comparator": item["matched_control"], "comparison": matched["id"],
                                     **own_model_conclusion(matched, failures[item["matched_control"]],
                                                            spec["own_model_conclusion"])}}
        if item["additional_comparator"]:
            extra = by_comparator[item["additional_comparator"]][0]
            conclusions["against_additional_comparator"] = {
                "comparator": item["additional_comparator"], "comparison": extra["id"],
                **own_model_conclusion(extra, failures[item["additional_comparator"]], spec["own_model_conclusion"])}
        reasons = []
        for comparator, comps in by_comparator.items():
            ids, comp = " = ".join(c["id"] for c in comps), comps[0]
            if comp["status"] != "complete":
                reasons.append(f"(a) incomplete evidence against {comparator} ({ids}): {comp.get('missing')}")
                continue
            lower = comp["unadjusted"]
            if not (lower["bootstrap"][0] > 0 and lower["t"][0] > 0):
                reasons.append(f"(b) not interval-supported against {comparator} ({ids}): bootstrap lower bound "
                               f"{100 * lower['bootstrap'][0]:+.2f}, t lower bound {100 * lower['t'][0]:+.2f} points")
            reasons += [f"(d) {failure}" for failure in failures[comparator]]
        against_selection = by_comparator[selection][0]
        if against_selection["status"] == "complete" and not against_selection["mean_difference"] > min_gain:
            reasons.append(f"(c) mean gain over the original selection "
                           f"{100 * against_selection['mean_difference']:+.2f} points, not above "
                           f"{100 * min_gain:.2f}")
        summary = summary_or_missing(collected, cell, e2.FOLDS)
        development = {}
        for comparator in by_comparator:
            pair = (cell_folds(collected, cell, (e2.DEVELOPMENT_FOLD,)),
                    cell_folds(collected, comparator, (e2.DEVELOPMENT_FOLD,)))
            development[comparator] = (None if pa.incomplete(pair[0]) + pa.incomplete(pair[1]) else
                                       pair[0][0]["metrics"]["common4"]["balanced_accuracy"]
                                       - pair[1][0]["metrics"]["common4"]["balanced_accuracy"])
        candidates[cell] = {"family": item["family"], "recipe": item["recipe"],
                            "matched_control": item["matched_control"],
                            "additional_comparator": item["additional_comparator"],
                            "comparisons": [comp["id"] for comp in mine], "conclusions": conclusions,
                            "replacement": {"eligible": not reasons, "reasons": reasons},
                            "summary_folds_1_4": summary,
                            "development_fold0_difference": {
                                "values": development,
                                "note": "fold 0, seed 42: the development fold; never inside an interval"},
                            "five_fold_summary_descriptive": summary_or_missing(collected, cell, range(v3.N_FOLDS))}
        if not reasons:
            eligible.append({"cell": cell, "family": item["family"], "recipe": item["recipe"], **summary})
    e2_units = manifest["units"]
    final_failures = sorted(name for name in e2_units if collected.get(name, {}).get("final_failure"))
    missing = sorted(name for name in e2_units if collected.get(name, {}).get("status") != "completed")
    incomplete_comparisons = sorted(cid for cid, comp in comparisons.items() if comp["status"] != "complete")
    if final_failures:
        status = f"blocked: {final_failures} failed after their one rerun; diagnosis before the final E2 assessment"
    elif missing or incomplete_comparisons:
        status = f"incomplete: {len(missing)} of {len(e2_units)} E2 units not verified; comparisons {incomplete_comparisons}"
    else:
        status = "complete"
    for cell, item in candidates.items():
        item["replacement"]["label"] = ("not eligible: " + "; ".join(item["replacement"]["reasons"])
                                        if not item["replacement"]["eligible"] else "eligible, not best"
                                        if status == "complete" else provisional_label(status))
    final = None
    if status == "complete":
        if eligible:
            top = max(row["mean_common4_ba"] for row in eligible)
            tied = [row for row in eligible if top - row["mean_common4_ba"] <= spec["tie_band"]]
            chosen = min(tied, key=tie_key)
            candidates[chosen["cell"]]["replacement"]["label"] = "selected"
            selected, reason = chosen["cell"], ("highest mean common-four BA on folds 1-4 among eligible candidates"
                                                if len(tied) == 1 else "tie-breakers within the tie band")
        else:
            selected, reason = selection, "no eligible candidate; the original selection is retained"
        final = {"selected": selected, "replaced_original": selected != selection, "reason": reason,
                 "original_selection": selection, "eligible": [row["cell"] for row in eligible], "refit": REFIT}
    return {"step": "E2", "status": status, "comparisons": comparisons, "candidates": candidates,
            "original_selection": {
                "cell": selection, "target": manifest["original_assessment"]["selection_target"],
                "assessment_sha256": manifest["original_assessment"]["sha256"],
                "summary_folds_1_4": summary_or_missing(collected, selection, e2.FOLDS),
                "five_fold_summary_descriptive": summary_or_missing(collected, selection, range(v3.N_FOLDS)),
                "note": "unchanged; native metrics under its own scheme"},
            "new_controls": {cell: {"summary_folds_1_4": summary_or_missing(collected, cell, e2.FOLDS)}
                             for cell in spec["new_control_fits"]},
            "missing_e2_units": missing, "final_failures": final_failures, "final_selection": final,
            "multiplicity": {"method": "bonferroni", "n_contrasts": spec["comparison_family_size"],
                             "role": spec["adjusted_intervals_role"]},
            "held_out_test_accessed": False}


def tie_key(row: dict[str, Any]) -> tuple:
    """E2 specification section 6, within the tie band: higher mean minimum common-four recall, higher worst-fold
    BA, lower SD over folds 1-4, simpler family (Only-GVP before late fusion), then the stable recipe name."""
    return (-row["mean_min_recall"], -row["worst_fold_common4_ba"], row["sd_common4_ba"],
            FAMILY_ORDER.index(row["family"]), row["recipe"])


def provisional_label(status: str) -> str:
    """An eligible candidate while no final selection exists (nothing is ranked yet)."""
    return f"eligible so far; no final selection (assessment {status.split(':')[0]})"


def withhold_selection(result: dict[str, Any], status: str) -> None:
    """Block the final selection and keep every replacement label consistent with that."""
    result["status"], result["final_selection"] = status, None
    for item in result["candidates"].values():
        if item["replacement"]["eligible"]:
            item["replacement"]["label"] = provisional_label(status)


# ---------------------------------------------------------------------------
# Self-check: the original A4 step E result from the same verified copies
# ---------------------------------------------------------------------------

def a4_unit_names(finals: dict[str, str]) -> list[str]:
    """The units pmm_v3_assessment.main collects for step E (same order)."""
    units = v3.step_units("C")[:9] + v3.step_units("E-neutral")
    units += v3.step_units("E-improvement", final_recipes=finals)
    units += [v3.Unit(f, "four_class", r, 0, 42) for f, r in finals.items()]
    return [unit.name for unit in dict.fromkeys(units)]


def differences(saved: Any, derived: Any, path: str = "") -> list[str]:
    if isinstance(saved, dict) and isinstance(derived, dict):
        out = [f"{path}/{k}: key differs" for k in sorted(set(saved) ^ set(derived))]
        for key in sorted(set(saved) & set(derived)):
            out += differences(saved[key], derived[key], f"{path}/{key}")
        return out
    if isinstance(saved, list) and isinstance(derived, list):
        if len(saved) != len(derived):
            return [f"{path}: length {len(saved)} != {len(derived)}"]
        return [m for i, (a, b) in enumerate(zip(saved, derived)) for m in differences(a, b, f"{path}[{i}]")]
    if isinstance(saved, float) or isinstance(derived, float):
        ok = (isinstance(saved, (int, float)) and isinstance(derived, (int, float))
              and math.isclose(saved, derived, rel_tol=0.0, abs_tol=PARITY_TOLERANCE))
        return [] if ok else [f"{path}: {saved!r} != {derived!r}"]
    return [] if saved == derived else [f"{path}: {saved!r} != {derived!r}"]


def a4_parity(collected: dict[str, Any], original: dict[str, Any], a4: dict[str, Any],
              finals: dict[str, str]) -> dict[str, Any]:
    names = a4_unit_names(finals)
    subset = {name: collected[name] for name in names}
    bad = sorted(name for name, record in subset.items() if record.get("status") != "completed")
    if bad:
        return {"equal": False, "problems": [f"not verified: {bad}"]}
    derived = json.loads(json.dumps(pa._jsonable(pa.assess_step_e(subset, a4, final_recipes=finals))))
    problems = differences(original["result"], derived)
    evidence = {name: record["evidence"] for name, record in subset.items()}
    problems += differences(original["unit_evidence"], evidence, "/unit_evidence")
    return {"equal": not problems, "units": len(names), "problems": problems[:20],
            "tolerance": PARITY_TOLERANCE}


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def run(*, data: Path, manifest_path: Path, evidence_dir: Path, out_dir: Path, cohort: Path = COHORT_SOURCE,
        purpose: str = "assessment", parity: bool = True) -> Path:
    spec = load_spec()
    a4 = pa.load_spec()
    istats = interval_spec(spec, a4)
    manifest = e2.read_json(manifest_path)
    problems = e2.verify_manifest(manifest)
    require(not problems, "The E2 manifest does not verify: " + "; ".join(problems))
    original = e2.read_json(Path(manifest["original_assessment"]["path"]))
    evidence = check_evidence_copy(evidence_dir)
    finals = manifest["original_assessment"]["improvement_recipes"]
    names = list(manifest["units"]) + list(manifest["reused_units"])
    if parity:
        names += a4_unit_names(finals)
    with tempfile.TemporaryDirectory(prefix="v3-e2-view-") as scratch:
        view_dir = Path(scratch) / "campaign"
        view = build_view(view_dir, data=data, evidence_dir=evidence_dir, manifest=manifest, cohort=cohort)
        paths = v3.V3Paths(view_dir)
        pa.check_spec_against_campaign(a4, e2.read_json(paths.manifest))
        collected = collect_verified(paths, names, data=data, bound=manifest["reused_units"],
                                     evidence_dir=evidence_dir)
    result = assess_e2(collected, manifest, spec, istats)
    check = a4_parity(collected, original, a4, finals) if parity else None
    if check is not None and not check["equal"] and result["final_selection"] is not None:
        withhold_selection(result, "blocked: the original A4 result could not be re-derived from these copies")
    payload = {
        "campaign_id": v3.CAMPAIGN_ID, "kind": "E2", "purpose": purpose,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "assessor_sha256": sha256(Path(__file__)), "launcher_sha256": sha256(ROOT / "pmm_v3_step_e2.py"),
        "a4_assessor_sha256": sha256(ROOT / "pmm_v3_assessment.py"),
        "e2_spec_sha256": manifest["e2_spec_sha256"], "a4_spec_sha256": manifest["a4_spec_sha256"],
        "manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        "original_assessment": {"path": manifest["original_assessment"]["path"],
                                "sha256": manifest["original_assessment"]["sha256"],
                                "selection": manifest["original_assessment"]["selection"]},
        "evidence_copy": evidence, "view": view, "a4_parity": check,
        "unit_evidence": {name: record.get("evidence") for name, record in sorted(collected.items())},
        "unit_status": {name: {"status": record.get("status"), "note": record.get("note"),
                               "durable": record.get("durable")} for name, record in sorted(collected.items())},
        "interpretation": interpretation(),
        "result": pa._jsonable(result), "held_out_test_accessed": False,
    }
    if result["final_selection"] is not None:
        payload["result"]["final_selection"].update({
            "e2_spec_sha256": manifest["e2_spec_sha256"], "manifest_sha256": payload["manifest"]["sha256"],
            "evidence": {name: payload["unit_evidence"][name] for name in manifest["units"]
                         + list(manifest["reused_units"])},
            "held_out_test_accessed": False})
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "assessment.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_dir / "assessment.json"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=e2.CAMPAIGN)
    parser.add_argument("--manifest", type=Path, help="default: the recorded E2 manifest")
    parser.add_argument("--evidence", type=Path, help="evidence copy (default: the newest step E2 copy)")
    parser.add_argument("--cohort", type=Path, default=COHORT_SOURCE)
    parser.add_argument("--out-dir", type=Path, help="default: step_e2_evidence/assessments/E2_<UTC>")
    parser.add_argument("--readiness-dry-run", action="store_true",
                        help="CPU readiness check before any E2 fit: allows another step's evidence copy and needs "
                             "--out-dir outside the E2 assessments")
    parser.add_argument("--no-a4-parity", action="store_true", help="skip re-deriving the original A4 result")
    args = parser.parse_args(argv)
    data = args.data_root
    evidence_root = data / e2.EVIDENCE_DIR
    manifest_path = args.manifest or evidence_root / e2.MANIFEST_NAME
    evidence_dir = args.evidence or newest_evidence(evidence_root)
    assessments = evidence_root / "assessments"
    if args.readiness_dry_run:
        require(args.out_dir is not None and assessments.resolve() not in args.out_dir.resolve().parents,
                "A readiness dry run writes outside the E2 assessments (--out-dir)")
        purpose = "readiness dry run (not an E2 assessment)"
    else:
        require(evidence_dir.resolve().parent == evidence_root.resolve(),
                "An E2 assessment reads an E2 evidence copy ('evidence' after the session)")
        purpose = "assessment"
    out_dir = args.out_dir or assessments / f"E2_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    path = run(data=data, manifest_path=manifest_path, evidence_dir=evidence_dir, out_dir=out_dir,
               cohort=args.cohort, purpose=purpose, parity=not args.no_a4_parity)
    written = json.loads(path.read_text())
    print(json.dumps({"assessment": str(path), "sha256": sha256(path), "status": written["result"]["status"],
                      "final_selection": written["result"]["final_selection"],
                      "a4_parity_equal": (written["a4_parity"] or {}).get("equal")}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"v3 E2 assessment refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
