#!/usr/bin/env python3
"""Validation-only assessment of the PMM ion metal v3 campaign (plan steps C, D, E and Stage 6).

A new file: the v2 assessors stay frozen. Every metric is recomputed from the UID-keyed
predictions of each unit's selected checkpoint (the terminal one under the v3 rule) and
reconciled with the run's receipt; only completed, independently replayed units count.

Decision rules (frozen in the A4 specification before any GPU run):
- one paired difference per fold; 95% intervals from a fold bootstrap (10,000 resamples,
  seed 42) and from a Student t interval (df = folds - 1); "better" needs both lower bounds
  above zero, "worse" both upper bounds below zero, otherwise "no clear difference";
- a claim across several contrasts uses Bonferroni-adjusted bounds from both methods;
- a challenger with a missing or zero common-four class recall, a common-four class mean
  recall drop above 3 points, or (five/six) a zero mean native recall cannot be "better";
- a target tie keeps four_class; Stage 6 keeps the Only-ESMC baseline four_class control
  unless an eligible, interval-supported candidate is best by mean common-four BA.

The tool never starts a fit and never opens held-out data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import pmm_v3_campaign as v3  # noqa: E402

SPEC_PATH = ROOT / "docs" / "campaigns" / "pmm_ion_metal_v3" / "assessment_spec.json"
FROZEN_SPEC_SHA256: str | None = None  # pinned when the A4 specification is frozen (log entry)

COMMON4 = ("Mn", "Cu", "Zn", "Class VIII")
NATIVE = {"four_class": COMMON4, "five_class": ("Mn", "Cu", "Zn", "Fe", "Class VIII"),
          "six_class": ("Mn", "Cu", "Zn", "Fe", "Co", "Ni")}
NATIVE_INDEX = {"four_class": {"MN": 0, "CU": 1, "ZN": 2, "FE": 3, "CO": 3, "NI": 3},
                "five_class": {"MN": 0, "CU": 1, "ZN": 2, "FE": 3, "CO": 4, "NI": 4},
                "six_class": {"MN": 0, "CU": 1, "ZN": 2, "FE": 3, "CO": 4, "NI": 5}}
COMMON4_INDEX = NATIVE_INDEX["four_class"]
PROBABILITY_TOLERANCE = 2.0e-6  # predictions are exported with eight decimals
RECONCILIATION_TOLERANCE = 1e-9
STAGE6_CONTROL = "only_esm__four_class__baseline"
# Step D: candidates screened against a matched control other than the family baseline.
SCREEN_CONTROLS = {"sitecountsangles": "sitenone"}
ROUND_CANDIDATES = {"D-A": ("meanagg", "resdrop01", "structlr", "gvpaux03", "esmdrop02"),
                    "D-B": ("sitecountsangles", "posnoise01", "outerdrop01", "vecnorm", "invsqrtw")}

SPEC_DEFAULTS: dict[str, Any] = {
    "spec_version": 1,
    "metric": "common-four balanced accuracy of the selected checkpoint; five/six probabilities summed into Class VIII",
    "primary_estimand": "mean over folds", "secondary_estimand": "pooled out-of-fold (descriptive)",
    "confidence": 0.95, "bootstrap_resamples": 10000, "bootstrap_seed": 42,
    "multiplicity": "bonferroni", "neutral_contrast_count": 6,
    "recall_drop_limit": 0.03, "tie_band": 0.002, "assessment_seed": 42,
    "screen_min_mean_delta": 0.015, "screen_min_recall_change": -0.03,
    "stage6_control": STAGE6_CONTROL,
    "pmm_published": {"cv": {"Mn": 90.3, "Cu": 62.9, "Zn": 73.8, "Class VIII": 73.3, "mean": 75.1},
                      "test_figure_2b": {"Mn": 88.6, "Cu": 59.4, "Zn": 65.9, "Class VIII": 57.5, "mean": 67.85},
                      "use": "descriptive context only; not a matched comparison; no interval for the difference"},
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_spec(path: Path = SPEC_PATH, *, expected_sha256: str | None = FROZEN_SPEC_SHA256) -> dict[str, Any]:
    """The frozen A4 specification; refused until its SHA-256 is pinned in this file."""
    require(expected_sha256 is not None, "The A4 assessment specification is not frozen yet")
    require(sha256(path) == expected_sha256, "Assessment specification differs from the frozen SHA-256")
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    # The frozen values must be the tested ones; a different value needs a new tested code drop.
    differing = sorted(key for key in SPEC_DEFAULTS if spec.get(key) != SPEC_DEFAULTS[key])
    require(not differing, f"Assessment specification differs from the tested rules in {differing}")
    return spec


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _outcome(bootstrap: tuple[float, float], t_interval: tuple[float, float]) -> str:
    if bootstrap[0] > 0 and t_interval[0] > 0:
        return "better"
    if bootstrap[1] < 0 and t_interval[1] < 0:
        return "worse"
    return "no clear difference"


def _single_method(interval: tuple[float, float]) -> str:
    return "better" if interval[0] > 0 else "worse" if interval[1] < 0 else "no clear difference"


def paired_intervals(differences, spec: dict[str, Any], *, n_contrasts: int = 1) -> dict[str, Any]:
    """Fold bootstrap and t intervals on paired fold differences, unadjusted and Bonferroni-adjusted."""
    d = np.asarray(differences, dtype=float)
    require(d.ndim == 1 and len(d) >= 2 and np.all(np.isfinite(d)), "Need two or more finite fold differences")
    require(n_contrasts >= 1, "n_contrasts must be positive")
    rng = np.random.default_rng(spec["bootstrap_seed"])
    means = d[rng.integers(0, len(d), size=(spec["bootstrap_resamples"], len(d)))].mean(axis=1)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(len(d)))

    def bounds(alpha: float) -> dict[str, Any]:
        q = float(stats.t.ppf(1 - alpha / 2, len(d) - 1))
        boot = (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))
        t_int = (mean - q * se, mean + q * se)
        return {"confidence": 1 - alpha, "bootstrap": list(boot), "t": list(t_int),
                "outcome": _outcome(boot, t_int),
                "methods_agree": _single_method(boot) == _single_method(t_int)}

    alpha = 1 - spec["confidence"]
    return {"fold_differences": d.tolist(), "n_folds": len(d), "mean_difference": mean,
            "t_degrees_of_freedom": len(d) - 1, "bootstrap_resamples": spec["bootstrap_resamples"],
            "bootstrap_seed": spec["bootstrap_seed"], "unadjusted": bounds(alpha),
            "adjusted": {**bounds(alpha / n_contrasts), "n_contrasts": n_contrasts, "method": "bonferroni"}}


# ---------------------------------------------------------------------------
# Units and predictions
# ---------------------------------------------------------------------------

def read_contract(paths: v3.V3Paths) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    """Frozen manifest and v3 fold membership, checked against the cohort and class-weight bindings."""
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    require(manifest.get("campaign_id") == v3.CAMPAIGN_ID, "Not a v3 campaign root")
    require(sha256(paths.cohort) == manifest["cohort"]["sha256"], "Frozen cohort changed")
    require(sha256(paths.fold_membership) == manifest["fold_set"]["fold_membership_sha256"], "Fold file changed")
    require(sha256(paths.fold_class_weights) == manifest["fold_class_weights_sha256"], "Class weights changed")
    with paths.cohort.open(encoding="utf-8", newline="") as handle:
        cohort = {row["source_uid"]: row for row in csv.DictReader(handle)}
    with paths.fold_membership.open(encoding="utf-8", newline="") as handle:
        membership = {row["source_uid"]: row for row in csv.DictReader(handle)}
    require(set(cohort) == set(membership), "Cohort and fold file cover different ions")
    for uid, row in membership.items():
        require(row["pdbid"] == cohort[uid]["group_id"] and row["native_element"] == cohort[uid]["native_element"]
                and row["physical_ion_id"] == cohort[uid]["physical_ion_id"], f"{uid}: fold row differs from cohort")
    return manifest, membership


def validate_rows(rows: list[dict[str, str]], membership: dict[str, dict[str, str]], *, fold: int, target: str,
                  seed: int, checkpoint_sha256: str, selected_epoch: int) -> None:
    """UID coverage, labels, probability vectors, argmax and probability-sum collapse of one unit."""
    expected = {uid for uid, row in membership.items() if int(row["fold"]) == fold}
    uids = [row["source_uid"] for row in rows]
    require(bool(rows) and len(set(uids)) == len(uids) and set(uids) == expected,
            f"Fold {fold}: predictions do not cover its ions exactly once")
    labels = NATIVE[target]
    groups = [(0,), (1,), (2,), tuple(range(3, len(labels)))]
    for row in rows:
        uid, member = row["source_uid"], membership[row["source_uid"]]
        require(int(row["fold"]) == fold and row["physical_ion_id"] == member["physical_ion_id"]
                and row["group_id"] == member["pdbid"] and row["native_element"] == member["native_element"],
                f"{uid}: prediction identity differs from the fold file")
        require(row["metal_label_scheme"] == v3.TARGET_SCHEMES[target], f"{uid}: label scheme differs")
        require(int(row["model_seed"]) == seed and row["checkpoint_sha256"] == checkpoint_sha256
                and int(row["selected_epoch"]) == selected_epoch, f"{uid}: seed, checkpoint or epoch differs")
        element = member["native_element"].upper()
        require(int(row["y_common4"]) == COMMON4_INDEX[element] and int(row["y_native"]) == NATIVE_INDEX[target][element],
                f"{uid}: label differs from the frozen element")

        def vector(prefix: str, names: tuple[str, ...]) -> list[float]:
            values = [float(row[f"{prefix}_{name.replace(' ', '_')}"]) for name in names]
            require(all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in values)
                    and abs(sum(values) - 1.0) <= PROBABILITY_TOLERANCE, f"{uid}: invalid {prefix} probabilities")
            return values

        native, common = vector("p_native", labels), vector("p_common4", COMMON4)
        for values, prediction in ((native, int(row["pred_native"])), (common, int(row["pred_common4"]))):
            require(0 <= prediction < len(values) and max(values) - values[prediction] <= PROBABILITY_TOLERANCE,
                    f"{uid}: prediction disagrees with its probabilities")
        collapsed = [sum(native[i] for i in group) for group in groups]
        require(all(abs(a - b) <= PROBABILITY_TOLERANCE for a, b in zip(common, collapsed)),
                f"{uid}: common-four probabilities are not the native probability sums")


def classification(y_true: list[int], y_pred: list[int], labels: tuple[str, ...]) -> dict[str, Any]:
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    np.add.at(matrix, (np.asarray(y_true), np.asarray(y_pred)), 1)
    support = matrix.sum(axis=1)
    recall = [float(matrix[i, i] / support[i]) if support[i] else None for i in range(len(labels))]
    present = [value for value in recall if value is not None]
    return {"balanced_accuracy": float(np.mean(present)) if present else None,
            "min_recall": float(min(present)) if present else None,
            "recall": dict(zip(labels, recall)), "support": dict(zip(labels, support.tolist())),
            "confusion_matrix": matrix.tolist()}


def prediction_metrics(rows: list[dict[str, str]], target: str) -> dict[str, Any]:
    common = classification([int(r["y_common4"]) for r in rows], [int(r["pred_common4"]) for r in rows], COMMON4)
    native = classification([int(r["y_native"]) for r in rows], [int(r["pred_native"]) for r in rows], NATIVE[target])
    return {"n": len(rows), "common4": common, "native": native}


def expected_identity(paths: v3.V3Paths, unit: v3.Unit, *, epochs: int) -> dict[str, Any]:
    """The identity the runner assigns to this unit under the recorded step-B setting."""
    settings = v3.read_execution_settings(paths)
    return v3.build_command(paths, unit, python_bin="python", train_dir=paths.root / "train",
                            esm_dir=paths.root / "esm", device="cuda", lane=0, epochs=epochs,
                            amp=bool(settings and settings["amp"]))[2]


def load_unit(paths: v3.V3Paths, unit: v3.Unit, statuses: dict[str, dict[str, Any]],
              membership: dict[str, dict[str, str]], *, epochs: int) -> dict[str, Any]:
    """A verified unit, or a record saying why it does not count."""
    settings = v3.read_execution_settings(paths) or {}
    run_name = settings.get("reuse", {}).get(unit.name, unit.name)  # a step-B run reused as a step-C unit
    record = statuses.get(run_name)
    reruns = len(v3.archived_attempts(paths, run_name))
    if record is None or record["status"] != "completed":
        status = "missing" if record is None else record["status"]
        final = status != "missing" and reruns >= v3.MAX_RERUNS
        return {"name": unit.name, "status": status, "archived_attempts": reruns,
                "final_failure": final, "note": ("failed after its one rerun" if final else
                                                 "rerun once (archive-failed, then run)" if status != "missing"
                                                 else "not run")}
    run_dir = Path(record["status_path"]).parent / "runs" / run_name
    recorded = record["identity"]
    expected = expected_identity(paths, unit, epochs=epochs)
    differing = sorted(k for k in set(expected) | set(recorded)
                       if k != "runner_sha256" and expected.get(k) != recorded.get(k))
    require(not differing, f"{unit.name}: recorded identity differs from the plan in {differing}")
    recipe = v3.resolve_recipe(unit.recipe)
    receipt = v3.verify_completed_unit(run_dir, recorded, recipe)
    v3.verify_independent_replay(run_dir, receipt)
    rows = _read_rows(run_dir / receipt["validation_predictions"]["path"])
    validate_rows(rows, membership, fold=unit.fold, target=unit.target, seed=unit.seed,
                  checkpoint_sha256=receipt["selected_checkpoint_sha256"], selected_epoch=receipt["selected_epoch"])
    metrics = prediction_metrics(rows, unit.target)
    for key, value in (("val_metal_collapsed4_balanced_acc", metrics["common4"]["balanced_accuracy"]),
                       ("val_metal_balanced_acc", metrics["native"]["balanced_accuracy"])):
        require(abs(float(receipt["metrics"][key]) - value) <= RECONCILIATION_TOLERANCE,
                f"{unit.name}: {key} from predictions differs from the receipt")
    with (run_dir / "epoch_metrics.csv").open(encoding="utf-8", newline="") as handle:
        history = [float(row["val_metal_collapsed4_balanced_acc"]) for row in csv.DictReader(handle)]
    return {"name": unit.name, "run_name": run_name, "status": "completed", "unit": unit, "rows": rows,
            "metrics": metrics,
            "selected_epoch": receipt["selected_epoch"], "selected_checkpoint": receipt["selected_checkpoint"],
            "descriptive_best_epoch": receipt.get("descriptive_best_epoch"),
            "terminal_history_common4_ba": history[-1] if history else None,
            "runner_sha256": recorded["runner_sha256"],
            "evidence": {"checkpoint_sha256": receipt["selected_checkpoint_sha256"],
                         "predictions_sha256": receipt["validation_predictions"]["sha256"]}}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def collect(paths: v3.V3Paths, units: list[v3.Unit], *, epochs: int = v3.PROFILE["epochs"]) -> dict[str, Any]:
    _, membership = read_contract(paths)
    statuses = v3.completed_units(paths)
    return {unit.name: load_unit(paths, unit, statuses, membership, epochs=epochs) for unit in units}


# ---------------------------------------------------------------------------
# Cells and contrasts
# ---------------------------------------------------------------------------

def cell_name(family: str, target: str, recipe: str) -> str:
    return f"{family}__{target}__{recipe}"


def cell_folds(collected: dict[str, Any], family: str, target: str, recipe: str, folds, seed: int) -> dict[int, Any]:
    """fold -> unit record for one cell; records that did not complete are returned as they are."""
    out = {}
    for fold in folds:
        name = v3.Unit(family, target, recipe, fold, seed).name
        out[fold] = collected.get(name, {"name": name, "status": "missing"})
    return out


def incomplete(folds: dict[int, Any]) -> list[str]:
    return [f"{unit['name']} ({unit['status']})" for unit in folds.values() if unit.get("status") != "completed"]


def _mean_recalls(folds: dict[int, Any], key: str, labels: tuple[str, ...], *, strict: bool) -> dict[str, Any]:
    out = {}
    for label in labels:
        values = [unit["metrics"][key]["recall"][label] for unit in folds.values()]
        present = [v for v in values if v is not None]
        out[label] = (None if (strict and len(present) != len(values)) or not present else float(np.mean(present)))
    return out


def summarize(folds: dict[int, Any], target: str) -> dict[str, Any]:
    require(not incomplete(folds), "Cannot summarize an incomplete cell")
    order = sorted(folds)
    scores = [folds[k]["metrics"]["common4"]["balanced_accuracy"] for k in order]
    summary = {"folds": order, "fold_common4_ba": scores, "mean_common4_ba": float(np.mean(scores)),
               "sd_common4_ba": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
               "worst_fold_common4_ba": float(min(scores)),
               "mean_min_recall": float(np.mean([folds[k]["metrics"]["common4"]["min_recall"] for k in order])),
               "mean_common4_recall": _mean_recalls(folds, "common4", COMMON4, strict=True),
               "mean_native_recall": _mean_recalls(folds, "native", NATIVE[target], strict=False)}
    if all("rows" in unit for unit in folds.values()):
        rows = [row for k in order for row in folds[k]["rows"]]
        pooled = prediction_metrics(rows, target)
        summary["pooled_oof"] = {"common4_ba": pooled["common4"]["balanced_accuracy"],
                                 "common4_recall": pooled["common4"]["recall"],
                                 "native_recall": pooled["native"]["recall"], "n": pooled["n"]}
    return summary


def compare(control: dict[int, Any], challenger: dict[int, Any], spec: dict[str, Any], *, challenger_target: str,
            n_contrasts: int = 1) -> dict[str, Any]:
    """Paired contrast on matched folds with the interval rule and the rare-class gates."""
    missing = incomplete(control) + incomplete(challenger)
    if missing or sorted(control) != sorted(challenger):
        return {"status": "incomplete", "missing": missing, "verdict": "incomplete"}
    order = sorted(control)
    a, b = summarize(control, "four_class"), summarize(challenger, challenger_target)
    differences = [challenger[k]["metrics"]["common4"]["balanced_accuracy"]
                   - control[k]["metrics"]["common4"]["balanced_accuracy"] for k in order]
    intervals = paired_intervals(differences, spec, n_contrasts=n_contrasts)
    drops = {label: (None if a["mean_common4_recall"][label] is None or b["mean_common4_recall"][label] is None
                     else a["mean_common4_recall"][label] - b["mean_common4_recall"][label]) for label in COMMON4}
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


# ---------------------------------------------------------------------------
# Step E: neutral target test, improvement check, Stage 6 selection
# ---------------------------------------------------------------------------

def neutral_target_test(collected: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """Per family five-vs-four and six-vs-four on all five folds, baseline recipe, seed 42."""
    seed, folds = spec["assessment_seed"], range(v3.N_FOLDS)
    contrasts, decisions = {}, {}
    for family in v3.FAMILIES:
        control = cell_folds(collected, family, "four_class", "baseline", folds, seed)
        better = []
        for target in ("five_class", "six_class"):
            result = compare(control, cell_folds(collected, family, target, "baseline", folds, seed), spec,
                             challenger_target=target, n_contrasts=spec["neutral_contrast_count"])
            contrasts[f"{family}: {target} vs four_class"] = result
            if result["verdict"] == "better":
                better.append((result["mean_difference"], target))
        if any(c["status"] == "incomplete" for k, c in contrasts.items() if k.startswith(family + ":")):
            decisions[family] = {"target": None, "reason": "incomplete grid"}
        elif not better:
            decisions[family] = {"target": "four_class", "reason": "no supported gain; a tie keeps four_class"}
        else:
            top = max(value for value, _ in better)
            tied = sorted(target for value, target in better if top - value <= spec["tie_band"])
            decisions[family] = {"target": tied[0], "reason": (
                "interval-supported gain" if len(better) == 1 else
                "both better; larger mean gain" if len(tied) == 1 else
                "both better within the tie band; five_class by selection convention "
                "(not evidence that five beats six)")}
    complete = all(c["status"] == "complete" for c in contrasts.values())
    across = (all(c["adjusted_claim_supported"] for c in contrasts.values()) if complete else None)
    return {"contrasts": contrasts, "family_target_decisions": decisions,
            "claim_all_families_five_and_six_better": across,
            "multiplicity": {"method": "bonferroni", "n_contrasts": spec["neutral_contrast_count"]},
            "label": "neutral target test (baseline recipe, frozen before any GPU run)"}


def improvement_check(collected: dict[str, Any], final_recipes: dict[str, str], spec: dict[str, Any]) -> dict[str, Any]:
    """Final recipe vs baseline four_class: folds 1-4 decide; fold 0 is the development fold."""
    seed, out = spec["assessment_seed"], {}
    for family, recipe in sorted(final_recipes.items()):
        control = cell_folds(collected, family, "four_class", "baseline", range(1, v3.N_FOLDS), seed)
        candidate = cell_folds(collected, family, "four_class", recipe, range(1, v3.N_FOLDS), seed)
        result = compare(control, candidate, spec, challenger_target="four_class", n_contrasts=len(final_recipes))
        dev_control = cell_folds(collected, family, "four_class", "baseline", (0,), seed)
        dev_candidate = cell_folds(collected, family, "four_class", recipe, (0,), seed)
        development = None
        if not incomplete(dev_control) + incomplete(dev_candidate):
            development = (dev_candidate[0]["metrics"]["common4"]["balanced_accuracy"]
                           - dev_control[0]["metrics"]["common4"]["balanced_accuracy"])
        if result["status"] == "incomplete":
            status = "incomplete"
        elif result["verdict"] == "better":
            status = "improvement interval-supported on folds 1-4"
        elif result["mean_difference"] > 0 and all(result["gates"].values()):
            status = "positive mean gain on folds 1-4, not interval-supported"
        else:
            status = "no positive mean gain on folds 1-4" if result["mean_difference"] <= 0 else "blocked by a recall gate"
        out[family] = {"recipe": recipe, "folds_1_4": result, "development_fold0_difference": development,
                       "status": status,
                       "improvement_claim_supported": status == "improvement interval-supported on folds 1-4",
                       "stage6_matched_pass": status in ("improvement interval-supported on folds 1-4",
                                                         "positive mean gain on folds 1-4, not interval-supported"),
                       "stage6_eligibility_note": "a positive mean gain permits Stage 6 consideration only; "
                                                  "it is not an improvement claim",
                       "label": "development-validation evidence (fold 0 influenced recipe selection), "
                                "not an unbiased estimate of generalization"}
    return out


def stage6_tie_key(row: dict[str, Any]) -> tuple:
    """Tie-breakers within the tie band: higher mean minimum recall, higher worst fold, lower SD,
    simpler family, baseline before an improved recipe, then the cell name."""
    return (-row["mean_min_recall"], -row["worst_fold_common4_ba"], row["sd_common4_ba"],
            v3.FAMILIES.index(row["family"]), row["recipe"] != "baseline", row["cell"])


def stage6_selection(collected: dict[str, Any], neutral: dict[str, Any], improvements: dict[str, Any],
                     spec: dict[str, Any]) -> dict[str, Any]:
    """Keep the control unless an eligible, interval-supported candidate is best by mean common-four BA."""
    seed, folds = spec["assessment_seed"], range(v3.N_FOLDS)
    cells = [(f, t, "baseline") for f in v3.FAMILIES for t in v3.TARGETS]
    cells += [(f, "four_class", item["recipe"]) for f, item in sorted(improvements.items())]
    control_name = spec["stage6_control"]
    control = cell_folds(collected, *control_name.split("__"), folds, seed)
    require(not incomplete(control), "Stage 6 needs the complete control cell")
    control_summary = summarize(control, "four_class")
    require(all(v not in (None, 0.0) for v in control_summary["mean_common4_recall"].values()),
            "The predeclared control has a missing or zero class recall; selection blocked")
    rows = []
    for family, target, recipe in cells:
        name = cell_name(family, target, recipe)
        cell = cell_folds(collected, family, target, recipe, folds, seed)
        if incomplete(cell):
            rows.append({"cell": name, "eligible": False, "reason": "incomplete: " + "; ".join(incomplete(cell))})
            continue
        summary = summarize(cell, target)
        if target != "four_class":
            matched = neutral["contrasts"][f"{family}: {target} vs four_class"]["verdict"] == "better"
        elif recipe != "baseline":
            matched = improvements[family]["stage6_matched_pass"]
        else:
            matched = True
        versus = None if name == control_name else compare(control, cell, spec, challenger_target=target)
        eligible = (name == control_name or (matched and versus["verdict"] == "better"))
        rows.append({"cell": name, "family": family, "target": target, "recipe": recipe,
                     "label": ("development-validation (fold 0 influenced recipe selection)"
                               if recipe != "baseline" else "baseline recipe frozen before any GPU run"),
                     "mean_common4_ba": summary["mean_common4_ba"], "mean_min_recall": summary["mean_min_recall"],
                     "worst_fold_common4_ba": summary["worst_fold_common4_ba"],
                     "sd_common4_ba": summary["sd_common4_ba"], "matched_pass": matched,
                     "versus_control": None if versus is None else versus["verdict"], "eligible": eligible})
    challengers = [row for row in rows if row["eligible"] and row["cell"] != control_name]
    if challengers:
        top = max(row["mean_common4_ba"] for row in challengers)
        tied = [row for row in challengers if top - row["mean_common4_ba"] <= spec["tie_band"]]
        chosen = min(tied, key=stage6_tie_key)
        selected, reason = chosen["cell"], ("best eligible candidate" if len(tied) == 1 else "tie-breakers")
    else:
        selected, reason = control_name, "no eligible, interval-supported replacement; the control is kept"
    return {"selected": selected, "reason": reason, "control": control_name, "tie_band": spec["tie_band"],
            "tie_breakers": ["mean minimum recall (higher)", "worst fold (higher)", "SD (lower)",
                             "simpler family (only_esm, only_gvp, gvp_late_fusion)", "baseline before improved recipe"],
            "candidates": sorted(rows, key=lambda row: (-(row.get("mean_common4_ba") or -1), row["cell"])),
            "refit": {"seed": spec["assessment_seed"], "checkpoint_rule": "terminal", "calibration": None,
                      "ensemble": False}}


def assess_step_e(collected: dict[str, Any], spec: dict[str, Any], *,
                  final_recipes: dict[str, str] | None = None) -> dict[str, Any]:
    neutral = neutral_target_test(collected, spec)
    improvements = improvement_check(collected, final_recipes or {}, spec)
    seed = spec["assessment_seed"]
    summaries = {}
    cells = [(f, t, "baseline") for f in v3.FAMILIES for t in v3.TARGETS]
    cells += [(f, "four_class", r) for f, r in sorted((final_recipes or {}).items())]
    for family, target, recipe in cells:
        folds = cell_folds(collected, family, target, recipe, range(v3.N_FOLDS), seed)
        summaries[cell_name(family, target, recipe)] = (
            summarize(folds, target) if not incomplete(folds) else {"incomplete": incomplete(folds)})
    complete = all(c["status"] == "complete" for c in neutral["contrasts"].values()) and all(
        item["status"] != "incomplete" for item in improvements.values())
    selection = stage6_selection(collected, neutral, improvements, spec) if complete else None
    return {"step": "E", "summaries": summaries, "neutral_target_test": neutral,
            "improvement_check": improvements, "stage6_selection": selection,
            "pmm_published": spec["pmm_published"], "held_out_test_accessed": False}


# ---------------------------------------------------------------------------
# Step D one-fold screen and combination rule
# ---------------------------------------------------------------------------

def screen_candidate(collected: dict[str, Any], family: str, recipe: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Screening gate on fold 0 with seeds 42 and 43 (not an improvement claim)."""
    control_recipe = SCREEN_CONTROLS.get(recipe, "baseline")
    deltas, changes, missing, final_failures = {}, {label: [] for label in COMMON4}, [], []
    for seed in v3.SEEDS:
        names = (v3.Unit(family, "four_class", control_recipe, 0, seed).name,
                 v3.Unit(family, "four_class", recipe, 0, seed).name)
        control, candidate = (collected.get(n, {"name": n, "status": "missing"}) for n in names)
        for unit in (control, candidate):
            if unit.get("status") != "completed":
                missing.append(f"{unit['name']} ({unit['status']})")
                if unit.get("final_failure") and unit is candidate:
                    final_failures.append(unit["name"])
        if missing:
            continue
        deltas[seed] = (candidate["metrics"]["common4"]["balanced_accuracy"]
                        - control["metrics"]["common4"]["balanced_accuracy"])
        for label in COMMON4:
            a, b = control["metrics"]["common4"]["recall"][label], candidate["metrics"]["common4"]["recall"][label]
            changes[label].append(None if a is None or b is None else b - a)
    if missing:
        return {"family": family, "recipe": recipe, "control": control_recipe, "passed": False,
                "status": ("did not pass the one-fold screen (a run failed after its rerun)" if final_failures
                           else "incomplete"), "missing": missing}
    mean_delta = float(np.mean(list(deltas.values())))
    recall_change = {label: (None if None in values else float(np.mean(values))) for label, values in changes.items()}
    checks = {"mean_delta_at_least_min": mean_delta >= spec["screen_min_mean_delta"] - 1e-12,
              "positive_delta_both_seeds": all(value > 0 for value in deltas.values()),
              "no_recall_change_below_limit": all(v is not None and v >= spec["screen_min_recall_change"] - 1e-12
                                                  for v in recall_change.values())}
    passed = all(checks.values())
    return {"family": family, "recipe": recipe, "control": control_recipe, "deltas": deltas,
            "mean_delta": mean_delta, "mean_recall_change": recall_change, "checks": checks, "passed": passed,
            "status": "passed the one-fold screen" if passed else "did not pass the one-fold screen"}


def screen_round(collected: dict[str, Any], round_name: str, spec: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for recipe in ROUND_CANDIDATES[round_name]:
        for family in v3.RECIPES[recipe]["families"]:
            out.append(screen_candidate(collected, family, recipe, spec))
    return out


def family_recipe_decision(screens: list[dict[str, Any]], family: str,
                           combo_screen: dict[str, Any] | None = None) -> dict[str, Any]:
    """Plan step D combination rule for one family."""
    passed = [s for s in screens if s["family"] == family and s["passed"]]
    if not passed:
        return {"family": family, "decision": "baseline", "recipe": "baseline"}
    best = max(passed, key=lambda s: (s["mean_delta"], s["recipe"]))
    if len(passed) == 1:
        return {"family": family, "decision": "single", "recipe": best["recipe"], "mean_delta": best["mean_delta"]}
    kept = {s["recipe"]: s for s in passed}
    for pair in v3.EXCLUSIVE_PAIRS:
        both = [kept[r] for r in sorted(pair) if r in kept]
        if len(both) == 2:
            del kept[min(both, key=lambda s: (s["mean_delta"], s["recipe"]))["recipe"]]
    if len(kept) == 1:
        return {"family": family, "decision": "single", "recipe": best["recipe"], "mean_delta": best["mean_delta"]}
    combo = "combo-" + "+".join(sorted(kept))
    if combo_screen is None:
        return {"family": family, "decision": "run combination", "recipe": combo,
                "best_single": best["recipe"], "best_single_mean_delta": best["mean_delta"]}
    require(combo_screen["recipe"] == combo and combo_screen["family"] == family, "Combination screen differs")
    adopt = combo_screen["passed"] and combo_screen["mean_delta"] > best["mean_delta"]
    return {"family": family, "decision": "combination" if adopt else "single",
            "recipe": combo if adopt else best["recipe"], "combination_mean_delta": combo_screen.get("mean_delta"),
            "best_single": best["recipe"], "best_single_mean_delta": best["mean_delta"]}


# ---------------------------------------------------------------------------
# Step C table (descriptive)
# ---------------------------------------------------------------------------

def assess_step_c(collected: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    rows, schedule = [], {}
    for unit in v3.step_units("C"):
        record = collected.get(unit.name, {"status": "missing"})
        if record.get("status") != "completed":
            rows.append({"unit": unit.name, "status": record.get("status")})
            continue
        metrics = record["metrics"]
        rows.append({"unit": unit.name, "status": "completed", "selected_epoch": record["selected_epoch"],
                     "selected_common4_ba": metrics["common4"]["balanced_accuracy"],
                     "terminal_history_common4_ba": record["terminal_history_common4_ba"],
                     "common4_recall": metrics["common4"]["recall"], "native_recall": metrics["native"]["recall"],
                     "descriptive_best_epoch": record.get("descriptive_best_epoch")})
    by_name = {row["unit"]: row for row in rows}
    for family in v3.FAMILIES:
        cosine = by_name[v3.Unit(family, "four_class", "baseline", 0, 42).name]
        fixed = by_name[v3.Unit(family, "four_class", "v2recipe", 0, 42).name]
        if cosine["status"] == fixed["status"] == "completed":
            schedule[family] = cosine["selected_common4_ba"] - fixed["terminal_history_common4_ba"]
    return {"step": "C", "fold": 0, "rows": rows,
            "schedule_effect_cosine_terminal_minus_fixed_terminal": schedule,
            "pmm_published": spec["pmm_published"],
            "label": "fold-0 values are exploratory context, not confirmation"}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, v3.Unit):
        return value.name
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def write_assessment(paths: v3.V3Paths, kind: str, result: dict[str, Any], collected: dict[str, Any],
                     spec_sha256: str) -> Path:
    """A new, never-overwritten assessment directory with its evidence bindings."""
    out = paths.root / "assessments" / f"{kind}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    out.mkdir(parents=True, exist_ok=False)
    evidence = {name: record.get("evidence") for name, record in sorted(collected.items())}
    payload = {"campaign_id": v3.CAMPAIGN_ID, "kind": kind, "assessor_sha256": sha256(Path(__file__)),
               "spec_sha256": spec_sha256, "manifest_sha256": sha256(paths.manifest),
               "runner_sha256": sorted({json.dumps(r["runner_sha256"], sort_keys=True)
                                        for r in collected.values() if r.get("status") == "completed"}),
               "unit_evidence": evidence, "result": _jsonable(result)}
    (out / "assessment.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--step", choices=("C", "D-A", "D-B", "D-combo", "E"), required=True)
    parser.add_argument("--family", choices=v3.GVP_FAMILIES, help="D-combo: the family")
    parser.add_argument("--recipe", help="D-combo: combo-a+b")
    parser.add_argument("--final-recipe", action="append", default=[], help="E: FAMILY=RECIPE, repeatable")
    args = parser.parse_args(argv)
    spec = load_spec()
    paths = v3.V3Paths(args.campaign_dir)
    if args.step == "C":
        collected = collect(paths, v3.step_units("C"))
        result = assess_step_c(collected, spec)
        regression = v3.completed_units(paths).get(v3.REGRESSION_NAME)
        result["regression_run"] = (None if regression is None else
                                    {"status": regression["status"], "gate": regression.get("gate")})
    elif args.step in ("D-A", "D-B"):
        units = v3.step_units("C")[:9] + v3.step_units("D-A")[:2] + v3.step_units(args.step)
        collected = collect(paths, list(dict.fromkeys(units)))
        screens = screen_round(collected, args.step, spec)
        result = {"step": args.step, "screens": screens,
                  "families": [family_recipe_decision(screens, family) for family in v3.GVP_FAMILIES]}
    elif args.step == "D-combo":
        units = v3.step_units("C")[:9] + v3.step_units("D-A")[:2] + v3.step_units(
            "D-combo", combo=args.recipe, family=args.family)
        collected = collect(paths, list(dict.fromkeys(units)))
        result = {"step": "D-combo", "screen": screen_candidate(collected, args.family, args.recipe, spec)}
    else:
        finals = dict(item.split("=", 1) for item in args.final_recipe)
        units = v3.step_units("C")[:9] + v3.step_units("E-neutral")
        if finals:
            units += v3.step_units("E-improvement", final_recipes=finals)
            units += [v3.Unit(f, "four_class", r, 0, 42) for f, r in finals.items()]
        collected = collect(paths, list(dict.fromkeys(units)))
        result = assess_step_e(collected, spec, final_recipes=finals)
    out = write_assessment(paths, args.step, result, collected, sha256(SPEC_PATH))
    print(json.dumps({"assessment": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"v3 assessment refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
