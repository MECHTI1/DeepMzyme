#!/usr/bin/env python3
"""Logged-epoch contrast of completed v3 runs on explicit folds (CPU only; descriptive; no fit, no replay,
no held-out access).

For each requested cell (family, target, recipe) and each requested fold of one model seed, this reads the
logged per-epoch validation history of the one completed run with exactly that identity under
DURABLE/lane*/runs, and reports the common-four balanced accuracy (BA), the per-class common-four recalls and
the common-four confusion matrix at two logged epochs (default: epoch 10 and the terminal epoch 50), the
per-fold differences and their means over exactly the requested folds.

Strictness (every violation stops the audit with a readable error; nothing is skipped silently):
- every requested (cell, fold, seed) has exactly one run directory across the lanes;
- the run's recorded identity (selected_checkpoint.json `campaign_run_identity`) matches the requested family,
  target, recipe, fold and seed; the fit completed with the terminal checkpoint at the planned final epoch;
- the validation history holds each epoch 1..planned exactly once, with the requested metrics present;
- the terminal BA equals the run's selected-checkpoint record and, when an assessment copy is passed, the
  assessment's fold value for that cell (tolerance 1e-9).
Aggregates are keyed by identity and computed over the requested folds in the requested order, so the lane
or directory order cannot change a result.

Limits written into every report: a logged epoch other than the terminal one has no saved checkpoint, so its
score is a hypothetical, not a recovered or independently replayed model; epoch t inside a 50-epoch cosine run
is a different treatment from a complete t-epoch cosine schedule; the folds are validation folds already used
for selection decisions; one seed; descriptive development evidence that selects, ranks or promotes nothing.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

CAMPAIGN = Path("/media/mechti/Data1/DeepMzyme_Data/campaigns/pmm_ion_metal_v3")
COMMON4 = ("Mn", "Cu", "Zn", "Class VIII")
BA = "val_metal_collapsed4_balanced_acc"
RECALLS = "val_metal_collapsed4_per_class_recall"
CONFUSION = "val_metal_collapsed4_confusion_matrix"
SUPPORT = "val_metal_collapsed4_per_class_support"
TOLERANCE = 1e-9
DEFAULT_CELLS = ("gvp_late_fusion:four_class:baseline", "only_esm:four_class:baseline",
                 "only_gvp:four_class:baseline", "gvp_late_fusion:four_class:headdrop03",
                 "only_gvp:four_class:meanagg")
LIMITS = ("A logged epoch other than the terminal one has no saved checkpoint: its score is a hypothetical, not a "
          "recovered or independently replayed model.",
          "Epoch t inside a 50-epoch cosine run is a different treatment from a complete t-epoch cosine schedule; "
          "these values do not estimate the result of a shorter schedule.",
          "The folds are validation folds already used for selection decisions; one model seed; descriptive "
          "development evidence only; nothing here selects, ranks or promotes a configuration.")


class AuditError(ValueError):
    """A completeness, uniqueness, identity or reconciliation check failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def unit_name(family: str, target: str, recipe: str, fold: int, seed: int) -> str:
    return f"{family}__{target}__{recipe}__fold{fold}__seed{seed}"


def parse_cell(text: str) -> tuple[str, str, str]:
    parts = text.split(":")
    require(len(parts) == 3 and all(parts), f"Cell {text!r} must be family:target:recipe")
    return parts[0], parts[1], parts[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_run(durable: Path, name: str) -> Path:
    matches = sorted(path for path in durable.glob(f"lane*/runs/{name}") if path.is_dir())
    require(matches, f"Missing run: {name} (no directory under {durable}/lane*/runs)")
    require(len(matches) == 1, f"Duplicate run directories for {name}: {[str(p) for p in matches]}")
    return matches[0]


def read_history(run_dir: Path, *, planned: int, epochs: tuple[int, ...]) -> dict[int, dict[str, str]]:
    path = run_dir / "val_metrics.csv"
    require(path.is_file(), f"{run_dir.name}: no val_metrics.csv")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_epoch: dict[int, dict[str, str]] = {}
    for row in rows:
        try:
            epoch = int(row["epoch"])
        except (KeyError, TypeError, ValueError):
            raise AuditError(f"{run_dir.name}: a history row has no integer epoch") from None
        require(epoch not in by_epoch, f"{run_dir.name}: duplicate history record for epoch {epoch}")
        by_epoch[epoch] = row
    require(sorted(by_epoch) == list(range(1, planned + 1)),
            f"{run_dir.name}: history epochs are not exactly 1..{planned} ({len(by_epoch)} unique epochs logged)")
    for epoch in epochs:
        require(epoch in by_epoch, f"{run_dir.name}: epoch {epoch} is outside the {planned}-epoch history")
        for column in (BA, RECALLS, CONFUSION):
            require(by_epoch[epoch].get(column) not in (None, "", "nan"),
                    f"{run_dir.name}: epoch {epoch} has no {column}")
    return by_epoch


def epoch_values(row: dict[str, str], run: str, epoch: int) -> dict[str, Any]:
    try:
        recalls = ast.literal_eval(row[RECALLS])
        confusion = ast.literal_eval(row[CONFUSION])
        support = ast.literal_eval(row[SUPPORT]) if row.get(SUPPORT) else None
    except (ValueError, SyntaxError):
        raise AuditError(f"{run}: epoch {epoch} has an unreadable recall or confusion record") from None
    require(sorted(recalls) == sorted(COMMON4), f"{run}: epoch {epoch} recalls are not the common four")
    require(len(confusion) == 4 and all(len(r) == 4 for r in confusion),
            f"{run}: epoch {epoch} confusion matrix is not 4 x 4")
    return {"ba": float(row[BA]), "recall": {k: float(recalls[k]) for k in COMMON4},
            "confusion": confusion, "support": support}


def load_run(durable: Path, family: str, target: str, recipe: str, fold: int, seed: int,
             epochs: tuple[int, ...]) -> dict[str, Any]:
    name = unit_name(family, target, recipe, fold, seed)
    run_dir = find_run(durable, name)
    receipt_path = run_dir / "selected_checkpoint.json"
    require(receipt_path.is_file(), f"{name}: no selected_checkpoint.json")
    receipt = json.loads(receipt_path.read_text())
    identity = receipt.get("campaign_run_identity") or {}
    recorded = (identity.get("family"), identity.get("target_scheme"), identity.get("recipe"),
                identity.get("fold"), identity.get("model_seed"))
    require(recorded == (family, target, recipe, fold, seed),
            f"{name}: recorded identity {recorded} differs from the requested unit")
    planned = receipt.get("planned_epochs")
    require(receipt.get("fit_status") == "completed", f"{name}: fit_status is {receipt.get('fit_status')!r}")
    require(receipt.get("checkpoint_rule") == "terminal", f"{name}: checkpoint rule is not terminal")
    require(isinstance(planned, int) and receipt.get("completed_epochs") == planned == identity.get("epochs"),
            f"{name}: completed/planned epochs disagree ({receipt.get('completed_epochs')}, {planned})")
    require(max(epochs) <= planned, f"{name}: requested epoch {max(epochs)} beyond the planned {planned}")
    history = read_history(run_dir, planned=planned, epochs=tuple(sorted(set(epochs) | {planned})))
    terminal_ba = float(history[planned][BA])
    recorded_ba = (receipt.get("metrics") or {}).get(BA)
    require(recorded_ba is not None and abs(terminal_ba - float(recorded_ba)) <= TOLERANCE,
            f"{name}: terminal history BA {terminal_ba} differs from the selected-checkpoint record {recorded_ba}")
    return {"unit": name, "lane": run_dir.parent.parent.name, "fold": fold,
            "val_metrics_sha256": sha256(run_dir / "val_metrics.csv"),
            "resolved_config_sha256": identity.get("resolved_config_sha256"),
            "terminal_epoch": planned, "terminal_ba": terminal_ba,
            "epochs": {epoch: epoch_values(history[epoch], name, epoch) for epoch in epochs}}


def reconcile(cell: str, runs: list[dict[str, Any]], assessment: dict[str, Any]) -> None:
    summary = ((assessment.get("result") or {}).get("summaries") or {}).get(cell)
    require(summary is not None and "fold_common4_ba" in summary,
            f"{cell}: the assessment has no complete summary to reconcile with")
    by_fold = dict(zip(summary["folds"], summary["fold_common4_ba"]))
    for run in runs:
        require(run["fold"] in by_fold, f"{run['unit']}: fold {run['fold']} is not in the assessment summary")
        require(abs(run["terminal_ba"] - by_fold[run["fold"]]) <= TOLERANCE,
                f"{run['unit']}: terminal BA {run['terminal_ba']} differs from the assessment {by_fold[run['fold']]}")


def contrast(runs: list[dict[str, Any]], early: int, late: int) -> dict[str, Any]:
    """Per-fold and mean values over exactly the given runs (already in the requested fold order)."""
    def mean(values):
        return statistics.fmean(values)
    per_fold = []
    for run in runs:
        a, b = run["epochs"][early], run["epochs"][late]
        per_fold.append({
            "fold": run["fold"], "unit": run["unit"], "lane": run["lane"],
            f"ba_epoch{early}": a["ba"], f"ba_epoch{late}": b["ba"], "ba_difference": a["ba"] - b["ba"],
            "recall_difference": {k: a["recall"][k] - b["recall"][k] for k in COMMON4},
            "class_viii_to_mn": {early: a["confusion"][3][0], late: b["confusion"][3][0]},
            "mn_to_class_viii": {early: a["confusion"][0][3], late: b["confusion"][0][3]},
            f"confusion_epoch{early}": a["confusion"], f"confusion_epoch{late}": b["confusion"]})
    return {
        "folds": [row["fold"] for row in per_fold],
        f"mean_ba_epoch{early}": mean(r[f"ba_epoch{early}"] for r in per_fold),
        f"mean_ba_epoch{late}": mean(r[f"ba_epoch{late}"] for r in per_fold),
        "mean_ba_difference": mean(r["ba_difference"] for r in per_fold),
        "mean_recall_difference": {k: mean(r["recall_difference"][k] for r in per_fold) for k in COMMON4},
        "per_fold": per_fold}


def run_audit(durable: Path, cells: list[tuple[str, str, str]], *, folds: list[int], seed: int,
              early: int, late: int, assessment: dict[str, Any] | None = None) -> dict[str, Any]:
    require(folds and len(set(folds)) == len(folds), f"Folds must be distinct and non-empty: {folds}")
    require(0 < early < late, f"Epochs must satisfy 0 < early < late ({early}, {late})")
    results = {}
    for family, target, recipe in cells:
        cell = f"{family}__{target}__{recipe}"
        require(cell not in results, f"Cell {cell} requested twice")
        runs = [load_run(durable, family, target, recipe, fold, seed, (early, late)) for fold in folds]
        for run in runs:
            require(run["terminal_epoch"] == late,
                    f"{run['unit']}: the late epoch {late} is not the terminal epoch {run['terminal_epoch']}")
        if assessment is not None:
            reconcile(cell, runs, assessment)
        results[cell] = contrast(runs, early, late)
    return {"kind": "v3 logged-epoch contrast", "folds": folds, "seed": seed, "early_epoch": early,
            "late_epoch": late, "cells": results, "limits": list(LIMITS),
            "assessment_reconciled": assessment is not None, "held_out_test_accessed": False}


def table(report: dict[str, Any]) -> str:
    early, late = report["early_epoch"], report["late_epoch"]
    lines = [f"Folds {report['folds']}, seed {report['seed']}; common-four BA in points; "
             f"difference = epoch {early} minus epoch {late}.", "",
             f"| Cell | epoch {late} | epoch {early} | difference | per-fold differences | "
             "mean recall difference Mn / Cu / Zn / Class VIII |", "|---|---:|---:|---:|---|---|"]
    for cell, item in report["cells"].items():
        folds = " / ".join(f"{100 * r['ba_difference']:+.3f}" for r in item["per_fold"])
        recall = " / ".join(f"{100 * item['mean_recall_difference'][k]:+.2f}" for k in COMMON4)
        lines.append(f"| {cell} | {100 * item[f'mean_ba_epoch{late}']:.4f} | {100 * item[f'mean_ba_epoch{early}']:.4f}"
                     f" | {100 * item['mean_ba_difference']:+.4f} | {folds} | {recall} |")
    lines += ["", f"Class VIII predicted as Mn / Mn predicted as Class VIII (epoch {early} -> {late}):", ""]
    for cell, item in report["cells"].items():
        parts = [f"fold {r['fold']}: {r['class_viii_to_mn'][early]}->{r['class_viii_to_mn'][late]} / "
                 f"{r['mn_to_class_viii'][early]}->{r['mn_to_class_viii'][late]}" for r in item["per_fold"]]
        lines.append(f"- {cell}: " + "; ".join(parts))
    lines += ["", "Contributing runs (unit, lane, val_metrics SHA-256):", ""]
    for item in report["cells"].values():
        for r in item["per_fold"]:
            lines.append(f"- {r['unit']} ({r['lane']})")
    lines += ["", "Limits:", ""] + [f"- {text}" for text in report["limits"]]
    return "\n".join(lines) + "\n"


def jsonable(report: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(report, default=str))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", type=Path, default=CAMPAIGN, help="workstation campaign data directory")
    parser.add_argument("--durable", type=Path, help="default: CAMPAIGN/durable")
    parser.add_argument("--cell", action="append", help="family:target:recipe (repeatable); default: the five "
                        "four-class step E cells")
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-epoch", type=int, default=10)
    parser.add_argument("--late-epoch", type=int, default=50, help="must be the terminal epoch")
    parser.add_argument("--assessment", type=Path, help="assessment.json copy to reconcile terminal values with")
    parser.add_argument("--out-dir", type=Path, help="write report.json, report.md and inputs.json here")
    args = parser.parse_args(argv)
    durable = args.durable or args.campaign / "durable"
    cells = [parse_cell(text) for text in (args.cell or DEFAULT_CELLS)]
    assessment = json.loads(args.assessment.read_text()) if args.assessment else None
    try:
        report = run_audit(durable, cells, folds=args.folds, seed=args.seed, early=args.early_epoch,
                           late=args.late_epoch, assessment=assessment)
    except AuditError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    if args.assessment:
        report["assessment"] = {"path": str(args.assessment), "sha256": sha256(args.assessment)}
    text = table(report)
    print(text)
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=False)
        (args.out_dir / "report.json").write_text(json.dumps(jsonable(report), indent=2, sort_keys=True) + "\n")
        (args.out_dir / "report.md").write_text(text)
        (args.out_dir / "inputs.json").write_text(json.dumps({
            "argv": sys.argv if argv is None else list(argv), "durable": str(durable),
            "script_sha256": sha256(Path(__file__)), "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
