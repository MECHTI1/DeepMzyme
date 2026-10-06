#!/usr/bin/env python3
"""Early-stopping diagnostic on saved validation histories (CPU only; no fit, no replay, no held-out access).

This is an exploratory audit, separate from the official results. The v3 checkpoint rule stays the terminal
epoch of a 50-epoch cosine schedule; nothing here changes a selected checkpoint, the trainer or the frozen A4
rules. For every completed full-length run under DURABLE/lane*/runs it simulates one explicit stopping rule
sequentially on the logged per-epoch validation common-four balanced accuracy (BA):

  stop after PATIENCE consecutive epochs without a strict improvement (improvement > MIN_DELTA), at most
  MAX_EPOCHS epochs, and select the best epoch reached before the stop (the earliest one in an exact tie).

The simulation never looks at an epoch after its own stop. Each run also gets, as labelled retrospective
context: the minimum-validation-loss epoch, the best BA epoch of the full history, and whether a later epoch
would have beaten the simulated selection. Cosine runs (the campaign's rule) and fixed-learning-rate runs are
reported in separate sections.

Limits written into every report: a logged score of an epoch whose checkpoint was not saved is a hypothetical,
not a recovered or independently replayed model; stopping a 50-epoch cosine run at epoch t is not a t-epoch
cosine run, so the performance of a shorter schedule cannot be read from these histories; the selected epoch
is scored on the validation fold that selected it, so its margin over the terminal epoch is optimistic; one
fold, so the values are development evidence only.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

CAMPAIGN = Path("/media/mechti/Data1/DeepMzyme_Data/campaigns/pmm_ion_metal_v3")
METRIC = "val_metal_collapsed4_balanced_acc"
RECALLS = {"Mn": "val_metal_collapsed4_mn_recall", "Cu": "val_metal_collapsed4_cu_recall",
           "Zn": "val_metal_collapsed4_zn_recall", "Class VIII": "val_metal_collapsed4_class_viii_recall"}
LIMITS = ("A logged score of an epoch without a saved checkpoint is a hypothetical, not a recovered or replayed model.",
          "Stopping a 50-epoch cosine run at epoch t is not a t-epoch cosine run; a shorter schedule is not measured here.",
          "One fold: exploratory development evidence, not confirmation; official results keep the terminal epoch.",
          "An early validation-loss minimum does not establish the best stopping epoch for balanced accuracy.",
          "The simulated selection is scored on the validation fold that selects its epoch, so its margin over the "
          "terminal epoch includes selection optimism (TECH-027); it is not an estimate of the gain on new data.")


def simulate(values: list[float], *, patience: int, min_delta: float = 0.0) -> dict[str, Any]:
    """Sequential early stopping on a metric to maximize. Epochs are numbered from 1. An epoch improves only if
    it exceeds the best so far by more than ``min_delta``, so an exact tie keeps the earliest epoch."""
    if patience < 1 or not values:
        raise ValueError("patience must be positive and the history non-empty")
    best, best_epoch, waited = float("-inf"), 0, 0
    for epoch, value in enumerate(values, start=1):
        if value > best + min_delta:
            best, best_epoch, waited = value, epoch, 0
        else:
            waited += 1
            if waited >= patience:
                return {"stop_epoch": epoch, "stopped_early": epoch < len(values), "selected_epoch": best_epoch,
                        "selected_value": best}
    return {"stop_epoch": len(values), "stopped_early": False, "selected_epoch": best_epoch, "selected_value": best}


def read_history(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / "epoch_metrics.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: str | None) -> float | None:
    try:
        return None if value in (None, "", "nan") else float(value)
    except ValueError:
        return None


def saved_epochs(run_dir: Path, receipt: dict[str, Any], max_epochs: int) -> dict[int, str]:
    """Epoch -> checkpoint file that exists in the run directory (per-epoch files are normally not kept)."""
    out: dict[int, str] = {}
    for path in sorted(run_dir.glob("epoch_*_checkpoint.pt")):
        out[int(path.name.split("_")[1])] = path.name
    for name in ("terminal_model_checkpoint.pt", "last_model_checkpoint.pt"):
        if (run_dir / name).is_file():
            out.setdefault(max_epochs, name)
    if (run_dir / "best_model_checkpoint.pt").is_file():
        epoch = receipt.get("descriptive_best_epoch") or (
            receipt.get("selected_epoch") if receipt.get("selected_checkpoint") == "best_model_checkpoint.pt" else None)
        if epoch is not None:
            out.setdefault(int(epoch), "best_model_checkpoint.pt")
    return out


def analyse_run(run_dir: Path, *, patience: int, min_delta: float, max_epochs: int,
                unit_name: str | None = None) -> dict[str, Any]:
    """One run's simulated rule and retrospective context, or the reason it is not analysed."""
    config = json.loads((run_dir / "run_config.json").read_text())["config"]
    receipt = json.loads((run_dir / "selected_checkpoint.json").read_text())
    rows = read_history(run_dir)
    name = unit_name or run_dir.name
    base = {"run": run_dir.name, "unit": name, "lr_schedule": config.get("lr_schedule"),
            "checkpoint_rule": config.get("checkpoint_rule"), "seed": config.get("seed"),
            "fold": config.get("fold_index"), "metal_label_scheme": config.get("metal_label_scheme"),
            "epochs_planned": config.get("epochs"), "epochs_logged": len(rows)}
    values = [number(row.get(METRIC)) for row in rows]
    if receipt.get("fit_status") != "completed" or len(rows) != max_epochs or config.get("epochs") != max_epochs \
            or any(value is None for value in values):
        return {**base, "analysed": False,
                "reason": f"not a completed {max_epochs}-epoch history with the metric at every epoch"}
    simulated = simulate(values, patience=patience, min_delta=min_delta)
    selected, stop = simulated["selected_epoch"], simulated["stop_epoch"]
    losses = [number(row.get("val_loss")) for row in rows]
    terminal = values[-1]
    retrospective = max(range(len(values)), key=lambda k: (values[k], -k))  # earliest maximum
    later = [(k + 1, values[k]) for k in range(stop, len(values)) if values[k] > simulated["selected_value"]]
    best_later = max(later, key=lambda item: (item[1], -item[0])) if later else None
    recall_change = {}
    for label, column in RECALLS.items():
        a, b = number(rows[selected - 1].get(column)), number(rows[-1].get(column))
        recall_change[label] = None if a is None or b is None else a - b
    saved = saved_epochs(run_dir, receipt, max_epochs)
    return {**base, "analysed": True,
            "official_selected_epoch": receipt.get("selected_epoch"),
            "official_selected_common4_ba": (receipt.get("metrics") or {}).get(METRIC),
            "terminal_epoch": max_epochs, "terminal_common4_ba": terminal,
            "simulated_stop_epoch": stop, "simulated_stopped_early": simulated["stopped_early"],
            "simulated_selected_epoch": selected, "simulated_selected_common4_ba": simulated["selected_value"],
            "simulated_selected_minus_terminal": simulated["selected_value"] - terminal,
            "simulated_selected_minus_terminal_recall": recall_change,
            "epochs_saved": max_epochs - stop,
            "later_improvement_missed": bool(later),
            "best_later_epoch": None if best_later is None else best_later[0],
            "best_later_common4_ba": None if best_later is None else best_later[1],
            "missed_gain": None if best_later is None else best_later[1] - simulated["selected_value"],
            "simulated_selected_checkpoint_saved": selected in saved,
            "simulated_selected_checkpoint": saved.get(selected),
            "simulated_selected_score_status": (
                "saved checkpoint exists" if selected in saved else
                "logged hypothetical: this epoch's checkpoint was not saved (not a recovered or replayed model)"),
            "retrospective": {"label": "uses the full history; exploratory only",
                              "best_common4_ba_epoch": retrospective + 1, "best_common4_ba": values[retrospective],
                              "best_minus_terminal": values[retrospective] - terminal,
                              "min_val_loss_epoch": (None if any(v is None for v in losses) else
                                                     min(range(len(losses)), key=lambda k: (losses[k], k)) + 1),
                              "min_val_loss": None if any(v is None for v in losses) else min(losses),
                              "terminal_val_loss": losses[-1]}}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n_runs": 0}
    gains = [row["simulated_selected_minus_terminal"] for row in rows]
    return {"n_runs": len(rows), "n_stopped_early": sum(row["simulated_stopped_early"] for row in rows),
            "mean_selected_minus_terminal": statistics.fmean(gains), "median_selected_minus_terminal": statistics.median(gains),
            "min_selected_minus_terminal": min(gains), "max_selected_minus_terminal": max(gains),
            "n_selected_above_terminal": sum(gain > 0 for gain in gains),
            "n_selected_below_terminal": sum(gain < 0 for gain in gains),
            "mean_epochs_saved": statistics.fmean(row["epochs_saved"] for row in rows),
            "n_later_improvement_missed": sum(row["later_improvement_missed"] for row in rows),
            "n_selected_checkpoint_saved": sum(row["simulated_selected_checkpoint_saved"] for row in rows),
            "median_simulated_selected_epoch": statistics.median(row["simulated_selected_epoch"] for row in rows),
            "median_retrospective_best_epoch": statistics.median(row["retrospective"]["best_common4_ba_epoch"] for row in rows),
            "median_min_val_loss_epoch": statistics.median(
                row["retrospective"]["min_val_loss_epoch"] for row in rows
                if row["retrospective"]["min_val_loss_epoch"] is not None)}


def paired_by_seed(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Cells that have several seeds (step D): the per-seed values next to each other."""
    cells: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        cells.setdefault(row["unit"].rsplit("__seed", 1)[0], []).append(row)
    return {cell: {"seeds": [r["seed"] for r in members],
                   "simulated_selected_epoch": [r["simulated_selected_epoch"] for r in members],
                   "simulated_selected_minus_terminal": [r["simulated_selected_minus_terminal"] for r in members],
                   "mean_simulated_selected_minus_terminal": statistics.fmean(
                       r["simulated_selected_minus_terminal"] for r in members)}
            for cell, members in sorted(cells.items()) if len(members) > 1}


def run_audit(durable: Path, *, patience: int, min_delta: float, max_epochs: int,
              reuse: dict[str, str] | None = None) -> dict[str, Any]:
    names = {run: unit for unit, run in (reuse or {}).items()}
    analysed, skipped = [], []
    for run_dir in sorted(durable.glob("lane*/runs/*")):
        if not (run_dir.is_dir() and (run_dir / "epoch_metrics.csv").is_file()
                and (run_dir / "selected_checkpoint.json").is_file() and (run_dir / "run_config.json").is_file()):
            continue
        row = analyse_run(run_dir, patience=patience, min_delta=min_delta, max_epochs=max_epochs,
                          unit_name=names.get(run_dir.name))
        (analysed if row["analysed"] else skipped).append(row)
    cosine = [row for row in analysed if row["lr_schedule"] == "cosine"]
    other = [row for row in analysed if row["lr_schedule"] != "cosine"]
    return {"audit": "v3 early-stopping diagnostic (exploratory; official checkpoints unchanged)",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "durable": str(durable),
            "rule": {"monitor": METRIC, "mode": "max", "patience": patience, "min_delta": min_delta,
                     "max_epochs": max_epochs, "tie": "earliest epoch", "selection": "best epoch before the stop",
                     "status": "exploratory candidate, not a tuned or adopted rule"},
            "limits": list(LIMITS),
            "cosine_campaign_runs": {"summary": summarize(cosine), "by_cell_with_several_seeds": paired_by_seed(cosine),
                                     "runs": cosine},
            "fixed_lr_diagnostic_runs": {"note": "kept apart from the cosine campaign results",
                                         "summary": summarize(other), "runs": other},
            "not_analysed": skipped, "held_out_access": False}


def table(rows: list[dict[str, Any]]) -> str:
    lines = ["unit | min-loss ep | retro best ep (BA) | stop ep | selected ep (BA) | terminal BA | sel-term | saved ep | missed later"]
    for row in rows:
        retro = row["retrospective"]
        lines.append(" | ".join([
            row["unit"], str(retro["min_val_loss_epoch"]),
            f"{retro['best_common4_ba_epoch']} ({100 * retro['best_common4_ba']:.1f})", str(row["simulated_stop_epoch"]),
            f"{row['simulated_selected_epoch']} ({100 * row['simulated_selected_common4_ba']:.1f})",
            f"{100 * row['terminal_common4_ba']:.1f}", f"{100 * row['simulated_selected_minus_terminal']:+.1f}",
            str(row["epochs_saved"]),
            "no" if not row["later_improvement_missed"] else
            f"epoch {row['best_later_epoch']} (+{100 * row['missed_gain']:.1f})"]))
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", type=Path, default=CAMPAIGN, help="workstation campaign data directory")
    parser.add_argument("--durable", type=Path, help="default: CAMPAIGN/durable")
    parser.add_argument("--settings", type=Path, help="execution_settings.json (names the reused step-B run)")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args(argv)
    durable = args.durable or args.campaign / "durable"
    settings = args.settings
    if settings is None:
        copies = sorted(args.campaign.glob("step_*_evidence/evidence_*/campaign/execution_settings.json"),
                        key=lambda path: path.parent.parent.name)
        settings = copies[-1] if copies else None
    reuse = json.loads(settings.read_text()).get("reuse", {}) if settings else {}
    report = run_audit(durable, patience=args.patience, min_delta=args.min_delta, max_epochs=args.max_epochs, reuse=reuse)
    out_dir = args.out_dir or args.campaign / "audits" / f"early_stopping_audit_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "early_stopping_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    text = ("Cosine campaign runs (official rule: terminal epoch)\n" + table(report["cosine_campaign_runs"]["runs"])
            + "\n\nFixed-learning-rate diagnostic runs (separate)\n" + table(report["fixed_lr_diagnostic_runs"]["runs"])
            + "\n\nLimits\n- " + "\n- ".join(LIMITS) + "\n")
    (out_dir / "early_stopping_audit.txt").write_text(text)
    print(text)
    print(json.dumps({"report": str(out_dir / "early_stopping_audit.json"),
                      "cosine_summary": report["cosine_campaign_runs"]["summary"],
                      "fixed_lr_summary": report["fixed_lr_diagnostic_runs"]["summary"],
                      "not_analysed": [row["run"] for row in report["not_analysed"]]}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"early-stopping audit refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
