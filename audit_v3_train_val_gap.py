#!/usr/bin/env python3
"""Train/validation gap of the v3 step D fold-0 runs (CPU only; descriptive; no fit, no replay, no held-out access).

This is an exploratory companion of the early-stopping audit. The v3 checkpoint rule stays the terminal epoch
of the 50-epoch cosine schedule and the step D screen stays the frozen A4 gate on validation BA; nothing here
selects, ranks or promotes anything. For every completed 50-epoch four-class fold-0 run of Only-GVP and late
fusion under DURABLE/lane*/runs it reads the terminal epoch's training-set and validation-set common-four
balanced accuracy (BA; the trainer evaluates the training ions in eval mode every tenth epoch) and losses,
defines

  gap = training BA - validation BA (points, terminal epoch),

pairs every candidate with its matched same-seed control (the family baseline, or `sitenone` for
`sitecountsangles`, as the assessor does), and reports per recipe the mean over seeds of the gap, the gap
change against the control, and the training and validation changes that produce it: by construction
d gap = d training BA - d validation BA. The screen outcome of each recipe is copied from the assessment file
passed in (the launcher's latest copy by default). Runs that are skipped, and the reason, are listed.

Limits written into every report: fold 0 only, two seeds, with the measured control seed spread reported;
where the training BA is saturated (near 100) the gap change simply repeats the validation change, so only the
training-BA change says whether a candidate fits the training fold less; the validation value is the official
screen quantity and is not re-decided here; a screen pass is a one-fold screen, not an improvement claim; the
losses are not on one scale across candidates that change the training objective, and the epoch training loss
is a training-mode running mean while the training BA is an eval-mode pass.
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
FAMILIES = ("only_gvp", "gvp_late_fusion")
TRAIN_BA = "train_metal_collapsed4_balanced_acc"
VAL_BA = "val_metal_collapsed4_balanced_acc"
TRAIN_MIN_RECALL = "train_metal_collapsed4_min_recall"
VAL_MIN_RECALL = "val_metal_collapsed4_min_recall"
# Candidates screened against a matched control other than the family baseline; the test keeps this equal to
# the frozen runner's SCREEN_CONTROLS. A control-only recipe is never a candidate.
SCREEN_CONTROLS = {"sitecountsangles": "sitenone"}
CONTROL_RECIPES = ("baseline", *SCREEN_CONTROLS.values())
SATURATED_TRAIN_BA = 97.0  # points; above this, d gap carries no information beyond d validation BA
LIMITS = ("Fold 0 only and two seeds: exploratory development evidence. The per-family validation-BA spread of the "
          "two control seeds is reported with the controls; single-seed deltas inside that spread are noise.",
          "gap = training BA - validation BA, so d gap = d training BA - d validation BA. Where the training BA is "
          f"saturated (above {SATURATED_TRAIN_BA:.0f} for candidate and control) the gap change repeats the "
          "validation change and says nothing about memorization; only d training BA measures whether the "
          "candidate fits the training fold less. A smaller gap is not evidence of better generalization.",
          "The validation BA is the official screen quantity and is not re-decided here; a screen status copied "
          "from the assessment is a one-fold screen, not an improvement claim; the improvement check is folds 1-4 "
          "under A4 (step E).",
          "Losses are class-weighted cross-entropy means and are not on one scale across candidates that change "
          "the training objective: changed class weights (invsqrtw) move both losses, the training-only GVP "
          "auxiliary term (gvpaux03) moves the training loss alone.",
          "The epoch training loss is the running mean over the epoch's minibatches in training mode (dropout and "
          "modality dropout active), while the training BA is an eval-mode pass, so a dropout candidate raises the "
          "training loss without necessarily fitting the training ions less; compare BA values, not losses, for "
          "those candidates.",
          "The minimum-validation-loss epoch is retrospective context only; an early loss minimum does not "
          "establish a stopping epoch for balanced accuracy (the early-stopping audit owns that question).",
          "No checkpoint, rule or assessment is changed; the terminal epoch stays the official checkpoint.")


def number(value: str | None) -> float | None:
    try:
        return None if value in (None, "", "nan") else float(value)
    except ValueError:
        return None


def parse_unit(name: str) -> dict[str, str] | None:
    """family__target__recipe__foldK__seedN of a campaign unit (probe and regression names do not parse)."""
    parts = name.split("__")
    if len(parts) != 5 or parts[0] not in FAMILIES or not parts[3].startswith("fold") or not parts[4].startswith("seed"):
        return None
    return {"family": parts[0], "target": parts[1], "recipe": parts[2], "fold": parts[3], "seed": parts[4]}


def terminal_values(run_dir: Path, *, max_epochs: int, parsed: dict[str, str]) -> dict[str, Any] | str:
    """Terminal-epoch training and validation values of one completed run, or the reason it is skipped."""
    config = json.loads((run_dir / "run_config.json").read_text())["config"]
    receipt = json.loads((run_dir / "selected_checkpoint.json").read_text())
    with (run_dir / "epoch_metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if receipt.get("fit_status") != "completed":
        return f"fit_status is {receipt.get('fit_status')!r}, not completed"
    if config.get("lr_schedule") != "cosine":
        return f"lr_schedule is {config.get('lr_schedule')!r}, not the campaign's cosine rule"
    if f"seed{config.get('seed')}" != parsed["seed"] or f"fold{config.get('fold_index')}" != parsed["fold"]:
        return (f"run_config seed/fold ({config.get('seed')}, {config.get('fold_index')}) disagree with the unit name "
                f"({parsed['seed']}, {parsed['fold']})")
    if config.get("epochs") != max_epochs or len(rows) != max_epochs or number(rows[-1].get("epoch")) != max_epochs:
        return f"not a completed {max_epochs}-epoch history (planned {config.get('epochs')}, {len(rows)} rows logged)"
    last = rows[-1]
    values = {"train_ba": number(last.get(TRAIN_BA)), "val_ba": number(last.get(VAL_BA)),
              "train_loss": number(last.get("train_loss")), "val_loss": number(last.get("val_loss")),
              "train_min_recall": number(last.get(TRAIN_MIN_RECALL)), "val_min_recall": number(last.get(VAL_MIN_RECALL))}
    if values["train_ba"] is None or values["val_ba"] is None:
        return "the terminal epoch has no training-set or validation-set common-four BA"
    losses = [number(row.get("val_loss")) for row in rows]
    values["min_val_loss_epoch"] = (None if any(v is None for v in losses)
                                    else min(range(len(losses)), key=lambda k: (losses[k], k)) + 1)
    return values


def screen_outcomes(assessment: dict[str, Any] | None) -> dict[tuple[str, str], str]:
    """(family, recipe) -> the assessor's screen status, from an assessment copy."""
    out: dict[tuple[str, str], str] = {}
    for round_result in ((assessment or {}).get("result") or {}).get("rounds", {}).values():
        for screen in round_result.get("screens") or []:
            out[(screen["family"], screen["recipe"])] = screen.get("status") or ("passed" if screen.get("passed") else "")
    return out


def pct(value: float | None) -> float | None:
    return None if value is None else 100 * value


def run_audit(durable: Path, *, max_epochs: int, reuse: dict[str, str] | None = None,
              assessment: dict[str, Any] | None = None, target: str = "four_class", fold: str = "fold0") -> dict[str, Any]:
    names = {run: unit for unit, run in (reuse or {}).items()}
    runs: dict[str, dict[str, Any]] = {}
    skipped: list[dict[str, str]] = []
    for run_dir in sorted(durable.glob("lane*/runs/*")):
        if not (run_dir.is_dir() and (run_dir / "epoch_metrics.csv").is_file()
                and (run_dir / "selected_checkpoint.json").is_file() and (run_dir / "run_config.json").is_file()):
            continue
        unit = names.get(run_dir.name, run_dir.name)
        parsed = parse_unit(unit)
        if parsed is None:
            skipped.append({"run": run_dir.name, "reason": "not a campaign unit of Only-GVP or late fusion"})
            continue
        if parsed["target"] != target or parsed["fold"] != fold:
            skipped.append({"run": run_dir.name, "reason": f"not a {target} {fold} unit"})
            continue
        values = terminal_values(run_dir, max_epochs=max_epochs, parsed=parsed)
        if isinstance(values, str):
            skipped.append({"run": run_dir.name, "reason": values})
            continue
        if unit in runs:
            raise ValueError(f"unit {unit} has two completed run directories: {runs[unit]['run']} and {run_dir.name}")
        runs[unit] = {**parsed, **values, "run": run_dir.name, "gap": 100 * (values["train_ba"] - values["val_ba"])}
    controls = {u: r for u, r in runs.items() if r["recipe"] in CONTROL_RECIPES}
    units = []
    for unit, row in sorted(runs.items()):
        if row["recipe"] in CONTROL_RECIPES:
            continue
        control_recipe = SCREEN_CONTROLS.get(row["recipe"], "baseline")
        control = controls.get(f"{row['family']}__{target}__{control_recipe}__{fold}__{row['seed']}")
        if control is None:
            units.append({"unit": unit, **{k: row[k] for k in ("family", "recipe", "seed")}, "paired": False,
                          "control_recipe": control_recipe,
                          "reason": f"no completed same-seed {control_recipe} control"})
            continue
        units.append({"unit": unit, "control": control["run"], "control_recipe": control_recipe, "paired": True,
                      **{k: row[k] for k in ("family", "recipe", "seed")},
                      "train_ba": 100 * row["train_ba"], "val_ba": 100 * row["val_ba"], "gap": row["gap"],
                      "control_train_ba": 100 * control["train_ba"], "control_val_ba": 100 * control["val_ba"],
                      "control_gap": control["gap"], "delta_gap": row["gap"] - control["gap"],
                      "delta_val_ba": 100 * (row["val_ba"] - control["val_ba"]),
                      "delta_train_ba": 100 * (row["train_ba"] - control["train_ba"]),
                      "train_ba_saturated_both": min(row["train_ba"], control["train_ba"]) * 100 > SATURATED_TRAIN_BA,
                      "train_loss": row["train_loss"], "val_loss": row["val_loss"],
                      "control_train_loss": control["train_loss"], "control_val_loss": control["val_loss"],
                      "val_min_recall": pct(row["val_min_recall"]), "min_val_loss_epoch": row["min_val_loss_epoch"]})
    outcomes = screen_outcomes(assessment)
    cells: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in units:
        if row["paired"]:
            cells.setdefault((row["family"], row["recipe"]), []).append(row)

    def mean(rows: list[dict[str, Any]], key: str) -> float | None:
        values = [r[key] for r in rows if r.get(key) is not None]
        return statistics.fmean(values) if values else None

    recipes = [{"family": family, "recipe": recipe, "control_recipe": rows[0]["control_recipe"],
                "seeds": [r["seed"] for r in rows],
                "mean_gap": mean(rows, "gap"), "mean_control_gap": mean(rows, "control_gap"),
                "mean_delta_gap": mean(rows, "delta_gap"), "mean_delta_val_ba": mean(rows, "delta_val_ba"),
                "mean_delta_train_ba": mean(rows, "delta_train_ba"),
                "train_ba_saturated_both_seeds": all(r["train_ba_saturated_both"] for r in rows),
                "mean_val_loss": mean(rows, "val_loss"), "mean_control_val_loss": mean(rows, "control_val_loss"),
                "mean_train_loss": mean(rows, "train_loss"), "mean_control_train_loss": mean(rows, "control_train_loss"),
                "delta_gap_negative_both_seeds": (None if len(rows) < 2 else all(r["delta_gap"] < 0 for r in rows)),
                "screen": outcomes.get((family, recipe), "no assessment record")}
               for (family, recipe), rows in sorted(cells.items())]
    control_spread = {}
    for family in FAMILIES:
        vals = [100 * r["val_ba"] for u, r in controls.items() if r["family"] == family and r["recipe"] == "baseline"]
        control_spread[family] = {"n_seeds": len(vals), "val_ba_spread_points": (max(vals) - min(vals)) if len(vals) > 1 else None}
    return {"audit": "v3 step D train/validation gap (exploratory; official checkpoints and screen unchanged)",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "durable": str(durable),
            "definition": {"gap": "100 x (terminal training common-four BA - terminal validation common-four BA)",
                           "terminal_epoch": max_epochs, "target": target, "fold": fold,
                           "training_ba": "trainer eval-mode pass over the training ions, logged every tenth epoch",
                           "controls": "same-seed family baseline, or the matched control of SCREEN_CONTROLS",
                           "screen_controls": SCREEN_CONTROLS, "saturated_train_ba_points": SATURATED_TRAIN_BA},
            "limits": list(LIMITS),
            "controls": {u: {"run": r["run"], "recipe": r["recipe"], "train_ba": 100 * r["train_ba"],
                             "val_ba": 100 * r["val_ba"], "gap": r["gap"], "train_loss": r["train_loss"],
                             "val_loss": r["val_loss"], "min_val_loss_epoch": r["min_val_loss_epoch"]}
                         for u, r in sorted(controls.items())},
            "control_seed_spread": control_spread,
            "units": units, "recipes": recipes, "skipped": skipped, "held_out_access": False}


def fmt(value: float | None, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def table(report: dict[str, Any]) -> str:
    lines = ["Controls: unit | train BA | val BA | gap | train loss | val loss"]
    for unit, c in report["controls"].items():
        lines.append(f"{unit} | {c['train_ba']:.1f} | {c['val_ba']:.1f} | {c['gap']:.1f} | "
                     f"{fmt(c['train_loss'], '.3f')} | {fmt(c['val_loss'], '.3f')}")
    spread = ", ".join(f"{fam} {fmt(s['val_ba_spread_points'], '.1f')} ({s['n_seeds']} seeds)"
                       for fam, s in report["control_seed_spread"].items())
    lines += [f"Baseline control validation-BA spread between seeds (points): {spread}", "",
              "Recipes (means over seeds; d = candidate - same-seed control; points):",
              "family | recipe | control | seeds | gap | control gap | d gap | d val BA | d train BA | train BA saturated | "
              "val loss (control) | train loss (control) | d gap < 0 both seeds | screen"]
    for r in report["recipes"]:
        both = r["delta_gap_negative_both_seeds"]
        lines.append(" | ".join([
            r["family"], r["recipe"], r["control_recipe"], "/".join(s.replace("seed", "") for s in r["seeds"]),
            fmt(r["mean_gap"], ".1f"), fmt(r["mean_control_gap"], ".1f"), fmt(r["mean_delta_gap"], "+.1f"),
            fmt(r["mean_delta_val_ba"], "+.1f"), fmt(r["mean_delta_train_ba"], "+.1f"),
            "yes" if r["train_ba_saturated_both_seeds"] else "no",
            f"{fmt(r['mean_val_loss'], '.3f')} ({fmt(r['mean_control_val_loss'], '.3f')})",
            f"{fmt(r['mean_train_loss'], '.3f')} ({fmt(r['mean_control_train_loss'], '.3f')})",
            "n/a (one seed)" if both is None else ("yes" if both else "no"), r["screen"]]))
    unpaired = [f"{u['unit']} ({u['reason']})" for u in report["units"] if not u["paired"]]
    if unpaired:
        lines += ["", "Not paired: " + "; ".join(unpaired)]
    if report["skipped"]:
        lines += ["", f"Skipped run directories: {len(report['skipped'])} (reasons in the JSON report)"]
    return "\n".join(lines) + "\n\nLimits\n- " + "\n- ".join(LIMITS) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", type=Path, default=CAMPAIGN, help="workstation campaign data directory")
    parser.add_argument("--durable", type=Path, help="default: CAMPAIGN/durable")
    parser.add_argument("--settings", type=Path, help="execution_settings.json (names the reused step-B run); "
                                                      "default: the latest evidence copy under CAMPAIGN")
    parser.add_argument("--assessment", type=Path, help="assessment.json copy; default: the latest step D copy under CAMPAIGN")
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
    assessment_path = args.assessment
    if assessment_path is None:
        copies = sorted(args.campaign.glob("step_d_evidence/assessments/*/assessment.json"),
                        key=lambda path: path.parent.name.split("_", 1)[1])
        assessment_path = copies[-1] if copies else None
    assessment = json.loads(assessment_path.read_text()) if assessment_path else None
    report = run_audit(durable, max_epochs=args.max_epochs, reuse=reuse, assessment=assessment)
    report["settings"] = None if settings is None else str(settings)
    report["reuse"] = reuse
    report["assessment"] = None if assessment_path is None else str(assessment_path)
    out_dir = args.out_dir or args.campaign / "audits" / f"train_val_gap_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "train_val_gap.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    text = table(report)
    (out_dir / "train_val_gap.txt").write_text(text)
    print(text)
    print(json.dumps({"report": str(out_dir / "train_val_gap.json"), "settings": report["settings"],
                      "assessment": report["assessment"], "n_controls": len(report["controls"]),
                      "n_paired_units": sum(u["paired"] for u in report["units"]),
                      "n_unpaired_units": sum(not u["paired"] for u in report["units"]),
                      "n_recipes": len(report["recipes"]), "n_skipped": len(report["skipped"])}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"train/validation gap audit refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
