#!/usr/bin/env python3
"""The v3 step F test report (F4; rules: docs/campaigns/pmm_ion_metal_v3/step_f_spec.json, section "report").

This is the only component that uses a test label value (the label-blind preparation passes the label-bearing cells
of the source CSV through the CSV reader and discards them unused). It runs only on the predictions of the single
claimed prediction pass, under the recorded Freeze 1 (whose code hashes include this file) and Freeze 2, all read at
their canonical paths under --campaign-dir. It is a pure function of the frozen predictions, the label file (the
pinned PMM test source CSV, column ``label_metal`` only) and the Freeze 2 row lists: rerunning it reproduces
report.json and report.md byte for byte (the receipt with times and paths is written separately).

Lines (spec): the full test (every source row with a four-class label code; unscorable rows are errors in every
class-recall denominator), the LINK-free and training-eligible sensitivities, and the clean subset (secondary; not
pristine). Metrics: the common-four balanced accuracy (mean of the Mn, Cu, Zn and Class VIII recalls) and the four
recalls with counts, each with a PDB-entry percentile bootstrap interval (10,000 resamples, seed 42) that describes the
uncertainty of the reported value only. The full-test line is set beside PMM's published test values with the allowed
wording and no interval, p-value or superiority claim for the difference. Labels: both_results_secondary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import pmm_v3_test_prep as prep

SPEC, SPEC_PATH = prep.SPEC, prep.SPEC_PATH
REPORT_SPEC = SPEC["report"]
LABELS = SPEC["label_file"]
COMMON4 = ("Mn", "Cu", "Zn", "Class VIII")
LINES = ("full_test", "link_free", "training_eligible", "clean_subset")
LINE_TITLES = {"full_test": "Full test", "link_free": "LINK-free sensitivity",
               "training_eligible": "Training-eligible sensitivity", "clean_subset": "Clean subset (secondary; not pristine)"}
BOOTSTRAP_CHUNK = 1000
require = prep.require


def sha256_file(path: Path) -> str:
    return prep.sha256_file(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def truthy(value: str) -> bool:
    require(value in ("True", "False"), f"Not a boolean cell: {value!r}")
    return value == "True"


# ---------------------------------------------------------------------------
# Verified inputs
# ---------------------------------------------------------------------------

def verified_inputs(*, campaign_dir: Path, predictions_dir: Path, prepared_dir: Path) -> dict[str, Any]:
    """The frozen predictions and prepared rows, each checked against its receipt, the recorded Freeze 1 (this code)
    and Freeze 2, and the single prediction claim."""
    tools = {"pmm_v3_step_f_report.py": Path(__file__), "pmm_v3_test_prep.py": Path(prep.__file__)}
    freeze1, freeze1_sha = prep.check_freeze1(campaign_dir, tools=tools)
    freeze2, freeze2_sha = prep.check_freeze2(campaign_dir, freeze1_sha256=freeze1_sha, prepared_dir=prepared_dir)
    claim, claim_sha = prep.check_prediction_claim(campaign_dir, freeze2_sha256=freeze2_sha,
                                                   predictions_dir=predictions_dir)
    receipt = json.loads((Path(predictions_dir) / "prediction_receipt.json").read_text(encoding="utf-8"))
    predictions_path = Path(predictions_dir) / "test_predictions.csv"
    require(sha256_file(predictions_path) == receipt["predictions_sha256"], "The predictions changed after the pass")
    require(receipt["freeze2_sha256"] == freeze2_sha and receipt.get("prediction_claim_sha256") == claim_sha
            and receipt["labels_read"] is False, "The predictions were not written by the claimed pass under Freeze 2")
    require(receipt["checkpoint_sha256"] == freeze2["checkpoint_sha256"] == freeze1["checkpoint_sha256"]
            == claim["checkpoint_sha256"], "The predictions used another checkpoint")
    prepared = read_csv(Path(prepared_dir) / "prepared_rows.csv")
    predictions = read_csv(predictions_path)
    require([r["source_uid"] for r in predictions] == [r["source_uid"] for r in prepared],
            "The predictions do not cover the prepared rows one to one, in order")
    for pred, row in zip(predictions, prepared):
        require(all(pred[key] == row[key] for key in ("source_row", "pdbid", "disposition", "link_flagged",
                                                      "alias_primary_source_uid", "clean_subset")),
                f"{row['source_uid']}: the prediction row disagrees with the frozen preparation")
        scored = row["disposition"] != prep.UNSCORABLE
        require((pred["pred_common4"] != "") == scored, f"{row['source_uid']}: a scored row lacks a prediction or an "
                                                        "unscorable row has one")
        require(pred["checkpoint_sha256"] == freeze2["checkpoint_sha256"], "A prediction row names another checkpoint")
    scored_uids = (Path(prepared_dir) / "scored_source_uids.txt").read_text(encoding="utf-8").split()
    clean_uids = (Path(prepared_dir) / "clean_subset_source_uids.txt").read_text(encoding="utf-8").split()
    require(scored_uids == [r["source_uid"] for r in prepared if r["disposition"] != prep.UNSCORABLE]
            and clean_uids == [r["source_uid"] for r in prepared if truthy(r["clean_subset"])],
            "The Freeze 2 row lists disagree with the prepared rows")
    return {"freeze2": freeze2, "receipt": receipt, "predictions": predictions,
            "go_aheads": {"freeze1": freeze1["user_go_ahead_log_entry"], "freeze2": freeze2["user_go_ahead_log_entry"],
                          "single_test_pass": claim["user_go_ahead_log_entry"]},
            "hashes": {"freeze1_sha256": freeze1_sha, "freeze2_sha256": freeze2_sha,
                       "prediction_claim_sha256": claim_sha, "predictions_sha256": receipt["predictions_sha256"],
                       "prepared_files_sha256": dict(sorted(freeze2["prepared_files_sha256"].items())),
                       "checkpoint_sha256": freeze2["checkpoint_sha256"]}}


def read_labels(source_csv: Path, *, expected_sha256: str, n_rows: int) -> list[dict[str, str]]:
    """The label column (and the PDB ID, to check the row key) of the pinned source CSV; nothing else is kept."""
    require(sha256_file(source_csv) == expected_sha256, "The label file is not the pinned test source CSV")
    rows = prep.read_allowlisted(source_csv, ("pdbid", LABELS["column"]))
    require(len(rows) == n_rows, f"The label file has {len(rows)} rows, not {n_rows}")
    return rows


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def joined_rows(predictions: list[dict[str, str]], labels: list[dict[str, str]]) -> tuple[list[dict[str, Any]],
                                                                                         list[dict[str, Any]]]:
    """Prediction rows with their common-four label; rows whose code is outside the four are set aside (listed)."""
    codes = LABELS["code_to_common_four"]
    rows, other = [], []
    for pred in predictions:
        number = int(pred["source_row"])
        label = labels[number - 1]
        require(label["pdbid"].lower() == pred["pdbid"], f"source row {number}: the PDB ID differs from the prediction")
        code = label[LABELS["column"]].strip()
        if code not in codes:
            other.append({"source_row": number, "label_code": code})
            continue
        rows.append({"source_uid": pred["source_uid"], "source_row": number, "pdbid": pred["pdbid"],
                     "label": COMMON4.index(codes[code]),
                     "pred": None if pred["pred_common4"] == "" else int(pred["pred_common4"]),
                     "scored": pred["disposition"] != prep.UNSCORABLE, "link_flagged": truthy(pred["link_flagged"]),
                     "primary": pred["alias_primary_source_uid"] == pred["source_uid"],
                     "clean": truthy(pred["clean_subset"])})
    return rows, other


def line_members(rows: list[dict[str, Any]], line: str) -> list[dict[str, Any]]:
    if line == "full_test":
        return list(rows)
    if line == "link_free":
        return [r for r in rows if not r["link_flagged"]]
    if line == "training_eligible":
        return [r for r in rows if r["scored"] and not r["link_flagged"] and r["primary"]]
    if line == "clean_subset":
        return [r for r in rows if r["clean"]]
    raise ValueError(line)


def counts(members: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    """Per-class support and correct counts; an unscorable row is in the support and never correct."""
    support, correct = [0] * 4, [0] * 4
    for row in members:
        support[row["label"]] += 1
        correct[row["label"]] += int(row["scored"] and row["pred"] == row["label"])
    return support, correct


def metric_values(support, correct) -> dict[str, Any]:
    recalls = [correct[c] / support[c] if support[c] else None for c in range(4)]
    defined = all(value is not None for value in recalls)
    return {"balanced_accuracy": sum(recalls) / 4 if defined else None,
            "recalls": dict(zip(COMMON4, recalls))}


def bootstrap(members: list[dict[str, Any]]) -> dict[str, Any]:
    """Percentile intervals over PDB entries (resampled with replacement): numpy.random.default_rng(seed), integer
    draws of shape (1,000, n_entries) in ten consecutive chunks. A resample without a row of some class has no BA
    (and no recall for that class); such resamples are counted and left out of that statistic's interval."""
    import numpy as np

    config = REPORT_SPEC["bootstrap"]
    entries = sorted({row["pdbid"] for row in members})
    if not entries:
        return {"entries": 0, "resamples": 0}
    index = {pdbid: i for i, pdbid in enumerate(entries)}
    support = np.zeros((len(entries), 4), dtype=np.int64)
    correct = np.zeros((len(entries), 4), dtype=np.int64)
    for row in members:
        support[index[row["pdbid"]], row["label"]] += 1
        correct[index[row["pdbid"]], row["label"]] += int(row["scored"] and row["pred"] == row["label"])
    rng = np.random.default_rng(int(config["seed"]))
    total = int(config["resamples"])
    require(total % BOOTSTRAP_CHUNK == 0, "The resample count must be a multiple of the chunk size")
    recalls = []
    for _ in range(total // BOOTSTRAP_CHUNK):
        draw = rng.integers(0, len(entries), size=(BOOTSTRAP_CHUNK, len(entries)))
        s, k = support[draw].sum(axis=1), correct[draw].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            recalls.append(np.where(s > 0, k / np.maximum(s, 1), np.nan))
    recalls = np.concatenate(recalls)
    alpha = (1.0 - float(config["confidence"])) / 2.0

    def interval(values):
        values = values[~np.isnan(values)]
        if values.size == 0:
            return {"lower": None, "upper": None, "defined_resamples": 0}
        low, high = np.quantile(values, [alpha, 1.0 - alpha])
        return {"lower": float(low), "upper": float(high), "defined_resamples": int(values.size)}

    ba = recalls.mean(axis=1)  # NaN whenever a class is absent from the resample
    return {"entries": len(entries), "resamples": total, "seed": int(config["seed"]),
            "confidence": float(config["confidence"]), "unit": config["unit"],
            "balanced_accuracy": interval(ba),
            "recalls": {label: interval(recalls[:, c]) for c, label in enumerate(COMMON4)}}


def line_report(rows: list[dict[str, Any]], line: str) -> dict[str, Any]:
    members = line_members(rows, line)
    support, correct = counts(members)
    return {"title": LINE_TITLES[line], "rows": len(members), "scored": sum(1 for r in members if r["scored"]),
            "unscorable_counted_as_errors": sum(1 for r in members if not r["scored"]),
            "pdb_entries": len({r["pdbid"] for r in members}),
            "support": dict(zip(COMMON4, support)), "correct": dict(zip(COMMON4, correct)),
            **metric_values(support, correct), "bootstrap": bootstrap(members)}


def pmm_comparison(full: dict[str, Any]) -> dict[str, Any]:
    """The full-test values beside PMM's published Figure 2b test values (percent); wording only, no inference."""
    published = REPORT_SPEC["pmm_comparison"]["published_test_figure_2b"]
    ours = {label: None if full["recalls"][label] is None else 100.0 * full["recalls"][label] for label in COMMON4}
    ours["mean"] = None if full["balanced_accuracy"] is None else 100.0 * full["balanced_accuracy"]
    out = {}
    for key in (*COMMON4, "mean"):
        value = ours[key]
        relation = None if value is None else ("higher than" if round(value, 2) > published[key]
                                               else "lower than" if round(value, 2) < published[key] else "equal to")
        out[key] = {"deepmzyme_v3_percent": None if value is None else round(value, 2),
                    "pmm_published_percent": published[key], "relation": relation}
    return {"values": out, "wording": REPORT_SPEC["pmm_comparison"]["allowed_wording"],
            "applies_to": REPORT_SPEC["pmm_comparison"]["applies_to"],
            "statement": "not a matched comparison; no interval, p-value or superiority claim for the difference"}


def build_report(*, campaign_dir: Path, predictions_dir: Path, prepared_dir: Path, source_csv: Path) -> dict[str, Any]:
    inputs = verified_inputs(campaign_dir=campaign_dir, predictions_dir=predictions_dir, prepared_dir=prepared_dir)
    freeze2 = inputs["freeze2"]
    labels = read_labels(source_csv, expected_sha256=freeze2["test_source_csv_sha256"],
                         n_rows=int(freeze2["test_source_rows"]))
    rows, other = joined_rows(inputs["predictions"], labels)
    lines = {line: line_report(rows, line) for line in LINES}
    scored_all = sum(1 for p in inputs["predictions"] if p["disposition"] != prep.UNSCORABLE)
    return {
        "schema_version": 1, "campaign_id": SPEC["campaign_id"], "step": "F4 report",
        "label": "both_results_secondary",
        "selection": {key: SPEC["selection"][key] for key in ("cell", "epochs", "seed")},
        "inputs": {**inputs["hashes"], "label_file_sha256": freeze2["test_source_csv_sha256"],
                   "step_f_spec_sha256": sha256_file(SPEC_PATH), "report_tool_sha256": sha256_file(Path(__file__))},
        "user_go_ahead_log_entries": inputs["go_aheads"],
        "source_rows": len(inputs["predictions"]),
        "scored_statement": f"{scored_all} of {len(inputs['predictions'])} scored",
        "rows_with_other_label_codes": {"count": len(other), "rows": other,
                                        "rule": LABELS["other_codes"]},
        "lines": lines,
        "pmm_comparison": pmm_comparison(lines["full_test"]),
        "disclosures": list(SPEC["disclosures"]),
        "metrics_rule": REPORT_SPEC["metrics"],
    }


# ---------------------------------------------------------------------------
# Rendering (deterministic)
# ---------------------------------------------------------------------------

def pct(value: float | None) -> str:
    return "undefined" if value is None else f"{100.0 * value:.2f}"


def interval_text(item: dict[str, Any]) -> str:
    if item.get("lower") is None:
        return "undefined"
    return f"[{100.0 * item['lower']:.2f}, {100.0 * item['upper']:.2f}]"


def render_markdown(report: dict[str, Any]) -> str:
    out = ["# DeepMzyme v3 step F test report", "",
           f"Label: **{report['label']}**. Selected model: `{report['selection']['cell']}`, "
           f"{report['selection']['epochs']} epochs, terminal checkpoint, seed {report['selection']['seed']}.", "",
           f"Source rows: {report['source_rows']}; {report['scored_statement']}. Rows with a label code outside the "
           f"four classes: {report['rows_with_other_label_codes']['count']} (excluded from every metric).", ""]
    for line in LINES:
        item = report["lines"][line]
        boot = item["bootstrap"]
        out += [f"## {item['title']}", "",
                f"{item['scored']} of {item['rows']} scored ({item['unscorable_counted_as_errors']} unscorable rows "
                f"counted as errors); {item['pdb_entries']} PDB entries.", "",
                "| Metric | Value (%) | 95% PDB-entry bootstrap (%) | Correct / support |", "|---|---:|---:|---:|",
                f"| Common-four balanced accuracy | {pct(item['balanced_accuracy'])} | "
                f"{interval_text(boot.get('balanced_accuracy', {}))} | |"]
        for label in COMMON4:
            out.append(f"| {label} recall | {pct(item['recalls'][label])} | "
                       f"{interval_text(boot.get('recalls', {}).get(label, {}))} | "
                       f"{item['correct'][label]} / {item['support'][label]} |")
        out.append("")
    comparison = report["pmm_comparison"]
    out += ["## Beside PMM's published test values (full test only)", "",
            f"Wording: {comparison['wording']}. {comparison['statement'].capitalize()}.", "",
            "| | DeepMzyme v3 (%) | PMM published, Fig. 2b (%) | Relation |", "|---|---:|---:|---|"]
    for key, value in comparison["values"].items():
        ours = "undefined" if value["deepmzyme_v3_percent"] is None else f"{value['deepmzyme_v3_percent']:.2f}"
        out.append(f"| {'Mean (BA)' if key == 'mean' else key} | {ours} | {value['pmm_published_percent']:.2f} | "
                   f"{value['relation'] or 'undefined'} |")
    out += ["", "## Disclosures", ""] + [f"- {text}" for text in report["disclosures"]]
    out += ["", "## Inputs (SHA-256)", ""] + [f"- {key}: `{value}`" for key, value in sorted(report["inputs"].items())
                                               if isinstance(value, str)]
    out += [f"- prepared files: " + ", ".join(f"{name} `{digest[:12]}`" for name, digest in
                                              report["inputs"]["prepared_files_sha256"].items()), ""]
    return "\n".join(out)


def write_report(report: dict[str, Any], out_dir: Path, *, paths: dict[str, str]) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    data_json = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    data_md = render_markdown(report).encode("utf-8")
    (out_dir / "report.json").write_bytes(data_json)
    (out_dir / "report.md").write_bytes(data_md)
    receipt = {"report_json_sha256": hashlib.sha256(data_json).hexdigest(),
               "report_md_sha256": hashlib.sha256(data_md).hexdigest(), "inputs": paths,
               "labels_read_by": "pmm_v3_step_f_report.py", "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                                        time.gmtime())}
    (out_dir / "report_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                                                 encoding="utf-8")
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("campaign-dir", "predictions-dir", "prepared-dir", "source-csv", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_report(campaign_dir=args.campaign_dir, predictions_dir=args.predictions_dir,
                          prepared_dir=args.prepared_dir, source_csv=args.source_csv)
    receipt = write_report(report, args.out_dir, paths={key: str(value) for key, value in vars(args).items()})
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"v3 step F report refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
