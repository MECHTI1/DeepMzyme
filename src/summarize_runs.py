from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize training run directories into a comparison CSV.")
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value_or_default(record: dict, metric_name: str, default: float) -> float:
    value = record.get(metric_name)
    if value is None:
        return default
    return float(value)


def build_rows(runs_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        run_config_path = run_dir / "run_config.json"
        if not run_config_path.exists():
            continue
        run_config = load_json(run_config_path)
        test_report_path = run_dir / "test_report.json"
        test_report = load_json(test_report_path) if test_report_path.exists() else {}
        config = run_config.get("config", {})
        history = run_config.get("history", [])
        selection_metric = str(config.get("selection_metric", "train_loss"))
        if selection_metric.endswith("_loss"):
            best_epoch_record = min(
                history,
                key=lambda record: metric_value_or_default(record, selection_metric, float("inf")),
                default={},
            )
        else:
            best_epoch_record = max(
                history,
                key=lambda record: metric_value_or_default(record, selection_metric, float("-inf")),
                default={},
            )
        test_metrics = test_report.get("metrics", {})
        rows.append(
            {
                "run_name": run_dir.name,
                "task": str(config.get("task", "")),
                "model_architecture": str(config.get("model_architecture", "")),
                "fusion_mode": str(config.get("fusion_mode", "")),
                "ec_label_depth": str(config.get("ec_label_depth", "")),
                "selection_metric": selection_metric,
                "best_epoch": str(best_epoch_record.get("epoch", "")),
                "val_metal_balanced_acc": str(best_epoch_record.get("val_metal_balanced_acc", "")),
                "val_ec_balanced_acc": str(best_epoch_record.get("val_ec_balanced_acc", "")),
                "test_metal_balanced_acc": str(test_metrics.get("test_metal_balanced_acc", "")),
                "test_metal_collapsed4_balanced_acc": str(test_metrics.get("test_metal_collapsed4_balanced_acc", "")),
                "test_ec_balanced_acc": str(test_metrics.get("test_ec_balanced_acc", "")),
            }
        )
    return rows


def write_rows(output_csv: Path, rows: list[dict[str, str]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "task",
        "model_architecture",
        "fusion_mode",
        "ec_label_depth",
        "selection_metric",
        "best_epoch",
        "val_metal_balanced_acc",
        "val_ec_balanced_acc",
        "test_metal_balanced_acc",
        "test_metal_collapsed4_balanced_acc",
        "test_ec_balanced_acc",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = build_rows(args.runs_dir)
    write_rows(args.output_csv, rows)
    print(f"Wrote {len(rows)} summarized run rows to {args.output_csv}")


if __name__ == "__main__":
    main()
