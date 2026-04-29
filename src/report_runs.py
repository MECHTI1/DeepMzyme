from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable


CSV_COLUMNS = [
    "run_name",
    "run_dir",
    "task",
    "model_architecture",
    "fusion_mode",
    "seed",
    "node_feature_set",
    "ec_label_depth",
    "ec_contrastive_weight",
    "ec_contrastive_temperature",
    "selection_metric",
    "selected_checkpoint",
    "split_name",
    "split_type",
    "train_test_overlap_detected",
    "overlap_warning",
    "best_validation_loss",
    "best_validation_metric_used_for_checkpoint_selection",
    "test_metal_acc",
    "test_metal_balanced_acc",
    "test_metal_macro_f1",
    "test_metal_collapsed4_acc",
    "test_metal_collapsed4_balanced_acc",
    "test_metal_collapsed4_macro_f1",
    "test_ec_level_1_acc",
    "test_ec_level_1_balanced_acc",
    "test_ec_level_1_macro_f1",
    "test_ec_level_2_acc",
    "test_ec_level_2_balanced_acc",
    "test_ec_level_2_macro_f1",
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"warning: could not read {path}: {exc}", file=sys.stderr)
        return {}
    return payload if isinstance(payload, dict) else {}


def nested_get(payload: dict[str, Any], keys: Iterable[str]) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def is_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def infer_split_identity_from_paths(*values: Any) -> dict[str, str | None]:
    path_text = " ".join(str(value) for value in values if value)
    normalized = path_text.lower()
    if "train_and_test_sets_structures_non_overlapped_pinmymetal" in normalized:
        return {
            "split_name": "train_and_test_sets_structures_non_overlapped_pinmymetal",
            "split_type": "non_overlapped_pinmymetal",
            "overlap_warning": None,
        }
    if "train_and_test_sets_structures_exact_pinmymetal" in normalized:
        return {
            "split_name": "train_and_test_sets_structures_exact_pinmymetal",
            "split_type": "exact_pinmymetal_possibly_overlapped",
            "overlap_warning": (
                "Exact PinMyMetal split may contain train/test overlap and should "
                "be interpreted only as a secondary/reference result."
            ),
        }
    return {"split_name": None, "split_type": None, "overlap_warning": None}


def history_from_payloads(run_config: dict[str, Any], run_metadata: dict[str, Any]) -> list[dict[str, Any]]:
    history = first_present(run_config.get("history"), run_metadata.get("history"))
    if isinstance(history, list):
        return [item for item in history if isinstance(item, dict)]
    return []


def best_history_values(history: list[dict[str, Any]], selection_metric: str | None) -> tuple[Any, Any]:
    val_losses = [float(record["val_loss"]) for record in history if is_number(record.get("val_loss"))]
    best_val_loss = min(val_losses) if val_losses else None

    if not selection_metric:
        return best_val_loss, None
    metric_values = [
        float(record[selection_metric])
        for record in history
        if is_number(record.get(selection_metric))
    ]
    if not metric_values:
        return best_val_loss, None
    best_metric = min(metric_values) if selection_metric.endswith("_loss") else max(metric_values)
    return best_val_loss, best_metric


def metrics_from_report(test_report: dict[str, Any]) -> dict[str, Any]:
    metrics = test_report.get("metrics")
    return metrics if isinstance(metrics, dict) else {}


def summarize_run(run_dir: Path) -> dict[str, Any]:
    run_config = read_json(run_dir / "run_config.json")
    run_metadata = read_json(run_dir / "run_metadata.json")
    dataset_summary = read_json(run_dir / "dataset_summary.json")
    test_report = read_json(run_dir / "test_report.json")

    config = first_present(run_metadata.get("config"), run_config.get("config"), {})
    if not isinstance(config, dict):
        config = {}
    dataset = first_present(run_metadata.get("dataset_summary"), run_config.get("dataset_summary"), dataset_summary, {})
    if not isinstance(dataset, dict):
        dataset = {}
    embedded_test_report = first_present(run_metadata.get("test_report"), run_config.get("test_report"))
    if not test_report and isinstance(embedded_test_report, dict):
        test_report = embedded_test_report
    metrics = metrics_from_report(test_report)

    selection_metric = first_present(
        run_metadata.get("selection_metric"),
        run_config.get("selection_metric"),
        dataset.get("selection_metric"),
        config.get("selection_metric"),
    )
    history = history_from_payloads(run_config, run_metadata)
    best_val_loss, best_selection_metric = best_history_values(history, selection_metric)
    inferred_split = infer_split_identity_from_paths(
        config.get("structure_dir"),
        config.get("summary_csv"),
        config.get("test_structure_dir"),
        config.get("test_summary_csv"),
        dataset.get("structure_dir"),
        dataset.get("summary_csv"),
        dataset.get("test_structure_dir"),
        dataset.get("test_summary_csv"),
    )

    row = {
        "run_name": first_present(config.get("run_name"), run_dir.name),
        "run_dir": str(run_dir),
        "task": first_present(config.get("task"), dataset.get("task")),
        "model_architecture": config.get("model_architecture"),
        "fusion_mode": config.get("fusion_mode"),
        "seed": config.get("seed"),
        "node_feature_set": first_present(config.get("node_feature_set"), dataset.get("node_feature_set")),
        "ec_label_depth": first_present(config.get("ec_label_depth"), dataset.get("ec_label_depth")),
        "ec_contrastive_weight": config.get("ec_contrastive_weight"),
        "ec_contrastive_temperature": config.get("ec_contrastive_temperature"),
        "selection_metric": selection_metric,
        "selected_checkpoint": first_present(
            run_metadata.get("selected_checkpoint"),
            run_config.get("selected_checkpoint"),
        ),
        "split_name": first_present(
            run_metadata.get("split_name"),
            test_report.get("split_name"),
            dataset.get("split_name"),
            config.get("split_name"),
            inferred_split["split_name"],
        ),
        "split_type": first_present(
            run_metadata.get("split_type"),
            test_report.get("split_type"),
            dataset.get("split_type"),
            config.get("split_type"),
            inferred_split["split_type"],
        ),
        "train_test_overlap_detected": first_present(
            run_metadata.get("train_test_overlap_detected"),
            test_report.get("train_test_overlap_detected"),
            dataset.get("train_test_overlap_detected"),
        ),
        "overlap_warning": first_present(
            run_metadata.get("overlap_warning"),
            test_report.get("overlap_warning"),
            dataset.get("overlap_warning"),
            config.get("overlap_warning"),
            inferred_split["overlap_warning"],
        ),
        "best_validation_loss": best_val_loss,
        "best_validation_metric_used_for_checkpoint_selection": first_present(
            run_metadata.get("selected_metric_value"),
            run_config.get("selected_metric_value"),
            best_selection_metric,
        ),
    }
    for metric_name in CSV_COLUMNS:
        if metric_name.startswith("test_"):
            row[metric_name] = metrics.get(metric_name)
    return row


def discover_run_dirs(runs_dir: Path | None, run_dirs: list[Path] | None) -> list[Path]:
    if run_dirs:
        return [path.resolve() for path in run_dirs]
    if runs_dir is None:
        raise ValueError("Either --runs-dir or --run-dirs must be provided.")

    runs_dir = runs_dir.resolve()
    marker_names = {"run_config.json", "run_metadata.json", "dataset_summary.json", "test_report.json"}
    if any((runs_dir / marker).exists() for marker in marker_names):
        return [runs_dir]
    children = [
        path
        for path in sorted(runs_dir.iterdir())
        if path.is_dir() and any((path / marker).exists() for marker in marker_names)
    ]
    return children


def normalize_csv_value(value: Any) -> Any:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def write_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: normalize_csv_value(row.get(column)) for column in CSV_COLUMNS})


def write_figure(rows: list[dict[str, Any]], out_figure: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"warning: matplotlib unavailable; skipping figure: {exc}", file=sys.stderr)
        return

    plot_rows = [
        row
        for row in rows
        if is_number(row.get("best_validation_metric_used_for_checkpoint_selection"))
    ]
    if not plot_rows:
        print("warning: no numeric selected validation metric available; skipping figure", file=sys.stderr)
        return

    labels = [str(row.get("run_name") or Path(str(row.get("run_dir"))).name) for row in plot_rows]
    values = [float(row["best_validation_metric_used_for_checkpoint_selection"]) for row in plot_rows]
    fig_width = max(6, min(18, len(plot_rows) * 1.2))
    fig, ax = plt.subplots(figsize=(fig_width, 4))
    ax.bar(range(len(values)), values, color="#4c78a8")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Selected validation metric")
    ax.set_title("DeepMzyme run comparison")
    fig.tight_layout()
    out_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_figure, dpi=160)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize DeepMzyme training runs into a comparison CSV.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--runs-dir", type=Path, default=None, help="Parent directory containing run directories.")
    source.add_argument("--run-dirs", type=Path, nargs="+", default=None, help="Explicit run directories to summarize.")
    parser.add_argument("--out-csv", type=Path, required=True, help="Output CSV path.")
    parser.add_argument("--out-figure", type=Path, default=None, help="Optional output figure path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_dirs = discover_run_dirs(args.runs_dir, args.run_dirs)
    rows = [summarize_run(run_dir) for run_dir in run_dirs]
    write_csv(rows, args.out_csv)
    if args.out_figure is not None:
        write_figure(rows, args.out_figure)
    print(f"Wrote {len(rows)} run summaries to {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
