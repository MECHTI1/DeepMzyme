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


def load_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    return load_json(path)


def csv_list(value) -> str:
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    return "" if value is None else str(value)


def metric_value_or_default(record: dict, metric_name: str, default: float) -> float:
    value = record.get(metric_name)
    if value is None:
        return default
    return float(value)


def model_display_label(config: dict) -> str:
    arch = str(config.get("model_architecture") or "unknown")
    fusion = str(config.get("fusion_mode") or "")
    use_esm_branch = config.get("use_esm_branch")
    if arch == "only_gvp":
        return "GVP only"
    if arch == "only_esm":
        return "ESM only"
    if arch == "simple_gnn_esm":
        return "SimpleGNN + ESM"
    if arch == "gvp" and use_esm_branch is False:
        return "GVP only"
    if arch == "gvp":
        fusion_labels = {
            "late_fusion": "late fusion",
            "early_fusion": "early fusion",
            "node_level_late_fusion": "node-level ESM fusion",
            "hybrid": "hybrid ESM fusion",
            "cross_modal_attention": "cross-modal attention",
        }
        suffix = fusion_labels.get(fusion, fusion.replace("_", " ") if fusion else "fusion")
        return f"GVP + ESM {suffix}"
    return arch.replace("_", " ")


def build_rows(runs_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        run_config_path = run_dir / "run_config.json"
        if not run_config_path.exists():
            continue
        run_config = load_json(run_config_path)
        test_report_path = run_dir / "test_report.json"
        test_report = load_json_if_exists(test_report_path)
        split_diagnostics = load_json_if_exists(run_dir / "split_diagnostics.json")
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
                "model_label": model_display_label(config),
                "ec_label_depth": str(config.get("ec_label_depth", "")),
                "joint_loss_weighting": str(config.get("joint_loss_weighting", "")),
                "split_by": str(split_diagnostics.get("split_by", config.get("split_by", ""))),
                "n_train_pockets": str(split_diagnostics.get("n_train_pockets", "")),
                "n_val_pockets": str(split_diagnostics.get("n_val_pockets", "")),
                "n_train_groups": str(split_diagnostics.get("n_train_groups", "")),
                "n_val_groups": str(split_diagnostics.get("n_val_groups", "")),
                "train_val_overlap_pdbid": str(split_diagnostics.get("train_val_overlap_pdbid", "")),
                "train_val_overlap_pdbid_chain": str(split_diagnostics.get("train_val_overlap_pdbid_chain", "")),
                "train_val_overlap_structure_id": str(split_diagnostics.get("train_val_overlap_structure_id", "")),
                "train_val_overlap_pocket_id": str(split_diagnostics.get("train_val_overlap_pocket_id", "")),
                "missing_train_metal_classes": csv_list(split_diagnostics.get("missing_train_metal_classes")),
                "missing_val_metal_classes": csv_list(split_diagnostics.get("missing_val_metal_classes")),
                "missing_train_ec_classes": csv_list(split_diagnostics.get("missing_train_ec_classes")),
                "missing_val_ec_classes": csv_list(split_diagnostics.get("missing_val_ec_classes")),
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
        "model_label",
        "ec_label_depth",
        "joint_loss_weighting",
        "split_by",
        "n_train_pockets",
        "n_val_pockets",
        "n_train_groups",
        "n_val_groups",
        "train_val_overlap_pdbid",
        "train_val_overlap_pdbid_chain",
        "train_val_overlap_structure_id",
        "train_val_overlap_pocket_id",
        "missing_train_metal_classes",
        "missing_val_metal_classes",
        "missing_train_ec_classes",
        "missing_val_ec_classes",
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
