from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable


CSV_COLUMNS = [
    "result_stage",
    "run_name",
    "run_dir",
    "task",
    "model_architecture",
    "fusion_mode",
    "model_label",
    "seed",
    "model_seed",
    "split_seed",
    "final_test_primary_report",
    "final_test_ensemble_mode",
    "final_test_result_role",
    "selected_config_id",
    "learning_rate",
    "weight_decay",
    "batch_size",
    "epochs",
    "use_ring_edges",
    "omit_node_features",
    "node_feature_set",
    "ec_label_depth",
    "ec_group_weighting",
    "ec_contrastive_weight",
    "ec_contrastive_temperature",
    "joint_loss_weighting",
    "metal_loss_weight",
    "ec_loss_weight",
    "metal_collapsed_loss_weight",
    "metal_class_weight_mode",
    "selection_metric",
    "selected_checkpoint",
    "split_name",
    "split_type",
    "split_by",
    "val_fraction",
    "n_folds",
    "fold_index",
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
    "train_test_overlap_detected",
    "overlap_warning",
    "best_validation_loss",
    "best_validation_metric_used_for_checkpoint_selection",
    "selected_val_joint_balanced_acc",
    "selected_val_metal_balanced_acc",
    "selected_val_metal_min_recall",
    "selected_val_metal_per_class_recall",
    "selected_val_metal_per_class_support",
    "selected_val_metal_collapsed4_balanced_acc",
    "selected_val_metal_collapsed4_min_recall",
    "selected_val_metal_collapsed4_per_class_recall",
    "selected_val_metal_collapsed4_per_class_support",
    "selected_val_ec_balanced_acc",
    "selected_val_ec_group_balanced_acc",
    "comparison_test_metric_name",
    "comparison_test_metric_value",
    "test_joint_balanced_acc",
    "test_joint_macro_f1",
    "test_metal_acc",
    "test_metal_balanced_acc",
    "test_metal_min_recall",
    "test_metal_balanced_acc_ci95",
    "test_metal_macro_f1",
    "test_metal_ece_equal_mass",
    "test_metal_ece_equal_mass_ci95",
    "test_metal_nll",
    "test_metal_temperature_scaled_balanced_acc",
    "test_metal_temperature_scaled_ece_equal_mass",
    "test_metal_temperature_scaled_nll",
    "test_metal_collapsed4_acc",
    "test_metal_collapsed4_balanced_acc",
    "test_metal_collapsed4_min_recall",
    "test_metal_collapsed4_balanced_acc_ci95",
    "test_metal_collapsed4_macro_f1",
    "test_metal_collapsed4_mn_recall",
    "test_metal_collapsed4_cu_recall",
    "test_metal_collapsed4_zn_recall",
    "test_metal_collapsed4_class_viii_recall",
    "test_ec_level_1_acc",
    "test_ec_level_1_balanced_acc",
    "test_ec_level_1_macro_f1",
    "test_ec_level_2_acc",
    "test_ec_level_2_balanced_acc",
    "test_ec_level_2_macro_f1",
    "test_ec_group_acc",
    "test_ec_group_balanced_acc",
    "test_ec_group_macro_f1",
    "test_ec_group_n_groups",
    "test_ec_group_n_conflicting_groups",
    "test_ec_group_level_1_acc",
    "test_ec_group_level_1_balanced_acc",
    "test_ec_group_level_1_macro_f1",
    "test_ec_group_level_2_acc",
    "test_ec_group_level_2_balanced_acc",
    "test_ec_group_level_2_macro_f1",
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


def csv_list(value: Any) -> Any:
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    return value


def is_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def infer_split_identity_from_paths(*values: Any) -> dict[str, str | None]:
    path_text = " ".join(str(value) for value in values if value)
    normalized = path_text.lower()
    if "train_and_test_sets_structures_harsh_pinmymetal" in normalized:
        return {
            "split_name": "Harsh Split PinMyMetal",
            "split_type": "harsh_pinmymetal",
            "overlap_warning": None,
        }
    if "train_and_test_sets_structures_non_overlapped_pinmymetal" in normalized:
        return {
            "split_name": "Non-overlapped PinMyMetal",
            "split_type": "non_overlapped_pinmymetal",
            "overlap_warning": None,
        }
    if "train_and_test_sets_structures_exact_pinmymetal" in normalized:
        return {
            "split_name": "Metal Split PinMyMetal",
            "split_type": "metal_split_pinmymetal_possibly_overlapped",
            "overlap_warning": (
                "Metal Split PinMyMetal follows the exact PinMyMetal split and may contain train/test overlap. "
                "It should "
                "be interpreted only as a secondary/reference result."
            ),
        }
    if "train_and_test_sets_structures_common_pdbid_70_30_pinmymetal" in normalized:
        return {
            "split_name": "Common-PDBID 70/30 Split PinMyMetal",
            "split_type": "common_pdbid_70_30_pinmymetal",
            "overlap_warning": (
                "Common-PDBID 70/30 Split PinMyMetal is a custom comparison split, "
                "not the trusted final held-out split."
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


def selected_history_record(
    history: list[dict[str, Any]],
    run_config: dict[str, Any],
    run_metadata: dict[str, Any],
    selection_metric: str | None,
) -> dict[str, Any]:
    if not history:
        return {}
    selected_epoch = first_present(
        run_metadata.get("selected_checkpoint_epoch"),
        run_config.get("selected_checkpoint_epoch"),
    )
    if selected_epoch is not None:
        try:
            selected_epoch_int = int(selected_epoch)
        except (TypeError, ValueError):
            selected_epoch_int = None
        if selected_epoch_int is not None:
            for record in history:
                try:
                    record_epoch = int(record.get("epoch", -1))
                except (TypeError, ValueError):
                    continue
                if record_epoch == selected_epoch_int:
                    return record

    if selection_metric:
        candidates = [record for record in history if is_number(record.get(selection_metric))]
        if candidates:
            key = lambda record: float(record[selection_metric])
            return min(candidates, key=key) if selection_metric.endswith("_loss") else max(candidates, key=key)
    return history[-1]


def selected_metric_value(
    selected_record: dict[str, Any],
    metric_name: str,
    selection_metric: str | None,
    selected_selection_value: Any,
) -> Any:
    return first_present(
        selected_record.get(metric_name),
        selected_selection_value if selection_metric == metric_name else None,
    )


def metrics_from_report(test_report: dict[str, Any]) -> dict[str, Any]:
    metrics = test_report.get("metrics")
    result = dict(metrics) if isinstance(metrics, dict) else {}
    calibrated = test_report.get("calibrated_metrics")
    if isinstance(calibrated, dict):
        result.update(calibrated)
    return result


def matching_test_metric_name(selection_metric: str | None, task: str | None) -> str | None:
    metric = str(selection_metric or "")
    if metric.startswith("val_"):
        return "test_" + metric.removeprefix("val_")
    if task == "metal":
        return "test_metal_balanced_acc"
    if task == "ec":
        return "test_ec_group_balanced_acc"
    return None


def infer_result_stage(run_dir: Path, metrics: dict[str, Any]) -> str:
    if metrics:
        report = read_json(run_dir / "test_report.json")
        if report.get("final_test_ensemble_mode") == "softmax_mean_5_seeds":
            return "final-test ensemble evaluated"
        return "final-test evaluated"
    name_text = str(run_dir.name).lower()
    if "group_kfold" in name_text or "group-kfold" in name_text or ("top" in name_text and "fold" in name_text):
        return "group-kfold validation"
    if "seed_repeat" in name_text or ("top" in name_text and "seed" in name_text):
        return "seed-repeat validation"
    return "validation-only"


def summarize_run(run_dir: Path) -> dict[str, Any]:
    run_config = read_json(run_dir / "run_config.json")
    run_metadata = read_json(run_dir / "run_metadata.json")
    dataset_summary = read_json(run_dir / "dataset_summary.json")
    split_diagnostics = read_json(run_dir / "split_diagnostics.json")
    test_report = read_json(run_dir / "test_report.json")

    config = first_present(run_metadata.get("config"), run_config.get("config"), {})
    if not isinstance(config, dict):
        config = {}
    dataset = first_present(run_metadata.get("dataset_summary"), run_config.get("dataset_summary"), dataset_summary, {})
    if not isinstance(dataset, dict):
        dataset = {}
    if not split_diagnostics:
        embedded_split_diagnostics = dataset.get("split_diagnostics")
        split_diagnostics = embedded_split_diagnostics if isinstance(embedded_split_diagnostics, dict) else {}
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
    selected_record = selected_history_record(history, run_config, run_metadata, selection_metric)
    selected_selection_value = first_present(
        run_metadata.get("selected_metric_value"),
        run_config.get("selected_metric_value"),
        best_selection_metric,
    )
    task = first_present(config.get("task"), dataset.get("task"))
    test_metric_name = matching_test_metric_name(selection_metric, task)
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

    omit_raw = config.get("omit_node_features")
    if isinstance(omit_raw, (list, tuple)):
        omit_str = ",".join(str(x) for x in omit_raw) if omit_raw else "none"
    elif omit_raw:
        omit_str = str(omit_raw)
    else:
        omit_str = "none"

    row = {
        "result_stage": infer_result_stage(run_dir, metrics),
        "run_name": first_present(config.get("run_name"), run_dir.name),
        "run_dir": str(run_dir),
        "task": task,
        "model_architecture": config.get("model_architecture"),
        "fusion_mode": config.get("fusion_mode"),
        "model_label": model_display_label(config),
        "seed": config.get("seed"),
        "model_seed": first_present(config.get("model_seed"), config.get("seed")),
        "split_seed": first_present(
            config.get("effective_split_seed"),
            split_diagnostics.get("effective_split_seed"),
            config.get("split_seed"),
            config.get("seed"),
        ),
        "final_test_primary_report": first_present(
            test_report.get("final_test_primary_report"),
            config.get("final_test_primary_report"),
        ),
        "final_test_ensemble_mode": first_present(
            test_report.get("final_test_ensemble_mode"),
            config.get("final_test_ensemble_mode"),
        ),
        "final_test_result_role": first_present(
            test_report.get("final_test_result_role"),
            config.get("final_test_result_role"),
        ),
        "selected_config_id": first_present(
            test_report.get("selected_config_id"),
            config.get("final_test_selected_config_id"),
        ),
        "learning_rate": config.get("learning_rate"),
        "weight_decay": config.get("weight_decay"),
        "batch_size": config.get("batch_size"),
        "epochs": config.get("epochs"),
        "use_ring_edges": config.get("use_ring_edges"),
        "omit_node_features": omit_str,
        "node_feature_set": first_present(config.get("node_feature_set"), dataset.get("node_feature_set")),
        "ec_label_depth": first_present(config.get("ec_label_depth"), dataset.get("ec_label_depth")),
        "ec_group_weighting": first_present(config.get("ec_group_weighting"), dataset.get("ec_group_weighting")),
        "ec_contrastive_weight": config.get("ec_contrastive_weight"),
        "ec_contrastive_temperature": config.get("ec_contrastive_temperature"),
        "joint_loss_weighting": config.get("joint_loss_weighting"),
        "metal_loss_weight": config.get("metal_loss_weight"),
        "ec_loss_weight": config.get("ec_loss_weight"),
        "metal_collapsed_loss_weight": config.get("metal_collapsed_loss_weight"),
        "metal_class_weight_mode": config.get("metal_class_weight_mode"),
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
        "split_by": first_present(
            split_diagnostics.get("split_by"),
            dataset.get("split_by"),
            config.get("split_by"),
        ),
        "val_fraction": first_present(
            split_diagnostics.get("val_fraction"),
            dataset.get("val_fraction"),
            config.get("val_fraction"),
        ),
        "n_folds": first_present(
            split_diagnostics.get("n_folds"),
            dataset.get("n_folds"),
            config.get("n_folds"),
        ),
        "fold_index": first_present(
            split_diagnostics.get("fold_index"),
            dataset.get("fold_index"),
            config.get("fold_index"),
        ),
        "n_train_pockets": first_present(
            split_diagnostics.get("n_train_pockets"),
            dataset.get("n_train_pockets"),
        ),
        "n_val_pockets": first_present(
            split_diagnostics.get("n_val_pockets"),
            dataset.get("n_val_pockets"),
        ),
        "n_train_groups": split_diagnostics.get("n_train_groups"),
        "n_val_groups": split_diagnostics.get("n_val_groups"),
        "train_val_overlap_pdbid": split_diagnostics.get("train_val_overlap_pdbid"),
        "train_val_overlap_pdbid_chain": split_diagnostics.get("train_val_overlap_pdbid_chain"),
        "train_val_overlap_structure_id": split_diagnostics.get("train_val_overlap_structure_id"),
        "train_val_overlap_pocket_id": split_diagnostics.get("train_val_overlap_pocket_id"),
        "missing_train_metal_classes": csv_list(split_diagnostics.get("missing_train_metal_classes")),
        "missing_val_metal_classes": csv_list(split_diagnostics.get("missing_val_metal_classes")),
        "missing_train_ec_classes": csv_list(split_diagnostics.get("missing_train_ec_classes")),
        "missing_val_ec_classes": csv_list(split_diagnostics.get("missing_val_ec_classes")),
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
        "best_validation_metric_used_for_checkpoint_selection": selected_selection_value,
        "selected_val_joint_balanced_acc": selected_metric_value(
            selected_record,
            "val_joint_balanced_acc",
            selection_metric,
            selected_selection_value,
        ),
        "selected_val_metal_balanced_acc": selected_metric_value(
            selected_record,
            "val_metal_balanced_acc",
            selection_metric,
            selected_selection_value,
        ),
        "selected_val_metal_min_recall": selected_metric_value(
            selected_record,
            "val_metal_min_recall",
            selection_metric,
            selected_selection_value,
        ),
        "selected_val_metal_per_class_recall": selected_record.get("val_metal_per_class_recall"),
        "selected_val_metal_per_class_support": selected_record.get("val_metal_per_class_support"),
        "selected_val_metal_collapsed4_balanced_acc": selected_metric_value(
            selected_record,
            "val_metal_collapsed4_balanced_acc",
            selection_metric,
            selected_selection_value,
        ),
        "selected_val_metal_collapsed4_min_recall": selected_metric_value(
            selected_record,
            "val_metal_collapsed4_min_recall",
            selection_metric,
            selected_selection_value,
        ),
        "selected_val_metal_collapsed4_per_class_recall": selected_record.get(
            "val_metal_collapsed4_per_class_recall"
        ),
        "selected_val_metal_collapsed4_per_class_support": selected_record.get(
            "val_metal_collapsed4_per_class_support"
        ),
        "selected_val_ec_balanced_acc": selected_metric_value(
            selected_record,
            "val_ec_balanced_acc",
            selection_metric,
            selected_selection_value,
        ),
        "selected_val_ec_group_balanced_acc": selected_metric_value(
            selected_record,
            "val_ec_group_balanced_acc",
            selection_metric,
            selected_selection_value,
        ),
        "comparison_test_metric_name": test_metric_name,
        "comparison_test_metric_value": metrics.get(test_metric_name) if test_metric_name else None,
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
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def write_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: normalize_csv_value(row.get(column)) for column in CSV_COLUMNS})


def model_display_label(row: dict[str, Any]) -> str:
    arch = str(row.get("model_architecture") or row.get("model") or "unknown")
    fusion = str(row.get("fusion_mode") or row.get("fusion") or "")
    use_esm_branch = row.get("use_esm_branch")

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
        suffix = fusion_labels.get(fusion, fusion.replace("_", " ") if fusion else "ESM fusion")
        return f"GVP + ESM {suffix}"
    return arch.replace("_", " ")


def _short_label(row: dict[str, Any]) -> str:
    model_label = str(row.get("model_label") or model_display_label(row))
    seed = row.get("seed")
    lr = row.get("learning_rate")
    parts: list[str] = []
    parts.append(model_label[:24])
    if lr is not None and is_number(lr):
        parts.append(f"lr={float(lr):.0e}")
    if seed is not None:
        parts.append(f"s{seed}")
    return " ".join(parts)


def write_figure(rows: list[dict[str, Any]], out_figure: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except Exception as exc:
        print(f"warning: matplotlib unavailable; skipping figure: {exc}", file=sys.stderr)
        return

    plot_rows = [
        row
        for row in rows
        if (
            is_number(row.get("best_validation_metric_used_for_checkpoint_selection"))
            or is_number(row.get("comparison_test_metric_value"))
        )
    ]
    if not plot_rows:
        print("warning: no numeric validation or held-out test metric available; skipping figure", file=sys.stderr)
        return

    # Sort ascending so best run appears at the top of the horizontal bar chart
    plot_rows = sorted(
        plot_rows,
        key=lambda r: (
            is_number(r.get("best_validation_metric_used_for_checkpoint_selection")),
            float(r["best_validation_metric_used_for_checkpoint_selection"])
            if is_number(r.get("best_validation_metric_used_for_checkpoint_selection"))
            else float("-inf"),
        ),
    )

    _ARCH_COLORS = [
        "#4c78a8", "#f58518", "#54a24b", "#e45756",
        "#72b7b2", "#b279a2", "#ff9da6", "#9d755d",
    ]
    architectures = [str(row.get("model_label") or model_display_label(row)) for row in plot_rows]
    unique_archs = list(dict.fromkeys(architectures))
    color_map = {arch: _ARCH_COLORS[i % len(_ARCH_COLORS)] for i, arch in enumerate(unique_archs)}
    bar_colors = [color_map[arch] for arch in architectures]

    labels = [_short_label(row) for row in plot_rows]
    val_values = [
        float(row["best_validation_metric_used_for_checkpoint_selection"])
        if is_number(row.get("best_validation_metric_used_for_checkpoint_selection"))
        else math.nan
        for row in plot_rows
    ]
    test_values = [
        float(row["comparison_test_metric_value"])
        if is_number(row.get("comparison_test_metric_value"))
        else math.nan
        for row in plot_rows
    ]
    has_test = any(not math.isnan(v) for v in test_values)
    paired_count = sum(
        1 for v, t in zip(val_values, test_values)
        if not math.isnan(v) and not math.isnan(t)
    )
    has_scatter = has_test and paired_count >= 2

    n_panels = 3 if has_scatter else (2 if has_test else 1)
    n_runs = len(plot_rows)
    bar_height = max(3.5, n_runs * 0.45 + 2.0)
    fig_width = 6.5 * n_panels

    fig, raw_axes = plt.subplots(1, n_panels, figsize=(fig_width, bar_height))
    axes: list[Any] = list(raw_axes) if n_panels > 1 else [raw_axes]

    val_metric_names = sorted(
        {str(row.get("selection_metric")) for row in plot_rows if row.get("selection_metric")}
    )
    test_metric_names = sorted(
        {
            str(row.get("comparison_test_metric_name"))
            for row in plot_rows
            if row.get("comparison_test_metric_name") not in {None, "NA"}
        }
    )
    val_xlabel = val_metric_names[0] if len(val_metric_names) == 1 else "validation metric"
    test_xlabel = test_metric_names[0] if len(test_metric_names) == 1 else "test metric"

    y = list(range(n_runs))

    def _draw_hbars(ax: Any, values: list[float], title: str, xlabel: str) -> None:
        ax.barh(y, values, color=bar_colors, alpha=0.85, height=0.65)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(title, fontsize=10, pad=6)
        finite = [v for v in values if not math.isnan(v)]
        if not finite:
            return
        vrange = max(finite) - min(finite)
        offset = max(vrange * 0.012, 1e-4)
        for yi, val in zip(y, values):
            if not math.isnan(val):
                ax.text(val + offset, yi, f"{val:.4f}", va="center", ha="left", fontsize=7)
        ax.set_xlim(right=max(finite) + max(vrange * 0.18, 0.04))

    _draw_hbars(axes[0], val_values, "Validation metric (checkpoint selection)", val_xlabel)

    if has_test:
        _draw_hbars(axes[1], test_values, "Held-out test metric", test_xlabel)

    if has_scatter:
        ax2 = axes[2]
        pairs = [
            (v, t, lbl, clr)
            for v, t, lbl, clr in zip(val_values, test_values, labels, bar_colors)
            if not math.isnan(v) and not math.isnan(t)
        ]
        xs, ys_s, slabels, scolors = zip(*pairs)
        ax2.scatter(xs, ys_s, c=scolors, s=70, alpha=0.9, zorder=3)
        for xi, yi_pt, lbl in zip(xs, ys_s, slabels):
            ax2.annotate(lbl, (xi, yi_pt), fontsize=6, textcoords="offset points", xytext=(5, 3))
        all_vals = list(xs) + list(ys_s)
        lo, hi = min(all_vals), max(all_vals)
        margin = (hi - lo) * 0.05 if hi > lo else 0.01
        ax2.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "k--", lw=0.8, alpha=0.4)
        ax2.set_xlabel(val_xlabel, fontsize=9)
        ax2.set_ylabel(test_xlabel, fontsize=9)
        ax2.set_title("Val vs. test correlation", fontsize=10, pad=6)

    if len(unique_archs) > 1:
        patches = [mpatches.Patch(color=color_map[a], label=a) for a in unique_archs]
        fig.legend(
            handles=patches,
            loc="upper center",
            ncol=min(len(unique_archs), 4),
            fontsize=8,
            bbox_to_anchor=(0.5, 1.04),
            title="Model",
            title_fontsize=8,
        )

    fig.suptitle("DeepMzyme run comparison", y=1.08, fontsize=12)
    fig.tight_layout()
    out_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_figure, dpi=150, bbox_inches="tight")
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
