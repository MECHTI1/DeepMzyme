from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from label_schemes import (
    COLLAPSED_METAL_LABELS,
    METAL_TARGET_LABELS,
    collapsed_metal_target_for_label_name,
)
from training.loop import classification_metrics_from_logits

_EPS = 1.0e-12


def _to_cpu_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu()


def _log_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
    return torch.log(probabilities.clamp_min(_EPS))


def _present_metric_values(values: list[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None]


def _metal_label_index(label_name: str) -> int | None:
    for label_idx, current_label_name in METAL_TARGET_LABELS.items():
        if current_label_name == label_name:
            return int(label_idx)
    return None


def collapse_metal_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
    grouped: list[torch.Tensor] = []
    for collapsed_idx in sorted(COLLAPSED_METAL_LABELS):
        source_indices = [
            label_idx
            for label_idx, label_name in METAL_TARGET_LABELS.items()
            if collapsed_metal_target_for_label_name(label_name) == collapsed_idx
        ]
        if not source_indices:
            grouped.append(torch.zeros((probabilities.size(0), 1), dtype=probabilities.dtype))
            continue
        grouped.append(probabilities[:, source_indices].sum(dim=-1, keepdim=True))
    return torch.cat(grouped, dim=-1)


def collapse_metal_targets(targets: torch.Tensor) -> torch.Tensor:
    return torch.tensor(
        [
            collapsed_metal_target_for_label_name(METAL_TARGET_LABELS[int(target_idx)])
            for target_idx in targets.tolist()
        ],
        dtype=torch.long,
    )


def metal_metrics_from_probabilities(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    prefix: str = "test",
) -> dict[str, Any]:
    probabilities = _to_cpu_tensor(probabilities).float()
    targets = _to_cpu_tensor(targets).long()
    logits = _log_probabilities(probabilities)
    metal_metrics = classification_metrics_from_logits(logits, targets)
    metal_recalls = _present_metric_values(metal_metrics["per_class_recall"])
    mn_idx = _metal_label_index("Mn")
    fe_idx = _metal_label_index("Fe")
    class_viii_idx = _metal_label_index("Class VIII")

    collapsed_probabilities = collapse_metal_probabilities(probabilities)
    collapsed_targets = collapse_metal_targets(targets)
    collapsed_metrics = classification_metrics_from_logits(
        _log_probabilities(collapsed_probabilities),
        collapsed_targets,
    )
    return {
        f"{prefix}_metal_acc": metal_metrics["accuracy"],
        f"{prefix}_metal_balanced_acc": metal_metrics["balanced_accuracy"],
        f"{prefix}_metal_macro_f1": metal_metrics["macro_f1"],
        f"{prefix}_metal_min_recall": float(min(metal_recalls)) if metal_recalls else None,
        f"{prefix}_metal_mn_recall": (
            metal_metrics["per_class_recall"][mn_idx] if mn_idx is not None else None
        ),
        f"{prefix}_metal_fe_recall": (
            metal_metrics["per_class_recall"][fe_idx] if fe_idx is not None else None
        ),
        f"{prefix}_metal_class_viii_recall": (
            metal_metrics["per_class_recall"][class_viii_idx] if class_viii_idx is not None else None
        ),
        f"{prefix}_metal_per_class_recall": {
            label_name: metal_metrics["per_class_recall"][label_idx]
            for label_idx, label_name in METAL_TARGET_LABELS.items()
        },
        f"{prefix}_metal_collapsed4_acc": collapsed_metrics["accuracy"],
        f"{prefix}_metal_collapsed4_balanced_acc": collapsed_metrics["balanced_accuracy"],
        f"{prefix}_metal_collapsed4_macro_f1": collapsed_metrics["macro_f1"],
        f"{prefix}_metal_collapsed4_mn_recall": collapsed_metrics["per_class_recall"][0],
        f"{prefix}_metal_collapsed4_cu_recall": collapsed_metrics["per_class_recall"][1],
        f"{prefix}_metal_collapsed4_zn_recall": collapsed_metrics["per_class_recall"][2],
        f"{prefix}_metal_collapsed4_class_viii_recall": collapsed_metrics["per_class_recall"][3],
        f"{prefix}_metal_collapsed4_per_class_recall": {
            label_name: collapsed_metrics["per_class_recall"][label_idx]
            for label_idx, label_name in COLLAPSED_METAL_LABELS.items()
        },
    }


def fit_temperature_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    max_iter: int = 50,
) -> float:
    """Fit one scalar temperature on validation logits by minimizing NLL."""
    logits = _to_cpu_tensor(logits).float()
    targets = _to_cpu_tensor(targets).long()
    if logits.ndim != 2 or logits.size(0) != targets.numel() or targets.numel() == 0:
        raise ValueError("temperature fitting requires non-empty [N, C] logits and [N] targets.")

    log_temperature = torch.zeros(1, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=int(max_iter),
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(0.05, 100.0)
        loss = F.cross_entropy(logits / temperature, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(torch.exp(log_temperature).clamp(0.05, 100.0).item())
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise FloatingPointError(f"Invalid fitted temperature: {temperature!r}")
    return temperature


def _equal_mass_bin_indices(confidences: torch.Tensor, n_bins: int) -> list[torch.Tensor]:
    n_examples = int(confidences.numel())
    if n_examples == 0:
        return []
    n_bins = max(1, min(int(n_bins), n_examples))
    sorted_indices = torch.argsort(confidences)
    return [chunk for chunk in torch.tensor_split(sorted_indices, n_bins) if int(chunk.numel()) > 0]


def equal_mass_ece(
    confidences: torch.Tensor,
    outcomes: torch.Tensor,
    *,
    n_bins: int = 15,
) -> tuple[float, list[dict[str, Any]]]:
    confidences = _to_cpu_tensor(confidences).float().view(-1)
    outcomes = _to_cpu_tensor(outcomes).float().view(-1)
    if confidences.numel() != outcomes.numel():
        raise ValueError("confidence and outcome tensors must have the same length.")
    if confidences.numel() == 0:
        raise ValueError("ECE requires at least one example.")

    bins: list[dict[str, Any]] = []
    ece = 0.0
    total = float(confidences.numel())
    for bin_index, indices in enumerate(_equal_mass_bin_indices(confidences, n_bins), start=1):
        bin_conf = confidences[indices]
        bin_outcomes = outcomes[indices]
        mean_confidence = float(bin_conf.mean().item())
        empirical_accuracy = float(bin_outcomes.mean().item())
        weight = float(indices.numel()) / total
        gap = abs(empirical_accuracy - mean_confidence)
        ece += weight * gap
        bins.append(
            {
                "bin": bin_index,
                "n": int(indices.numel()),
                "mean_confidence": mean_confidence,
                "empirical_accuracy": empirical_accuracy,
                "abs_gap": float(gap),
            }
        )
    return float(ece), bins


def calibration_metrics_from_probabilities(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    n_bins: int = 15,
    prefix: str = "test_metal",
) -> dict[str, Any]:
    probabilities = _to_cpu_tensor(probabilities).float()
    targets = _to_cpu_tensor(targets).long()
    confidences, predictions = probabilities.max(dim=-1)
    correctness = (predictions == targets).float()
    ece, bins = equal_mass_ece(confidences, correctness, n_bins=n_bins)
    nll = float(F.nll_loss(_log_probabilities(probabilities), targets).item())

    classwise_ece: dict[str, float | None] = {}
    classwise_bins: dict[str, list[dict[str, Any]]] = {}
    for class_idx, label_name in METAL_TARGET_LABELS.items():
        if class_idx >= probabilities.size(-1):
            classwise_ece[label_name] = None
            classwise_bins[label_name] = []
            continue
        class_confidence = probabilities[:, class_idx]
        class_outcome = (targets == int(class_idx)).float()
        class_ece, class_bins = equal_mass_ece(class_confidence, class_outcome, n_bins=n_bins)
        classwise_ece[label_name] = class_ece
        classwise_bins[label_name] = class_bins

    return {
        f"{prefix}_ece_equal_mass": ece,
        f"{prefix}_nll": nll,
        f"{prefix}_classwise_ece_equal_mass": classwise_ece,
        f"{prefix}_calibration_bins": bins,
        f"{prefix}_classwise_calibration_bins": classwise_bins,
    }


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute quantile of an empty list.")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - lower_index
    return sorted_values[lower_index] * (1.0 - fraction) + sorted_values[upper_index] * fraction


def _stratified_bootstrap_indices(targets: torch.Tensor, rng: random.Random) -> torch.Tensor:
    sampled_indices: list[int] = []
    for class_id in sorted({int(value) for value in targets.tolist()}):
        class_indices = (targets == class_id).nonzero(as_tuple=False).view(-1).tolist()
        if not class_indices:
            continue
        sampled_indices.extend(rng.choice(class_indices) for _ in class_indices)
    rng.shuffle(sampled_indices)
    return torch.tensor(sampled_indices, dtype=torch.long)


def _metric_ci(values: list[float], confidence_level: float) -> list[float]:
    values = sorted(float(value) for value in values)
    alpha = 1.0 - float(confidence_level)
    return [
        float(_quantile(values, alpha / 2.0)),
        float(_quantile(values, 1.0 - alpha / 2.0)),
    ]


def metal_bootstrap_support_report(
    targets: torch.Tensor,
    *,
    low_support_threshold: int = 3,
) -> dict[str, Any]:
    targets = _to_cpu_tensor(targets).long().view(-1)
    counts = torch.bincount(targets, minlength=len(METAL_TARGET_LABELS))
    support_by_class = {
        label_name: int(counts[int(label_idx)].item())
        for label_idx, label_name in METAL_TARGET_LABELS.items()
    }
    low_support_classes = {
        label_name: count
        for label_name, count in support_by_class.items()
        if 0 < count < int(low_support_threshold)
    }
    warning = None
    if low_support_classes:
        warning = (
            "One or more metal classes have very low held-out support; "
            "stratified bootstrap confidence intervals may understate minority-class uncertainty."
        )
    return {
        "support_by_class": support_by_class,
        "low_support_threshold": int(low_support_threshold),
        "low_support_classes": low_support_classes,
        "low_support_warning": warning,
    }


def bootstrap_metric_cis(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    metric_fn: Callable[[torch.Tensor, torch.Tensor], dict[str, Any]],
    *,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 20260518,
) -> dict[str, Any]:
    probabilities = _to_cpu_tensor(probabilities).float()
    targets = _to_cpu_tensor(targets).long()
    if probabilities.size(0) != targets.numel() or targets.numel() == 0:
        raise ValueError("bootstrap requires non-empty probabilities and matching targets.")
    if int(n_bootstrap) < 1:
        raise ValueError(f"n_bootstrap must be positive, got {n_bootstrap}.")
    if not 0.0 < float(confidence_level) < 1.0:
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}.")

    rng = random.Random(int(seed))
    samples_by_metric: dict[str, list[float]] = {}
    for _ in range(int(n_bootstrap)):
        indices = _stratified_bootstrap_indices(targets, rng)
        sample_metrics = metric_fn(probabilities[indices], targets[indices])
        for metric_name, metric_value in sample_metrics.items():
            if isinstance(metric_value, (int, float)) and not isinstance(metric_value, bool):
                numeric = float(metric_value)
                if math.isfinite(numeric):
                    samples_by_metric.setdefault(metric_name, []).append(numeric)

    return {
        f"{metric_name}_ci95": _metric_ci(values, confidence_level)
        for metric_name, values in samples_by_metric.items()
        if values
    }


def metal_bootstrap_metric_cis(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 20260518,
    n_bins: int = 15,
) -> dict[str, Any]:
    def metric_fn(sample_probabilities: torch.Tensor, sample_targets: torch.Tensor) -> dict[str, Any]:
        metrics = metal_metrics_from_probabilities(sample_probabilities, sample_targets, prefix="test")
        calibration = calibration_metrics_from_probabilities(
            sample_probabilities,
            sample_targets,
            n_bins=n_bins,
            prefix="test_metal",
        )
        selected: dict[str, Any] = {
            "test_metal_balanced_acc": metrics.get("test_metal_balanced_acc"),
            "test_metal_collapsed4_balanced_acc": metrics.get("test_metal_collapsed4_balanced_acc"),
            "test_metal_ece_equal_mass": calibration.get("test_metal_ece_equal_mass"),
        }
        for label_name in METAL_TARGET_LABELS.values():
            value = metrics.get("test_metal_per_class_recall", {}).get(label_name)
            if value is not None:
                selected[f"test_metal_{label_name.lower().replace(' ', '_')}_recall"] = value
        for label_name in COLLAPSED_METAL_LABELS.values():
            value = metrics.get("test_metal_collapsed4_per_class_recall", {}).get(label_name)
            if value is not None:
                selected[f"test_metal_collapsed4_{label_name.lower().replace(' ', '_')}_recall"] = value
        return selected

    return bootstrap_metric_cis(
        probabilities,
        targets,
        metric_fn,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=seed,
    )


def write_reliability_diagram(
    bins: list[dict[str, Any]],
    path: Path,
    *,
    title: str = "Reliability Diagram",
) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    if not bins:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    confidence = [float(item["mean_confidence"]) for item in bins]
    accuracy = [float(item["empirical_accuracy"]) for item in bins]
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.plot([0.0, 1.0], [0.0, 1.0], color="#666666", linewidth=1.0, linestyle="--")
    ax.plot(confidence, accuracy, marker="o", color="#4c78a8", linewidth=1.8)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def write_confidence_histogram(
    probabilities: torch.Tensor,
    path: Path,
    *,
    title: str = "Confidence Histogram",
) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    probabilities = _to_cpu_tensor(probabilities).float()
    if probabilities.numel() == 0:
        return None
    confidences = probabilities.max(dim=-1).values.numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.hist(confidences, bins=15, range=(0.0, 1.0), color="#4c78a8", edgecolor="white")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Predicted-class confidence")
    ax.set_ylabel("Examples")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def prediction_artifact_payload(
    *,
    task: str,
    metal_logits: torch.Tensor | None = None,
    metal_y: torch.Tensor | None = None,
    metal_probabilities: torch.Tensor | None = None,
    metal_calibrated_probabilities: torch.Tensor | None = None,
    metal_temperature: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"task": task, "metadata": metadata or {}}
    if metal_logits is not None:
        payload["metal_logits"] = _to_cpu_tensor(metal_logits).float()
    if metal_y is not None:
        payload["metal_y"] = _to_cpu_tensor(metal_y).long()
    if metal_probabilities is not None:
        payload["metal_probabilities"] = _to_cpu_tensor(metal_probabilities).float()
    if metal_calibrated_probabilities is not None:
        payload["metal_calibrated_probabilities"] = _to_cpu_tensor(metal_calibrated_probabilities).float()
    if metal_temperature is not None:
        payload["metal_temperature"] = float(metal_temperature)
    return payload


def write_prediction_artifact(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def load_prediction_artifact(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Prediction artifact is not a dictionary: {path}")
    return payload


def build_metal_final_reporting_payload(
    *,
    output_dir: Path,
    metrics: dict[str, Any],
    test_logits: torch.Tensor,
    test_targets: torch.Tensor,
    val_logits: torch.Tensor | None,
    val_targets: torch.Tensor | None,
    task: str,
    enable_calibration: bool = True,
    enable_temperature_scaling: bool = True,
    enable_bootstrap_ci: bool = True,
    n_bins: int = 15,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 20260518,
    artifact_prefix: str = "test",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    test_logits = _to_cpu_tensor(test_logits).float()
    test_targets = _to_cpu_tensor(test_targets).long()
    probabilities = torch.softmax(test_logits, dim=-1)
    metrics = dict(metrics)
    metrics.update(metal_metrics_from_probabilities(probabilities, test_targets, prefix="test"))

    calibration_payload: dict[str, Any] = {}
    plot_paths: dict[str, str | None] = {}
    if enable_calibration:
        calibration_payload = calibration_metrics_from_probabilities(
            probabilities,
            test_targets,
            n_bins=n_bins,
            prefix="test_metal",
        )
        metrics.update(
            {
                "test_metal_ece_equal_mass": calibration_payload["test_metal_ece_equal_mass"],
                "test_metal_nll": calibration_payload["test_metal_nll"],
                "test_metal_classwise_ece_equal_mass": calibration_payload[
                    "test_metal_classwise_ece_equal_mass"
                ],
            }
        )
        reliability_path = write_reliability_diagram(
            calibration_payload["test_metal_calibration_bins"],
            output_dir / f"{artifact_prefix}_reliability_diagram.png",
            title="Final Test Reliability Diagram",
        )
        histogram_path = write_confidence_histogram(
            probabilities,
            output_dir / f"{artifact_prefix}_confidence_histogram.png",
            title="Final Test Confidence Histogram",
        )
        plot_paths["reliability_diagram_path"] = str(reliability_path) if reliability_path else None
        plot_paths["confidence_histogram_path"] = str(histogram_path) if histogram_path else None

    temperature: float | None = None
    calibrated_probabilities: torch.Tensor | None = None
    calibrated_metrics: dict[str, Any] = {}
    calibration_artifact_path: Path | None = None
    if (
        enable_calibration
        and enable_temperature_scaling
        and val_logits is not None
        and val_targets is not None
        and int(val_targets.numel()) > 0
    ):
        temperature = fit_temperature_from_logits(val_logits, val_targets)
        calibrated_probabilities = torch.softmax(test_logits / temperature, dim=-1)
        raw_calibrated_metrics = metal_metrics_from_probabilities(
            calibrated_probabilities,
            test_targets,
            prefix="test",
        )
        calibrated_metrics = {
            key.replace("test_metal", "test_metal_temperature_scaled"): value
            for key, value in raw_calibrated_metrics.items()
        }
        calibrated_calibration = calibration_metrics_from_probabilities(
            calibrated_probabilities,
            test_targets,
            n_bins=n_bins,
            prefix="test_metal_temperature_scaled",
        )
        calibrated_metrics.update(
            {
                "test_metal_temperature_scaled_ece_equal_mass": calibrated_calibration[
                    "test_metal_temperature_scaled_ece_equal_mass"
                ],
                "test_metal_temperature_scaled_nll": calibrated_calibration[
                    "test_metal_temperature_scaled_nll"
                ],
                "test_metal_temperature_scaled_classwise_ece_equal_mass": calibrated_calibration[
                    "test_metal_temperature_scaled_classwise_ece_equal_mass"
                ],
            }
        )
        calibrated_reliability_path = write_reliability_diagram(
            calibrated_calibration["test_metal_temperature_scaled_calibration_bins"],
            output_dir / f"{artifact_prefix}_temperature_scaled_reliability_diagram.png",
            title="Temperature-Scaled Final Test Reliability",
        )
        calibrated_histogram_path = write_confidence_histogram(
            calibrated_probabilities,
            output_dir / f"{artifact_prefix}_temperature_scaled_confidence_histogram.png",
            title="Temperature-Scaled Final Test Confidence",
        )
        plot_paths["temperature_scaled_reliability_diagram_path"] = (
            str(calibrated_reliability_path) if calibrated_reliability_path else None
        )
        plot_paths["temperature_scaled_confidence_histogram_path"] = (
            str(calibrated_histogram_path) if calibrated_histogram_path else None
        )
        calibration_artifact_path = write_prediction_artifact(
            output_dir / f"{artifact_prefix}_temperature_validation_predictions.pt",
            prediction_artifact_payload(
                task=task,
                metal_logits=val_logits,
                metal_y=val_targets,
                metadata={"temperature_fitting_rule": "single scalar NLL minimization on validation logits"},
            ),
        )

    bootstrap_settings = {
        "enabled": bool(enable_bootstrap_ci),
        "method": "stratified_by_true_class",
        "n_bootstrap": int(n_bootstrap),
        "confidence_level": float(confidence_level),
        "seed": int(bootstrap_seed),
        "missing_class_rule": "stratified resampling preserves every class present in the original test labels",
        **metal_bootstrap_support_report(test_targets),
    }
    if enable_bootstrap_ci:
        metrics.update(
            metal_bootstrap_metric_cis(
                probabilities,
                test_targets,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                seed=bootstrap_seed,
                n_bins=n_bins,
            )
        )
        if calibrated_probabilities is not None:
            calibrated_metrics.update(
                {
                    key.replace("test_metal", "test_metal_temperature_scaled"): value
                    for key, value in metal_bootstrap_metric_cis(
                        calibrated_probabilities,
                        test_targets,
                        n_bootstrap=n_bootstrap,
                        confidence_level=confidence_level,
                        seed=bootstrap_seed,
                        n_bins=n_bins,
                    ).items()
                }
            )

    prediction_path = write_prediction_artifact(
        output_dir / f"{artifact_prefix}_predictions.pt",
        prediction_artifact_payload(
            task=task,
            metal_logits=test_logits,
            metal_y=test_targets,
            metal_probabilities=probabilities,
            metal_calibrated_probabilities=calibrated_probabilities,
            metal_temperature=temperature,
        ),
    )

    return {
        "metrics": metrics,
        "calibrated_metrics": calibrated_metrics,
        "fitted_temperatures": {"metal": temperature} if temperature is not None else {},
        "temperature_scaling": {
            "enabled": bool(enable_temperature_scaling),
            "fitted_on": "validation_logits",
            "selection_rule": "fixed before held-out test evaluation; no test metric used",
            "validation_prediction_artifact_path": str(calibration_artifact_path) if calibration_artifact_path else None,
            "unavailable_reason": None
            if temperature is not None
            else "validation metal logits were unavailable or temperature scaling was disabled",
        },
        "bootstrap_settings": bootstrap_settings,
        "calibration_settings": {
            "enabled": bool(enable_calibration),
            "n_equal_mass_bins": int(n_bins),
            "overall_ece_definition": "predicted-class confidence binned into equal-mass bins",
            "classwise_ece_definition": "one-vs-rest class probability binned into equal-mass bins",
        },
        "calibration_plot_paths": plot_paths,
        "reliability_diagram_path": plot_paths.get("reliability_diagram_path"),
        "confidence_histogram_path": plot_paths.get("confidence_histogram_path"),
        "prediction_artifact_path": str(prediction_path),
        "n_test_pockets": int(test_targets.numel()),
    }


def build_softmax_mean_ensemble_payload(
    *,
    output_dir: Path,
    prediction_artifact_paths: list[Path],
    task: str,
    enable_calibration: bool = True,
    enable_bootstrap_ci: bool = True,
    n_bins: int = 15,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 20260518,
) -> dict[str, Any]:
    if len(prediction_artifact_paths) != 5:
        raise ValueError(
            "softmax_mean_5_seeds requires exactly five fixed prediction artifacts; "
            f"got {len(prediction_artifact_paths)}."
        )
    artifacts = [load_prediction_artifact(Path(path)) for path in prediction_artifact_paths]
    probabilities = [artifact.get("metal_probabilities") for artifact in artifacts]
    targets = [artifact.get("metal_y") for artifact in artifacts]
    if not all(isinstance(item, torch.Tensor) for item in probabilities):
        raise ValueError("Every ensemble artifact must contain metal_probabilities.")
    if not all(isinstance(item, torch.Tensor) for item in targets):
        raise ValueError("Every ensemble artifact must contain metal_y.")

    target = _to_cpu_tensor(targets[0]).long()
    for index, current_target in enumerate(targets[1:], start=2):
        current_target = _to_cpu_tensor(current_target).long()
        if current_target.numel() != target.numel() or not torch.equal(current_target, target):
            raise ValueError(f"Ensemble prediction artifact {index} has different target ordering.")

    stacked_probabilities = torch.stack([_to_cpu_tensor(item).float() for item in probabilities], dim=0)
    mean_probabilities = stacked_probabilities.mean(dim=0)
    metrics = metal_metrics_from_probabilities(mean_probabilities, target, prefix="test")
    calibration_payload: dict[str, Any] = {}
    plot_paths: dict[str, str | None] = {}
    if enable_calibration:
        calibration_payload = calibration_metrics_from_probabilities(
            mean_probabilities,
            target,
            n_bins=n_bins,
            prefix="test_metal",
        )
        metrics.update(
            {
                "test_metal_ece_equal_mass": calibration_payload["test_metal_ece_equal_mass"],
                "test_metal_nll": calibration_payload["test_metal_nll"],
                "test_metal_classwise_ece_equal_mass": calibration_payload[
                    "test_metal_classwise_ece_equal_mass"
                ],
            }
        )
        reliability_path = write_reliability_diagram(
            calibration_payload["test_metal_calibration_bins"],
            output_dir / "ensemble_reliability_diagram.png",
            title="Softmax-Mean Ensemble Reliability",
        )
        histogram_path = write_confidence_histogram(
            mean_probabilities,
            output_dir / "ensemble_confidence_histogram.png",
            title="Softmax-Mean Ensemble Confidence",
        )
        plot_paths["reliability_diagram_path"] = str(reliability_path) if reliability_path else None
        plot_paths["confidence_histogram_path"] = str(histogram_path) if histogram_path else None

    calibrated_metrics: dict[str, Any] = {}
    fitted_temperatures: list[float | None] = [
        artifact.get("metal_temperature") if isinstance(artifact.get("metal_temperature"), (int, float)) else None
        for artifact in artifacts
    ]
    calibrated_probabilities = [
        artifact.get("metal_calibrated_probabilities") for artifact in artifacts
    ]
    ensemble_calibrated_probabilities: torch.Tensor | None = None
    if all(isinstance(item, torch.Tensor) for item in calibrated_probabilities):
        ensemble_calibrated_probabilities = torch.stack(
            [_to_cpu_tensor(item).float() for item in calibrated_probabilities],
            dim=0,
        ).mean(dim=0)
        raw_calibrated_metrics = metal_metrics_from_probabilities(
            ensemble_calibrated_probabilities,
            target,
            prefix="test",
        )
        calibrated_metrics.update(
            {
                key.replace("test_metal", "test_metal_temperature_scaled"): value
                for key, value in raw_calibrated_metrics.items()
            }
        )
        calibrated_calibration = calibration_metrics_from_probabilities(
            ensemble_calibrated_probabilities,
            target,
            n_bins=n_bins,
            prefix="test_metal_temperature_scaled",
        )
        calibrated_metrics.update(
            {
                "test_metal_temperature_scaled_ece_equal_mass": calibrated_calibration[
                    "test_metal_temperature_scaled_ece_equal_mass"
                ],
                "test_metal_temperature_scaled_nll": calibrated_calibration[
                    "test_metal_temperature_scaled_nll"
                ],
                "test_metal_temperature_scaled_classwise_ece_equal_mass": calibrated_calibration[
                    "test_metal_temperature_scaled_classwise_ece_equal_mass"
                ],
            }
        )
        calibrated_reliability_path = write_reliability_diagram(
            calibrated_calibration["test_metal_temperature_scaled_calibration_bins"],
            output_dir / "ensemble_temperature_scaled_reliability_diagram.png",
            title="Temperature-Scaled Ensemble Reliability",
        )
        calibrated_histogram_path = write_confidence_histogram(
            ensemble_calibrated_probabilities,
            output_dir / "ensemble_temperature_scaled_confidence_histogram.png",
            title="Temperature-Scaled Ensemble Confidence",
        )
        plot_paths["temperature_scaled_reliability_diagram_path"] = (
            str(calibrated_reliability_path) if calibrated_reliability_path else None
        )
        plot_paths["temperature_scaled_confidence_histogram_path"] = (
            str(calibrated_histogram_path) if calibrated_histogram_path else None
        )

    bootstrap_settings = {
        "enabled": bool(enable_bootstrap_ci),
        "method": "stratified_by_true_class",
        "n_bootstrap": int(n_bootstrap),
        "confidence_level": float(confidence_level),
        "seed": int(bootstrap_seed),
        "missing_class_rule": "stratified resampling preserves every class present in the original test labels",
        **metal_bootstrap_support_report(target),
    }
    if enable_bootstrap_ci:
        metrics.update(
            metal_bootstrap_metric_cis(
                mean_probabilities,
                target,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                seed=bootstrap_seed,
                n_bins=n_bins,
            )
        )
        if ensemble_calibrated_probabilities is not None:
            calibrated_metrics.update(
                {
                    key.replace("test_metal", "test_metal_temperature_scaled"): value
                    for key, value in metal_bootstrap_metric_cis(
                        ensemble_calibrated_probabilities,
                        target,
                        n_bootstrap=n_bootstrap,
                        confidence_level=confidence_level,
                        seed=bootstrap_seed,
                        n_bins=n_bins,
                    ).items()
                }
            )

    prediction_path = write_prediction_artifact(
        output_dir / "ensemble_predictions.pt",
        prediction_artifact_payload(
            task=task,
            metal_y=target,
            metal_probabilities=mean_probabilities,
            metal_calibrated_probabilities=ensemble_calibrated_probabilities,
            metadata={"ensemble_rule": "unweighted arithmetic mean of five fixed softmax probability vectors"},
        ),
    )
    return {
        "metrics": metrics,
        "calibrated_metrics": calibrated_metrics,
        "fitted_temperatures": {"metal_per_checkpoint": fitted_temperatures},
        "temperature_scaling": {
            "enabled": ensemble_calibrated_probabilities is not None,
            "rule": "fit one scalar temperature on each checkpoint validation fold, then average calibrated softmax probabilities",
            "selection_rule": "fixed before held-out test evaluation; no test metric used",
            "unavailable_reason": None
            if ensemble_calibrated_probabilities is not None
            else "one or more checkpoint prediction artifacts lacked calibrated probabilities",
        },
        "bootstrap_settings": bootstrap_settings,
        "calibration_settings": {
            "enabled": bool(enable_calibration),
            "n_equal_mass_bins": int(n_bins),
            "overall_ece_definition": "predicted-class confidence binned into equal-mass bins",
            "classwise_ece_definition": "one-vs-rest class probability binned into equal-mass bins",
        },
        "calibration_plot_paths": plot_paths,
        "reliability_diagram_path": plot_paths.get("reliability_diagram_path"),
        "confidence_histogram_path": plot_paths.get("confidence_histogram_path"),
        "prediction_artifact_path": str(prediction_path),
        "n_test_pockets": int(target.numel()),
    }
