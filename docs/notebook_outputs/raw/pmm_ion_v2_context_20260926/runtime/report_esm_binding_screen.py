"""Report the approved single-fold pair from verified artifacts, without training."""
from pathlib import Path
import argparse
import csv
import datetime as dt
import hashlib
import json
import math

import torch

C = Path(__file__).resolve().parent.parent
LABELS = ["Mn", "Cu", "Zn", "Class VIII"]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(readout):
    run = C / "runs" / f"only_esm__four_class__{readout}__fold0__seed42"
    selected = read(run / "selected_checkpoint.json")
    replay = read(run / "independent_validation_replay/replay_receipt.json")
    assert selected["fit_status"] == replay["fit_status"] == "completed"
    assert selected["completed_epochs"] == replay["completed_epochs"] == 50
    assert replay["independent_replay"] and replay["reconciliation_status"] == "match"
    checkpoint = run / "best_model_checkpoint.pt"
    assert sha(checkpoint) == selected["selected_checkpoint_sha256"] == replay["selected_checkpoint_sha256"]
    predictions = list(csv.DictReader((run / "val_predictions.csv").open()))
    independent = list(csv.DictReader((run / "independent_validation_replay/val_predictions.csv").open()))
    assert sha(run / "val_predictions.csv") == selected["validation_predictions"]["sha256"]
    assert sha(run / "independent_validation_replay/val_predictions.csv") == replay["validation_predictions"]["sha256"]
    assert [(r["source_uid"], r["y_common4"], r["pred_common4"]) for r in predictions] == [
        (r["source_uid"], r["y_common4"], r["pred_common4"]) for r in independent]
    assert len({r["source_uid"] for r in predictions}) == len(predictions) == 1492
    matrix = [[0] * 4 for _ in LABELS]
    for row in predictions:
        matrix[int(row["y_common4"])][int(row["pred_common4"])] += 1
    recalls = [matrix[i][i] / sum(matrix[i]) for i in range(4)]
    f1s = [2 * matrix[i][i] / (sum(matrix[i]) + sum(row[i] for row in matrix)) for i in range(4)]
    metrics = {"balanced_accuracy": sum(recalls) / 4, "macro_f1": sum(f1s) / 4,
               "accuracy": sum(matrix[i][i] for i in range(4)) / len(predictions),
               "per_class_recall": dict(zip(LABELS, recalls)), "confusion_matrix": matrix}
    for key, value in (("val_metal_balanced_acc", metrics["balanced_accuracy"]),
                       ("val_metal_macro_f1", metrics["macro_f1"])):
        assert math.isclose(value, replay["metrics"][key], abs_tol=1e-7)
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    biases = {key: value.detach().cpu().tolist()
              for key, value in checkpoint_data["model_state_dict"].items() if "binding_bias" in key}
    epochs = list(csv.DictReader((run / "epoch_metrics.csv").open()))
    assert len(epochs) == 50
    curves = [{key: float(row[key]) for key in ("epoch", "train_loss", "val_loss", "val_metal_balanced_acc")}
              for row in epochs]
    summary = {"run": str(run.relative_to(C)), "readout": readout,
               "identity": selected["campaign_run_identity"], "selected_epoch": selected["selected_epoch"],
               "checkpoint_sha256": sha(checkpoint), "metrics": metrics,
               "learned_biases": biases, "first_shell_support": read(run / "dataset_summary.json")["first_shell_support"],
               "learning_curves": curves, "independent_replay": "match", "n_validation": len(predictions),
               "validation_membership_sha256": hashlib.sha256("\n".join(sorted(
                   r["source_uid"] + ':' + r["y_common4"] for r in predictions)).encode()).hexdigest()}
    return summary


def main():
    global C
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, default=C)
    C = parser.parse_args().campaign_dir.resolve()
    baseline, binding = summarize("none"), summarize("first_shell_bias")
    for key in baseline["identity"]:
        if key not in {"readout", "resolved_config_sha256"}:
            assert baseline["identity"][key] == binding["identity"][key], key
    assert baseline["validation_membership_sha256"] == binding["validation_membership_sha256"]
    assert baseline["first_shell_support"] == binding["first_shell_support"]
    assert len(binding["learned_biases"]) == 2
    arm_rows = []
    for arm in (baseline, binding):
        rows = list(csv.DictReader((C / arm["run"] / "val_predictions.csv").open()))
        arm_rows.append({r["source_uid"]: r for r in rows})
    probability_columns = ["p_common4_Mn", "p_common4_Cu", "p_common4_Zn", "p_common4_Class_VIII"]
    probability_differences = []
    changed_predictions = 0
    for uid, left in arm_rows[0].items():
        right = arm_rows[1][uid]
        for key in ("physical_ion_id", "group_id", "example_id", "parent_pocket_id", "y_common4"):
            assert left[key] == right[key], (uid, key)
        changed_predictions += left["pred_common4"] != right["pred_common4"]
        probability_differences.extend(abs(float(left[k]) - float(right[k])) for k in probability_columns)
    paired_predictions = {"changed_class_predictions": changed_predictions,
                          "matched_validation_ions": len(arm_rows[0]),
                          "max_absolute_probability_difference": max(probability_differences),
                          "mean_absolute_probability_difference": sum(probability_differences) / len(probability_differences)}
    delta = {key: binding["metrics"][key] - baseline["metrics"][key]
             for key in ("balanced_accuracy", "macro_f1", "accuracy")}
    delta["per_class_recall"] = {key: binding["metrics"]["per_class_recall"][key]
                               - baseline["metrics"]["per_class_recall"][key] for key in LABELS}
    report = {"schema_version": 1, "status": "completed_exploratory_single_fold_pair",
              "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "held_out_access": False, "baseline": baseline, "binding_aware": binding,
              "binding_minus_baseline": delta, "class_order": LABELS,
              "paired_predictions": paired_predictions,
              "interpretation": "One validation fold and seed; screening trend only. No model promotion or superiority claim.",
              "future_confirmation": "Complete the remaining four matched folds before final selection/reporting; disclose fold-0 screening."}
    runtime = C / "runtime"
    (runtime / "esm_binding_screen.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Exploratory ESMC binding-awareness screen", "",
             "Frozen fold 0, seed 42; 50 epochs per arm; 1,492 matched validation ions. No held-out access.", "",
             "| Metric | Ordinary ESMC | Binding-aware ESMC | Difference (pp) |",
             "|---|---:|---:|---:|"]
    for name, key in (("Balanced accuracy", "balanced_accuracy"), ("Macro-F1", "macro_f1"), ("Accuracy", "accuracy")):
        lines.append(f"| {name} | {100*baseline['metrics'][key]:.3f}% | {100*binding['metrics'][key]:.3f}% | {100*delta[key]:+.3f} |")
    for key in LABELS:
        lines.append(f"| {key} recall | {100*baseline['metrics']['per_class_recall'][key]:.3f}% | {100*binding['metrics']['per_class_recall'][key]:.3f}% | {100*delta['per_class_recall'][key]:+.3f} |")
    lines += ["", f"Validation-selected epochs: ordinary {baseline['selected_epoch']}; binding-aware {binding['selected_epoch']}.", "",
              "Both checkpoints passed independent replay and were copied to the verified host backup.", "",
              f"Class predictions changed for {changed_predictions}/1,492 matched validation ions. "
              f"Maximum absolute probability difference: {max(probability_differences):.6g}. "
              "Equal selected metrics do not establish identical models or equivalence.", "",
              f"Learned pooling logit biases: `{json.dumps(binding['learned_biases'], sort_keys=True)}`.", "",
              f"First-shell coverage: `{json.dumps(binding['first_shell_support'], sort_keys=True)}`.", "",
              "The shell flag is a geometry-derived donor-distance proxy. Positive pooling biases "
              "show a learned relative weighting; they are not ligand annotations or evidence of improved accuracy.", "",
              "![Learning curves and confusion matrices](esm_binding_screen.png)", "",
              report["interpretation"], report["future_confirmation"], ""]
    (runtime / "esm_binding_screen.md").write_text("\n".join(lines))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for arm, label in ((baseline, "Ordinary ESMC"), (binding, "Binding-aware ESMC")):
        curves = arm["learning_curves"]
        x = [e["epoch"] for e in curves]
        axes[0, 0].plot(x, [e["val_metal_balanced_acc"] for e in curves], label=label)
        axes[0, 1].plot(x, [e["val_loss"] for e in curves], label=label)
    axes[0, 0].set(title="Validation balanced accuracy", xlabel="Epoch", ylabel="Balanced accuracy")
    axes[0, 1].set(title="Validation loss", xlabel="Epoch", ylabel="Loss")
    axes[0, 0].legend()
    axes[0, 1].legend()
    for ax, arm, label in zip(axes[1], (baseline, binding), ("Ordinary ESMC", "Binding-aware ESMC")):
        matrix = arm["metrics"]["confusion_matrix"]
        ax.imshow(matrix, cmap="Blues")
        ax.set(xticks=range(4), yticks=range(4), xticklabels=LABELS, yticklabels=LABELS,
               xlabel="Predicted", ylabel="Observed", title=f"{label}: selected checkpoint")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, str(matrix[i][j]), ha="center", va="center", color="white" if matrix[i][j] > max(map(max, matrix)) / 2 else "black")
    fig.suptitle("Single-fold ESMC binding-awareness screen (exploratory)")
    fig.savefig(runtime / "esm_binding_screen.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"status": report["status"], "baseline_BA": baseline["metrics"]["balanced_accuracy"],
                      "binding_BA": binding["metrics"]["balanced_accuracy"], "delta_BA_pp": 100*delta["balanced_accuracy"]}))


if __name__ == "__main__":
    main()
