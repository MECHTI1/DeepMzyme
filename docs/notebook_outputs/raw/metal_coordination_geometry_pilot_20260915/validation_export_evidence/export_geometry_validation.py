"""Export selected-checkpoint validation predictions for the fifteen geometry fits.

Operations helper only: invoke against the frozen geometry source checkout.
No training, test evaluation, checkpoint reselection, or model averaging occurs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gc
import json
import math
import os
from pathlib import Path
import signal
import sys
import time


LABELS = {0: "Mn", 1: "Cu", 2: "Zn", 3: "Class VIII"}
ALLOWED_CONFIG_DIFFERENCES = {"site_geometry_features", "metal_node_mode", "learning_rate", "seed", "model_seed", "training_sampler_seed", "run_name", "runs_dir"}
CONTRASTS = (("B", "A"), ("C", "B"), ("D", "B"), ("E", "D"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_forward_configs(reference, candidate):
    keys = (set(reference) | set(candidate)) - ALLOWED_CONFIG_DIFFERENCES
    mismatches = sorted(k for k in keys if reference.get(k) != candidate.get(k))
    require(not mismatches, f"Unmatched saved forward configuration fields: {mismatches}")
    require(candidate["task"] == "metal" and candidate["model_architecture"] == "only_gvp"
            and candidate["metal_label_scheme"] == "merge_fe_class_viii"
            and candidate["structural_readout_scope"] == "residue_only", "Wrong scientific task or architecture")
    require(not candidate["run_test_eval"] and candidate["test_structure_dir"] is None
            and candidate["test_summary_csv"] is None, "Held-out paths/evaluation are forbidden")


def verified_prediction_rows(logits, targets, examples, selected, run, labels):
    import torch
    from training.loop import classification_metrics_from_logits
    require(labels == LABELS, "The selected checkpoint does not have the native four-class label order")
    require(logits.shape == (len(examples), 4) and targets.shape == (len(examples),), "Prediction dimensions differ from validation membership")
    require(torch.isfinite(logits).all(), "Nonfinite validation logits")
    require(targets.tolist() == [int(e["y_metal"]) for e in examples], "Validation prediction order or true labels changed")
    metrics = classification_metrics_from_logits(logits, targets)
    require(math.isclose(metrics["balanced_accuracy"], selected["val_metal_balanced_acc"], abs_tol=1e-8, rel_tol=0),
            "Recomputed validation balanced accuracy does not match the selected checkpoint")
    require(metrics["confusion_matrix"] == selected["val_metal_confusion_matrix"], "Recomputed validation confusion matrix differs")
    for index, label in labels.items():
        require(math.isclose(metrics["per_class_recall"][index], selected["val_metal_per_class_recall"][label], abs_tol=1e-8, rel_tol=0),
                f"Recomputed {label} recall differs")
    probabilities = logits.softmax(-1)
    predictions = logits.argmax(-1).tolist()
    output = []
    for i, example in enumerate(examples):
        row = {k: run[k] for k in ("id", "arm", "geometry", "metal_node_mode", "lr", "seed", "selected_epoch")}
        row.update({k: example[k] for k in ("structure_id", "pocket_id", "group")})
        row.update(target_id=int(targets[i]), target_label=labels[int(targets[i])],
                   prediction_id=predictions[i], prediction_label=labels[predictions[i]],
                   correct=predictions[i] == int(targets[i]))
        row.update({f"probability_{label}": float(probabilities[i, index]) for index, label in labels.items()})
        output.append(row)
    return output, metrics


def paired_errors(records):
    results = []
    for candidate_arm, reference_arm in CONTRASTS:
        for seed in (42, 43):
            candidates = [r for r in records if r["arm"] == candidate_arm and r["seed"] == seed]
            references = [r for r in records if r["arm"] == reference_arm and r["seed"] == seed]
            for candidate in candidates:
                matched = [r for r in references if seed == 43 or r["lr"] == candidate["lr"]]
                require(len(matched) == 1, "Missing or ambiguous paired geometry run")
                reference = matched[0]
                a, b = candidate["predictions"], reference["predictions"]
                keys = ("structure_id", "pocket_id", "group", "target_id")
                require([[r[k] for k in keys] for r in a] == [[r[k] for k in keys] for r in b], "Paired prediction identities differ")
                counts = dict(both_correct=0, candidate_only_correct=0, reference_only_correct=0, both_wrong=0)
                for left, right in zip(a, b):
                    key = ("both_correct" if right["correct"] else "candidate_only_correct") if left["correct"] else (
                        "reference_only_correct" if right["correct"] else "both_wrong")
                    counts[key] += 1
                results.append(dict(contrast=f"{candidate_arm}-{reference_arm}", seed=seed,
                     candidate_run_id=candidate["id"], reference_run_id=reference["id"],
                     candidate_lr=candidate["lr"], reference_lr=reference["lr"],
                     matched_learning_rate=candidate["lr"] == reference["lr"],
                     interpretation="shared-LR feature contrast" if seed == 42 else
                     "validation-selected LR configurations; differing LRs do not isolate the feature change",
                     balanced_accuracy_delta=candidate["balanced_accuracy"] - reference["balanced_accuracy"],
                     validation_sites=len(a), counts=counts, descriptive_only=True))
    return results


def export(args):
    started = time.time()
    os.environ["DEEPGM_METAL_LABEL_SCHEME"] = "four_class"
    sys.path.insert(0, str(args.root / "src"))
    import run_metal_coordination_geometry_pilot as geometry
    base = geometry.base
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    attempts = base.read(output / "analysis_attempts.json", [])
    require(not any(a["status"] == "running" for a in attempts), "Reconcile the prior interrupted analysis before retrying")
    prior_used = sum(float(a.get("elapsed_seconds") or 0) for a in attempts)
    elapsed = base.allocation_elapsed(args.budget_root, args.allocation_started_epoch)
    available = min(570 * 60 - elapsed,
                    geometry.shared_budget_available(args.campaign_dir, args.budget_root, 0) - prior_used - (time.time() - started))
    deadline = min(args.deadline_epoch, time.time() + available)
    require(deadline - time.time() > 20, "No budget remains for validation prediction export")

    def timeout(signum, frame):
        raise TimeoutError("Validation export reached its allocation/category deadline")

    old_alarm = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, deadline - time.time())
    attempt = dict(attempt_id=len(attempts) + 1, status="running", started_epoch=started,
                   deadline_epoch=deadline, helper_sha256=base.digest(__file__))
    attempts.append(attempt)
    base.save(output / "analysis_attempts.json", attempts)
    try:
        manifest = geometry.verify_manifest(args.root, args.campaign_dir)
        require(Path(manifest["parent_campaign_dir"]) == args.budget_root, "Wrong original allocation budget root")
        geometry.ensure_parent_idle(args.budget_root)
        geometry.ensure_parent_idle(args.campaign_dir)
        base.verify_training_cache(args.campaign_dir)
        results = [r for r in geometry.rows(args.campaign_dir, manifest) if r["block"] != "S"]
        require(len(results) == 15 and all(sum(r["block"] == block for r in results) == 5 for block in ("G1", "G2", "GR")),
                "Complete all fifteen verified geometry runs before paired prediction export")
        selected_lrs = geometry.selected_lrs(results)
        require(all(r["lr"] == selected_lrs[r["arm"]] for r in results if r["seed"] == 43), "Unmatched selected-LR seed repeat")
        planned = {r["id"]: r for r in manifest["runs"]}
        payloads = {r["id"]: base.read(Path(r["run_dir"]) / "run_config.json") for r in results}
        reference_config = payloads[results[0]["id"]]["config"]
        for row in results:
            verify_forward_configs(reference_config, payloads[row["id"]]["config"])

        import torch
        from torch.utils.data import SequentialSampler
        from label_schemes import METAL_TARGET_LABELS
        from training.run import prepare_run, set_seed, normalization_stats_payload, to_jsonable
        from training.loop import evaluate_epoch_with_predictions
        require(torch.cuda.is_available(), "Use the already allocated CUDA runtime for this bounded export")
        require(METAL_TARGET_LABELS == LABELS, "Active native labels are not four-class")
        topology_metadata, exports, prediction_records = [], [], []
        for nodes, anchor_arm in (("none", "A"), ("per_metal", "D")):
            anchor = next(r for r in results if r["arm"] == anchor_arm and r["block"] == "G1")
            anchor_payload = payloads[anchor["id"]]
            config = base.parse_config(planned[anchor["id"]]["command"])
            parsed = to_jsonable(asdict(config))
            require(all(anchor_payload["config"].get(k) == v for k, v in parsed.items() if k not in ("run_name", "runs_dir")),
                    "Canonical preparation command differs from the saved anchor configuration")
            config = replace(config, runs_dir=output / "preparation", run_name=f"export_{anchor_arm}_attempt{attempt['attempt_id']}")
            set_seed(config.seed, deterministic=config.deterministic)
            prepared = prepare_run(config)
            require(base.cohort_identity(prepared.dataset_summary) == base.cohort_identity(anchor_payload["dataset_summary"]), "Prepared validation cohort changed")
            norm_hash = base.fingerprint(to_jsonable(normalization_stats_payload(prepared.normalization_stats)))
            require(norm_hash == base.fingerprint(anchor_payload["normalization_stats"]), "Fresh preparation changed training-fitted normalization")
            loader = prepared.val_loader
            require(loader is not None and isinstance(loader.sampler, SequentialSampler) and not loader.drop_last,
                    "Validation loader must preserve every example in its saved order")
            examples = prepared.dataset_summary["retained_split_identity"]["validation"]["examples"]
            require(len(loader.dataset) == len(examples), "Validation membership length changed")
            model = prepared.model
            require(model.site_feature_encoder[0].in_features == 12 and model.node_type_embedding is not None, "Explicit geometry model shape changed")
            topology_metadata.append(dict(metal_node_mode=nodes, preparation_anchor=anchor["id"],
                normalization_stats_sha256=norm_hash, validation_sites=len(examples),
                parameter_count=sum(p.numel() for p in model.parameters())))
            for run in (r for r in results if r["metal_node_mode"] == nodes):
                payload = payloads[run["id"]]
                require(base.fingerprint(payload["normalization_stats"]) == norm_hash, "Normalization differs within a topology")
                require(payload["dataset_summary"]["retained_split_identity"]["validation"]["examples"] == examples,
                        "Saved validation order, identity, or target changed")
                checkpoint_path = Path(run["run_dir"]) / "best_model_checkpoint.pt"
                checkpoint_sha = base.digest(checkpoint_path)
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                require(checkpoint["epoch"] == run["selected_epoch"] and checkpoint["selection_metric"] == geometry.METRIC,
                        "The checkpoint is not the recorded native-selected epoch")
                require(base.fingerprint(to_jsonable(checkpoint["normalization_stats"])) == norm_hash, "Checkpoint normalization differs")
                checkpoint_config = to_jsonable(checkpoint["config"])
                require(checkpoint_config == payload["config"], "Checkpoint and run configuration differ")
                require(base.cohort_identity(checkpoint["dataset_summary"]) == base.cohort_identity(prepared.dataset_summary), "Checkpoint cohort differs")
                labels = {int(k): str(v) for k, v in checkpoint["metal_labels"].items()}
                selected = next(r for r in payload["history"] if r["epoch"] == run["selected_epoch"])
                model.site_geometry_features = run["geometry"]
                model.use_site_angle_features = run["geometry"] == "counts_angles"
                model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                predictions = evaluate_epoch_with_predictions(model, loader, device=config.device)
                exported, metrics = verified_prediction_rows(predictions["metal_logits"], predictions["metal_y"], examples, selected, run, labels)
                destination = output / f"{run['id']}_validation_predictions.csv"
                base.csv_save(destination, exported, list(exported[0]))
                metadata = dict(run_id=run["id"], arm=run["arm"], lr=run["lr"], seed=run["seed"],
                    selected_epoch=run["selected_epoch"], checkpoint_path=str(checkpoint_path), checkpoint_sha256=checkpoint_sha,
                    saved_config_sha256=base.fingerprint(payload["config"]), normalization_stats_sha256=norm_hash,
                    prediction_csv=str(destination), prediction_csv_sha256=base.digest(destination),
                    balanced_accuracy=metrics["balanced_accuracy"], confusion_matrix=metrics["confusion_matrix"],
                    per_class_recall={LABELS[k]: v for k, v in enumerate(metrics["per_class_recall"])},
                    selected_checkpoint_metrics_verified=True)
                exports.append(metadata)
                prediction_records.append({**run, "predictions": exported})
                print(json.dumps({"exported": run["id"], "balanced_accuracy": metrics["balanced_accuracy"], "validation_sites": len(exported)}), flush=True)
                del checkpoint, predictions
            del prepared, model, loader
            gc.collect()
            torch.cuda.empty_cache()
        pairs = paired_errors(prediction_records)
        base.save(output / "paired_validation_errors.json", pairs)
        report = dict(status="complete", profile=geometry.PROFILE, exported_runs=15, labels=LABELS,
            helper_sha256=attempt["helper_sha256"], manifest_sha256=base.digest(args.campaign_dir / "campaign_manifest.json"),
            source_files=manifest["source_files"], cohort_sha256=manifest["cohort_sha256"], topology_preparations=topology_metadata,
            runtime=dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0)),
            held_out_evaluation=False, training_performed=False, checkpoint_reselection=False,
            paired_comparisons_descriptive=True, confidence_intervals_computed=False, runs=exports)
        base.save(output / "validation_prediction_export.json", report)
        attempt["status"] = "completed"
        return report
    except BaseException as exc:
        attempt.update(status="failed", error=str(exc))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_alarm)
        attempt.update(ended_epoch=time.time(), elapsed_seconds=time.time() - started)
        base.save(output / "analysis_attempts.json", attempts)
        base.save(args.budget_root / "coordination_geometry_analysis_budget_usage.json", dict(
            analysis_output_dir=str(output), normal_elapsed_seconds=sum(a["elapsed_seconds"] for a in attempts),
            helper_sha256=base.digest(__file__), training_performed=False))
        base.allocation_elapsed(args.budget_root, args.allocation_started_epoch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "campaign-dir", "output-dir", "budget-root"):
        parser.add_argument(f"--{name}", type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--allocation-started-epoch", type=float, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args()
    require(math.isfinite(args.deadline_epoch), "Supply a finite absolute deadline")
    report = export(args)
    print(json.dumps({"status": report["status"], "exported_runs": report["exported_runs"], "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
