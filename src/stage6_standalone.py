from __future__ import annotations

import csv
import concurrent.futures
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _slug(value: object) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("+", " ").replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "na"


def _numeric(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _as_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", "none", ""}:
        return False
    return None


def _write_rows_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    columns.append(key)
                    seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in columns})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_safe(value), sort_keys=True)
    return value


def _parse_ints(text: str, name: str) -> list[int]:
    values = []
    for token in str(text or "").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(int(token))
        except Exception as exc:
            raise ValueError(f"{name} must be comma-separated integers.") from exc
    if not values:
        raise ValueError(f"{name} must contain at least one integer.")
    return values


def _resolve_top_k(value: object, completed_count: int) -> int:
    text = str(value or "auto").strip().lower()
    if text == "auto":
        if completed_count < 50:
            return min(5, completed_count)
        if completed_count < 150:
            return min(10, completed_count)
        return min(20, completed_count)
    parsed = int(text)
    if parsed < 1:
        raise ValueError("Top-K must be 'auto' or a positive integer.")
    return min(parsed, completed_count)


def _stage6_units(mode: str, seeds: list[int], n_folds: int, split_seed: int) -> list[dict[str, Any]]:
    mode = str(mode or "").strip().lower()
    if mode not in {"group_kfold", "group_kfold_seed_repeat"}:
        raise ValueError("Standalone Stage 6 supports group_kfold or group_kfold_seed_repeat.")
    if n_folds < 2:
        raise ValueError("Stage 6 grouped-fold confirmation requires at least 2 folds.")
    active_seeds = seeds if mode == "group_kfold_seed_repeat" else seeds[:1]
    return [
        {
            "validation_unit": f"fold_{fold}_seed_{seed}" if mode == "group_kfold_seed_repeat" else f"fold_{fold}",
            "fold_unit": f"fold_{fold}",
            "model_seed": int(seed),
            "split_seed": int(split_seed),
            "n_folds": int(n_folds),
            "fold_index": int(fold),
        }
        for fold in range(n_folds)
        for seed in active_seeds
    ]


def _selected_history_record(payloads: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    histories = []
    selected_epoch = None
    for payload in payloads:
        history = payload.get("history")
        if isinstance(history, list):
            histories.extend(row for row in history if isinstance(row, dict))
        if selected_epoch is None:
            selected_epoch = payload.get("selected_checkpoint_epoch")
    if histories and selected_epoch is not None:
        try:
            selected_epoch_int = int(selected_epoch)
        except Exception:
            selected_epoch_int = None
        if selected_epoch_int is not None:
            for row in histories:
                try:
                    if int(row.get("epoch", -1)) == selected_epoch_int:
                        return dict(row)
                except Exception:
                    continue
    metric_rows = [row for row in histories if _numeric(row.get(metric)) is not None]
    if metric_rows:
        return min(metric_rows, key=lambda row: float(row[metric])) if metric.endswith("_loss") else max(metric_rows, key=lambda row: float(row[metric]))
    return {}


@dataclass
class Candidate:
    candidate_id: str
    run_dir: Path
    run_name: str
    config: dict[str, Any]
    selection_metric: str
    selected_metric_value: float | None
    val_metal_balanced_acc: float | None
    val_metal_min_recall: float | None
    val_loss: float | None
    selected_checkpoint: str | None
    skip_reason: str | None = None


def _candidate_from_run_dir(run_dir: Path, selection_metric: str) -> Candidate | None:
    run_config_payload = _read_json(run_dir / "run_config.json")
    metadata = _read_json(run_dir / "run_metadata.json")
    active = _read_json(run_dir / "active_run_config.json")
    if not run_config_payload and not metadata and not active:
        return None
    config: dict[str, Any] = {}
    if isinstance(run_config_payload.get("config"), dict):
        config.update(run_config_payload["config"])
    elif run_config_payload:
        config.update(run_config_payload)
    if isinstance(metadata.get("config"), dict):
        config.update(metadata["config"])
    if isinstance(active.get("run_config"), dict):
        config.update(active["run_config"])
    extra = active.get("extra") if isinstance(active.get("extra"), dict) else {}
    sampled = extra.get("sampled_params") if isinstance(extra.get("sampled_params"), dict) else {}
    config.update(sampled)

    metric = str(selection_metric or metadata.get("selection_metric") or run_config_payload.get("selection_metric") or config.get("selection_metric") or "val_metal_balanced_acc")
    selected = _numeric(metadata.get("selected_metric_value") or run_config_payload.get("selected_metric_value"))
    selected_record = _selected_history_record([run_config_payload, metadata], metric)
    if selected is None:
        selected = _numeric(selected_record.get(metric))
    val_balanced = _numeric(selected_record.get("val_metal_balanced_acc"))
    if val_balanced is None:
        val_balanced = _numeric(metadata.get("val_metal_balanced_acc") or run_config_payload.get("val_metal_balanced_acc"))
    if val_balanced is None and metric == "val_metal_balanced_acc":
        val_balanced = selected
    val_min_recall = _numeric(selected_record.get("val_metal_min_recall"))
    val_loss = _numeric(selected_record.get("val_loss"))
    trial = extra.get("trial_number") or metadata.get("trial_number") or config.get("trial_number")
    try:
        trial_label = f"trial{int(trial):04d}" if trial is not None else None
    except Exception:
        trial_label = None
    candidate_id = trial_label or _slug(config.get("run_name") or run_dir.name)
    return Candidate(
        candidate_id=candidate_id,
        run_dir=run_dir,
        run_name=str(config.get("run_name") or run_dir.name),
        config=config,
        selection_metric=metric,
        selected_metric_value=selected,
        val_metal_balanced_acc=val_balanced,
        val_metal_min_recall=val_min_recall,
        val_loss=val_loss,
        selected_checkpoint=str(metadata.get("selected_checkpoint") or run_config_payload.get("selected_checkpoint") or "") or None,
    )


def _candidate_skip_reason(candidate: Candidate) -> str | None:
    label_text = " ".join([candidate.candidate_id, candidate.run_name, candidate.run_dir.name]).lower()
    full_text = " ".join([label_text, str(candidate.run_dir)]).lower()
    config = candidate.config
    if (candidate.run_dir / "test_report.json").exists():
        return "held-out test artifact: test_report.json exists"
    if any(token in full_text for token in ("final_test", "final-test", "held_out", "held-out", "test_eval", "test-eval")):
        return "held-out/final-test run name or path"
    if any(token in label_text for token in ("stage6", "group_kfold", "seed_repeat")):
        return "Stage 6 or seed-repeat rerun, not an original HPO candidate"
    if _as_bool(config.get("run_test_eval")) or _as_bool(config.get("run_held_out_test_eval")):
        return "held-out test evaluation was enabled"
    if config.get("n_folds") not in (None, "", "None") and config.get("fold_index") not in (None, "", "None"):
        return "fold-level validation rerun, not an original HPO candidate"
    if any(token in label_text for token in ("debug", "smoke")):
        return "debug/smoke run"
    epochs = _numeric(config.get("epochs") or config.get("trial_epochs"))
    if epochs is not None and epochs <= 3:
        return "debug-length trial (epochs <= 3)"
    if candidate.selected_metric_value is None and candidate.val_metal_balanced_acc is None:
        return "no validation metric found"
    return None


def _discover_candidates(existing_dir: Path, selection_metric: str) -> tuple[list[Candidate], list[dict[str, Any]]]:
    run_dirs = sorted({path.parent for marker in ("run_config.json", "run_metadata.json") for path in existing_dir.rglob(marker)})
    candidates = []
    report_rows = []
    seen_dirs = set()
    for run_dir in run_dirs:
        if run_dir in seen_dirs:
            continue
        seen_dirs.add(run_dir)
        candidate = _candidate_from_run_dir(run_dir, selection_metric)
        if candidate is None:
            continue
        reason = _candidate_skip_reason(candidate)
        candidate.skip_reason = reason
        report_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "source_run_dir": str(candidate.run_dir),
                "run_name": candidate.run_name,
                "status": "skipped" if reason else "compatible",
                "reason": reason or "",
                "selection_metric": candidate.selection_metric,
                "selected_metric_value": candidate.selected_metric_value,
                "val_metal_balanced_acc": candidate.val_metal_balanced_acc,
                "val_metal_min_recall": candidate.val_metal_min_recall,
                "val_loss": candidate.val_loss,
                "model_architecture": candidate.config.get("model_architecture"),
                "fusion_mode": candidate.config.get("fusion_mode"),
                "model_preset": candidate.config.get("model_preset"),
            }
        )
        if reason is None:
            candidates.append(candidate)
    return candidates, report_rows


def _ranking_value(candidate: Candidate, selection_metric: str) -> float | None:
    if selection_metric == "val_metal_balanced_acc":
        return candidate.val_metal_balanced_acc if candidate.val_metal_balanced_acc is not None else candidate.selected_metric_value
    return candidate.selected_metric_value


def _rank_candidates(candidates: list[Candidate], selection_metric: str) -> list[Candidate]:
    reverse = not str(selection_metric).endswith("_loss")

    def key(candidate: Candidate) -> tuple[float, float, float]:
        primary = _ranking_value(candidate, selection_metric)
        if primary is None:
            primary = float("-inf") if reverse else float("inf")
        min_recall = candidate.val_metal_min_recall if candidate.val_metal_min_recall is not None else float("-inf")
        loss = candidate.val_loss if candidate.val_loss is not None else float("inf")
        return (float(primary), float(min_recall), -float(loss))

    return sorted(candidates, key=key, reverse=reverse)


def _add_flag(cmd: list[str], flag: str, value: object | None = None) -> None:
    if value is None or value == "":
        return
    cmd.append(flag)
    if value is not True:
        cmd.append(str(value))


def _config_get(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = config.get(key)
        if value is not None and value != "":
            return value
    return default


def _build_train_command(config: dict[str, Any], repo_dir: Path) -> list[str]:
    cmd = [sys.executable, str(repo_dir / "src" / "train.py")]
    scalar_flags = [
        ("task", "--task"),
        ("metal_label_scheme", "--metal-label-scheme"),
        ("structure_dir", "--structure-dir"),
        ("summary_csv", "--summary-csv"),
        ("esm_embeddings_dir", "--esm-embeddings-dir"),
        ("ring_features_dir", "--ring-features-dir"),
        ("external_features_root_dir", "--external-features-root-dir"),
        ("external_feature_source", "--external-feature-source"),
        ("runs_dir", "--runs-dir"),
        ("run_name", "--run-name"),
        ("device", "--device"),
        ("epochs", "--epochs"),
        ("batch_size", "--batch-size"),
        ("esm_dim", "--esm-dim"),
        ("model_architecture", "--model-architecture"),
        ("edge_radius", "--edge-radius"),
        ("learning_rate", "--learning-rate"),
        ("grad_clip_norm", "--grad-clip-norm"),
        ("grad_accum_steps", "--grad-accum-steps"),
        ("num_workers", "--num-workers"),
        ("weight_decay", "--weight-decay"),
        ("seed", "--seed"),
        ("split_seed", "--split-seed"),
        ("hidden_s", "--hidden-s"),
        ("hidden_v", "--hidden-v"),
        ("edge_hidden", "--edge-hidden"),
        ("gvp_layers", "--gvp-layers"),
        ("esm_fusion_dim", "--esm-fusion-dim"),
        ("head_mlp_layers", "--head-mlp-layers"),
        ("head_mlp_dropout", "--head-mlp-dropout"),
        ("esm_graph_encoder_dropout", "--esm-graph-encoder-dropout"),
        ("node_rbf_sigma", "--node-rbf-sigma"),
        ("edge_rbf_sigma", "--edge-rbf-sigma"),
        ("classifier_pool_distance_cutoff", "--classifier-pool-distance-cutoff"),
        ("metal_node_mode", "--metal-node-mode"),
        ("structural_readout_scope", "--structural-readout-scope"),
        ("position_noise_std", "--position-noise-std"),
        ("second_shell_dropout", "--second-shell-dropout"),
        ("outer_residue_dropout", "--outer-residue-dropout"),
        ("node_feature_set", "--node-feature-set"),
        ("fusion_mode", "--fusion-mode"),
        ("cross_attention_layers", "--cross-attention-layers"),
        ("cross_attention_heads", "--cross-attention-heads"),
        ("cross_attention_dropout", "--cross-attention-dropout"),
        ("cross_attention_neighborhood", "--cross-attention-neighborhood"),
        ("early_esm_dim", "--early-esm-dim"),
        ("early_esm_dropout", "--early-esm-dropout"),
        ("early_esm_scope", "--early-esm-scope"),
        ("lr_schedule", "--lr-schedule"),
        ("lr_step_size", "--lr-step-size"),
        ("lr_decay_gamma", "--lr-decay-gamma"),
        ("val_fraction", "--val-fraction"),
        ("n_folds", "--n-folds"),
        ("fold_index", "--fold-index"),
        ("mn_loss_multiplier", "--mn-loss-multiplier"),
        ("cu_loss_multiplier", "--cu-loss-multiplier"),
        ("zn_loss_multiplier", "--zn-loss-multiplier"),
        ("fe_loss_multiplier", "--fe-loss-multiplier"),
        ("co_loss_multiplier", "--co-loss-multiplier"),
        ("ni_loss_multiplier", "--ni-loss-multiplier"),
        ("class_viii_loss_multiplier", "--class-viii-loss-multiplier"),
        ("joint_loss_weighting", "--joint-loss-weighting"),
        ("metal_loss_weight", "--metal-loss-weight"),
        ("ec_loss_weight", "--ec-loss-weight"),
        ("metal_class_weight_mode", "--metal-class-weight-mode"),
        ("metal_loss_function", "--metal-loss-function"),
        ("metal_focal_gamma", "--metal-focal-gamma"),
        ("metal_label_smoothing", "--metal-label-smoothing"),
        ("metal_collapsed_loss_weight", "--metal-collapsed-loss-weight"),
        ("unsupported_metal_policy", "--unsupported-metal-policy"),
        ("invalid_structure_policy", "--invalid-structure-policy"),
        ("ec_label_depth", "--ec-label-depth"),
        ("ec_group_weighting", "--ec-group-weighting"),
        ("ec_contrastive_weight", "--ec-contrastive-weight"),
        ("ec_contrastive_temperature", "--ec-contrastive-temperature"),
        ("selection_metric", "--selection-metric"),
        ("split_by", "--split-by"),
    ]
    for key, flag in scalar_flags:
        _add_flag(cmd, flag, config.get(key))
    omit = config.get("omit_node_features")
    if isinstance(omit, (list, tuple)):
        omit = ",".join(str(item) for item in omit)
    _add_flag(cmd, "--omit-node-features", omit)
    bool_flags = [
        ("deterministic", "--deterministic"),
        ("use_amp", "--amp"),
        ("pin_memory", "--pin-memory"),
        ("node_rbf_use_raw_distances", "--node-rbf-use-raw-distances"),
        ("normalize_message_aggregation", "--gvp-normalize-message-aggregation"),
        ("cross_attention_bidirectional", "--cross-attention-bidirectional"),
        ("use_early_esm", "--use-early-esm"),
        ("early_esm_raw", "--early-esm-raw"),
        ("use_ring_edges", "--use-ring-edges"),
        ("require_ring_edges", "--require-ring-edges"),
        ("allow_missing_esm_embeddings", "--allow-missing-esm-embeddings"),
        ("allow_missing_external_features", "--allow-missing-external-features"),
        ("prepare_missing_ring_edges", "--prepare-missing-ring-edges"),
        ("save_epoch_checkpoints", "--save-epoch-checkpoints"),
        ("log_per_class_metrics", "--log-per-class-metrics"),
        ("balance_metal_site_symbols", "--balance-metal-site-symbols"),
        ("require_all_task_classes", "--require-all-task-classes"),
    ]
    for key, flag in bool_flags:
        if _as_bool(config.get(key)):
            cmd.append(flag)
    if _as_bool(config.get("use_esm_branch")) is False:
        cmd.append("--disable-esm-branch")
    if _as_bool(config.get("prepare_missing_esm_embeddings")) is False:
        cmd.append("--no-prepare-missing-esm-embeddings")
    if _as_bool(config.get("prepare_missing_ring_edges")) is False:
        cmd.append("--no-prepare-missing-ring-edges")
    if "--run-test-eval" in cmd:
        raise RuntimeError("Internal safety error: standalone Stage 6 command contains --run-test-eval.")
    return cmd


def _command_text(cmd: list[str], env: dict[str, str] | None = None) -> str:
    import shlex

    prefix = []
    for key, value in sorted((env or {}).items()):
        if value:
            prefix.append(f"{key}={shlex.quote(str(value))}")
    return " ".join(prefix + [shlex.join([str(part) for part in cmd])])


def _stream_command(
    cmd: list[str],
    *,
    cwd: Path,
    stdout_log: Path,
    stderr_log: Path,
    print_to_console: bool = True,
) -> tuple[int, str]:
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    stderr_log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    tail: list[str] = []
    with stdout_log.open("w", encoding="utf-8") as out_handle, stderr_log.open("w", encoding="utf-8") as err_handle:
        process = subprocess.Popen(
            [str(part) for part in cmd],
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            if print_to_console:
                print(line, end="", flush=True)
            out_handle.write(line)
            err_handle.write(line)
            out_handle.flush()
            err_handle.flush()
            tail.append(line.rstrip())
            del tail[:-40]
        return process.wait(), "\n".join(tail)


def _write_active_snapshot(run_dir: Path, config: dict[str, Any], extra: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_by": "stage6_standalone.py",
        "source_mode": "standalone existing-HPO Stage 6",
        "result_stage": "group-kfold validation",
        "run_config": _json_safe({k: v for k, v in config.items() if k not in {"command"}}),
        "command": _command_text(config["command"]),
        "extra": _json_safe(extra),
    }
    (run_dir / "active_run_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "active_run_config.md").write_text(
        "\n".join(
            [
                "# Active Run Configuration",
                "",
                f"- source_mode: {payload['source_mode']}",
                f"- result_stage: {payload['result_stage']}",
                f"- command: `{payload['command']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _make_stage6_run_name(candidate: Candidate, unit: dict[str, Any], config: dict[str, Any]) -> str:
    parts = [
        "stage6",
        _slug(candidate.candidate_id),
        _slug(unit["validation_unit"]),
        _slug(config.get("task", "task")),
        _slug(config.get("model_architecture", "model")),
        "lr" + _slug(config.get("learning_rate", "na")),
        "wd" + _slug(config.get("weight_decay", "na")),
        "seed" + str(unit["model_seed"]),
    ]
    name = "_".join(parts)
    if len(name) > 150:
        import hashlib

        digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
        name = name[:141].rstrip("_") + "_" + digest
    return name


def _build_rerun_configs(
    selected: list[Candidate],
    units: list[dict[str, Any]],
    *,
    output_runs_dir: Path,
    repo_dir: Path,
    epochs: int | None,
    device: str,
    selection_metric: str,
) -> list[dict[str, Any]]:
    reruns = []
    for rank, candidate in enumerate(selected, start=1):
        for unit in units:
            cfg = dict(candidate.config)
            cfg["seed"] = int(unit["model_seed"])
            cfg["split_seed"] = int(unit["split_seed"])
            cfg["n_folds"] = int(unit["n_folds"])
            cfg["fold_index"] = int(unit["fold_index"])
            cfg["split_by"] = "pdbid"
            cfg["val_fraction"] = 0.0
            cfg["selection_metric"] = selection_metric
            if epochs is not None:
                cfg["epochs"] = int(epochs)
            if device:
                cfg["device"] = device
            cfg["runs_dir"] = str(output_runs_dir)
            cfg["run_name"] = _make_stage6_run_name(candidate, unit, cfg)
            run_dir = output_runs_dir / cfg["run_name"]
            cfg["command"] = _build_train_command(cfg, repo_dir)
            cfg["shell_command"] = _command_text(cfg["command"])
            cfg["run_dir"] = str(run_dir)
            cfg["stdout_log_path"] = str(output_runs_dir / "_execution_logs" / f"{cfg['run_name']}.stdout.log")
            cfg["stderr_log_path"] = str(output_runs_dir / "_execution_logs" / f"{cfg['run_name']}.stderr.log")
            cfg["candidate_id"] = f"imported_top{rank}_{_slug(candidate.candidate_id)}"
            cfg["imported_candidate_id"] = candidate.candidate_id
            cfg["validation_unit"] = str(unit["validation_unit"])
            cfg["fold_unit"] = str(unit["fold_unit"])
            cfg["top_rank"] = rank
            cfg["source_run_dir"] = str(candidate.run_dir)
            cfg["selected_checkpoint"] = candidate.selected_checkpoint
            cfg["original_validation_metric"] = _ranking_value(candidate, selection_metric)
            cfg["original_val_metal_balanced_acc"] = candidate.val_metal_balanced_acc
            cfg["original_val_metal_min_recall"] = candidate.val_metal_min_recall
            reruns.append(cfg)
    return reruns


def _stage6_candidate_batches(configs: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    current_key: tuple[int, str] | None = None
    for config in configs:
        key = (int(config.get("top_rank", 0)), str(config.get("candidate_id") or ""))
        if current_key != key:
            batches.append([])
            current_key = key
        batches[-1].append(config)
    return batches


def _path_contains(parent: Path, child: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _stage6_manifest_payload(
    *,
    existing_dir: Path,
    runs_dir: Path,
    optuna_dir: Path,
    study_name: str,
    top_k_requested: str | int,
    top_k_resolved: int,
    reevaluation_mode: str,
    repeat_seeds: list[int],
    n_folds: int,
    split_seed: int,
    epochs: int | None,
    device: str,
    selection_metric: str,
    raw_improvement_threshold: float,
    selected: list[Candidate],
) -> dict[str, Any]:
    return {
        "created_by": "stage6_standalone.py",
        "schema_version": 1,
        "protocol_stage": "Stage 6",
        "existing_runs_dir": str(existing_dir.resolve()),
        "output_runs_dir": str(runs_dir.resolve()),
        "optuna_dir": str(optuna_dir.resolve()),
        "study_name": str(study_name),
        "parameters": {
            "top_k_requested": str(top_k_requested),
            "top_k_resolved": int(top_k_resolved),
            "reevaluation_mode": str(reevaluation_mode),
            "repeat_seeds": [int(seed) for seed in repeat_seeds],
            "n_folds": int(n_folds),
            "split_seed": int(split_seed),
            "epochs": int(epochs) if epochs is not None else None,
            "device": str(device),
            "selection_metric": str(selection_metric),
            "raw_improvement_threshold": float(raw_improvement_threshold),
        },
        "selected_candidates": [
            {
                "rank": index,
                "candidate_id": candidate.candidate_id,
                "run_dir": str(candidate.run_dir),
                "selected_checkpoint": candidate.selected_checkpoint,
                "selection_metric_value": candidate.selected_metric_value,
                "val_metal_balanced_acc": candidate.val_metal_balanced_acc,
                "val_metal_min_recall": candidate.val_metal_min_recall,
            }
            for index, candidate in enumerate(selected, start=1)
        ],
        "compatibility_rule": (
            "Standalone Stage 6 output may be continued only when this manifest "
            "matches the requested source, output, parameters, and selected candidates."
        ),
    }


def _stage6_manifest_mismatches(existing: dict[str, Any], requested: dict[str, Any]) -> list[dict[str, Any]]:
    checks = [
        ("existing_runs_dir", existing.get("existing_runs_dir"), requested.get("existing_runs_dir")),
        ("study_name", existing.get("study_name"), requested.get("study_name")),
        ("parameters", existing.get("parameters"), requested.get("parameters")),
        ("selected_candidates", existing.get("selected_candidates"), requested.get("selected_candidates")),
    ]
    mismatches = []
    if int(existing.get("schema_version") or 0) != int(requested.get("schema_version") or 0):
        mismatches.append(
            {
                "field": "schema_version",
                "existing": existing.get("schema_version"),
                "requested": requested.get("schema_version"),
            }
        )
    for field, old_value, new_value in checks:
        if old_value != new_value:
            mismatches.append({"field": field, "existing": old_value, "requested": new_value})
    return mismatches


def _format_stage6_manifest_mismatches(mismatches: list[dict[str, Any]]) -> str:
    lines = []
    for mismatch in mismatches[:20]:
        lines.append(
            f"- {mismatch['field']}: existing={mismatch.get('existing')!r}; requested={mismatch.get('requested')!r}"
        )
    if len(mismatches) > 20:
        lines.append(f"- ... {len(mismatches) - 20} additional mismatch(es)")
    return "\n".join(lines)


def _planned_stage6_counts(units: list[dict[str, Any]]) -> dict[str, Any]:
    planned_units = {str(unit.get("validation_unit")) for unit in units if unit.get("validation_unit") not in (None, "")}
    planned_folds = {str(unit.get("fold_unit")) for unit in units if unit.get("fold_unit") not in (None, "")}
    planned_seeds = {str(unit.get("model_seed")) for unit in units if unit.get("model_seed") not in (None, "")}
    planned_fold_seed_units = {
        f"{unit.get('fold_unit')}|seed_{unit.get('model_seed')}"
        for unit in units
        if unit.get("fold_unit") not in (None, "") and unit.get("model_seed") not in (None, "")
    }
    return {
        "n_planned_units": len(planned_units) or len(units),
        "n_planned_fold_seed_units": len(planned_fold_seed_units) or len(units),
        "n_planned_folds": len(planned_folds),
        "n_planned_seeds": len(planned_seeds),
        "planned_model_seeds": ",".join(sorted(planned_seeds)),
    }


def _stage6_completion_status(
    summary_rows: list[dict[str, Any]],
    *,
    expected_candidate_count: int,
    units: list[dict[str, Any]],
) -> dict[str, Any]:
    planned = _planned_stage6_counts(units)
    annotated_rows = []
    for row in summary_rows:
        row = dict(row)
        row.update({key: row.get(key, value) for key, value in planned.items()})
        completed_units = int(_numeric(row.get("n_units_completed")) or 0)
        planned_units = int(_numeric(row.get("n_planned_units")) or 0)
        completed_fold_seed_units = int(_numeric(row.get("n_fold_seed_units_completed")) or completed_units)
        planned_fold_seed_units = int(_numeric(row.get("n_planned_fold_seed_units")) or planned_units)
        complete = (
            planned_units > 0
            and completed_units >= planned_units
            and planned_fold_seed_units > 0
            and completed_fold_seed_units >= planned_fold_seed_units
        )
        row["stage6_candidate_complete"] = complete
        row["stage6_completion_status"] = "complete" if complete else "partial"
        annotated_rows.append(row)
    missing_candidate_count = max(0, int(expected_candidate_count) - len(annotated_rows))
    complete = bool(annotated_rows) and missing_candidate_count == 0 and all(row.get("stage6_candidate_complete") for row in annotated_rows)
    return {
        "stage6_complete": complete,
        "expected_candidate_count": int(expected_candidate_count),
        "observed_candidate_count": len(annotated_rows),
        "missing_candidate_count": missing_candidate_count,
        "planned_counts": planned,
        "rows": annotated_rows,
    }


def _write_stage6_partial_outputs(
    *,
    optuna_dir: Path,
    records: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    pairwise_rows: list[dict[str, Any]],
    completion: dict[str, Any],
    manifest: dict[str, Any],
    canonical_outputs_written: bool,
) -> dict[str, Path]:
    partial_dir = optuna_dir / "stage6_partial"
    partial_dir.mkdir(parents=True, exist_ok=True)
    results_csv = partial_dir / "stage6_partial_results.csv"
    ranked_csv = partial_dir / "stage6_partial_ranked_candidates.csv"
    pairwise_csv = partial_dir / "stage6_partial_pairwise_bootstrap.csv"
    manifest_json = partial_dir / "stage6_partial_manifest.json"
    report_md = partial_dir / "stage6_partial_report.md"

    _write_rows_csv(results_csv, records)
    _write_rows_csv(ranked_csv, summary_rows)
    _write_rows_csv(pairwise_csv, pairwise_rows)
    payload = {
        "created_by": "stage6_standalone.py",
        "protocol_stage": "Stage 6 partial progress report",
        "selection_basis": "partial validation/CV metrics only; not final promotion evidence",
        "canonical_outputs_written": bool(canonical_outputs_written),
        "safe_to_resume_stage6": True,
        "resume_rule": "Rerun Stage 6 with identical manifest/settings and SKIP_EXISTING_RUNS=True; this stage6_partial directory is ignored by resume planning.",
        "completion": completion,
        "stage6_manifest": manifest,
        "files": {
            "partial_results_csv": str(results_csv),
            "partial_ranked_candidates_csv": str(ranked_csv),
            "partial_pairwise_bootstrap_csv": str(pairwise_csv),
            "partial_manifest_json": str(manifest_json),
            "partial_report_md": str(report_md),
        },
    }
    manifest_json.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Stage 6 Partial Progress Report",
        "",
        "This report is provisional validation-only evidence. It is safe to resume Stage 6 from the same output directory with identical settings and `SKIP_EXISTING_RUNS=True`.",
        "",
        f"- Canonical Stage 6 outputs written: `{bool(canonical_outputs_written)}`",
        f"- Stage 6 complete: `{bool(completion.get('stage6_complete'))}`",
        f"- Expected candidates: `{completion.get('expected_candidate_count')}`",
        f"- Observed candidates with completed units: `{completion.get('observed_candidate_count')}`",
        f"- Missing candidates: `{completion.get('missing_candidate_count')}`",
        "",
        "## Candidate Completion",
        "",
        "| candidate_id | status | units | fold_seed_units | folds | seeds | mean_val_metal_balanced_acc |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {candidate} | {status} | {units}/{planned_units} | {fold_units}/{planned_fold_units} | {folds}/{planned_folds} | {seeds}/{planned_seeds} | {score} |".format(
                candidate=row.get("candidate_id", ""),
                status=row.get("stage6_completion_status", "partial"),
                units=row.get("n_units_completed", ""),
                planned_units=row.get("n_planned_units", ""),
                fold_units=row.get("n_fold_seed_units_completed", row.get("n_units_completed", "")),
                planned_fold_units=row.get("n_planned_fold_seed_units", row.get("n_planned_units", "")),
                folds=row.get("n_folds_completed", ""),
                planned_folds=row.get("n_planned_folds", ""),
                seeds=row.get("n_seeds_completed", ""),
                planned_seeds=row.get("n_planned_seeds", ""),
                score=row.get("mean_val_metal_balanced_acc", ""),
            )
        )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "partial_dir": partial_dir,
        "partial_results_csv": results_csv,
        "partial_ranked_candidates_csv": ranked_csv,
        "partial_pairwise_bootstrap_csv": pairwise_csv,
        "partial_manifest_json": manifest_json,
        "partial_report_md": report_md,
    }


def _summary_rows(records: list[dict[str, Any]], selection_metric: str, units: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    planned = _planned_stage6_counts(units or [])
    for record in records:
        if record.get("status") not in {"completed", "existing"}:
            continue
        groups.setdefault(str(record["candidate_id"]), []).append(record)
    rows = []
    for candidate_id, items in groups.items():
        values = [_numeric(item.get("selected_best_validation_metric_value")) for item in items]
        values = [value for value in values if value is not None]
        if not values:
            continue
        balanced = [_numeric(item.get("val_metal_balanced_acc")) for item in items]
        balanced = [value for value in balanced if value is not None]
        min_recalls = [_numeric(item.get("val_metal_min_recall")) for item in items]
        min_recalls = [value for value in min_recalls if value is not None]
        source_dirs = sorted({str(item.get("run_dir")) for item in items if item.get("run_dir")})
        fold_seed_units = {
            f"{item.get('fold_unit')}|seed_{item.get('model_seed')}"
            for item in items
            if item.get("fold_unit") not in (None, "") and item.get("model_seed") not in (None, "")
        }
        row = {
            "candidate_id": candidate_id,
            "selection_metric": selection_metric,
            "mean_validation_metric": statistics.mean(values) if values else None,
            "std_validation_metric": statistics.stdev(values) if len(values) > 1 else 0.0,
            "mean_val_metal_balanced_acc": statistics.mean(balanced) if balanced else None,
            "mean_val_metal_min_recall": statistics.mean(min_recalls) if min_recalls else None,
            "n_units_completed": len(items),
            "n_planned_units": planned["n_planned_units"],
            "n_fold_seed_units_completed": len(fold_seed_units),
            "n_planned_fold_seed_units": planned["n_planned_fold_seed_units"],
            "n_folds_completed": len({item.get("fold_unit") for item in items if item.get("fold_unit")}),
            "n_planned_folds": planned["n_planned_folds"],
            "n_seeds_completed": len({item.get("model_seed") for item in items if item.get("model_seed") is not None}),
            "n_planned_seeds": planned["n_planned_seeds"],
            "planned_model_seeds": planned["planned_model_seeds"],
            "validation_units": ",".join(str(item.get("validation_unit")) for item in items),
            "selected_stage6_source_run_dirs": ";".join(source_dirs),
            "primary_source_run_dir": source_dirs[0] if source_dirs else "",
            "primary_source_checkpoint": str(items[0].get("selected_checkpoint") or ""),
            "top_config_reevaluation_mode": str(items[0].get("top_config_reevaluation_mode") or ""),
            "primary_score_definition": f"mean {selection_metric} over completed Stage 6 fold/active-seed runs",
        }
        rows.append(row)
    rows.sort(
        key=lambda row: (
            row.get("mean_validation_metric") if row.get("mean_validation_metric") is not None else float("-inf"),
            row.get("mean_val_metal_min_recall") if row.get("mean_val_metal_min_recall") is not None else float("-inf"),
            -(row.get("std_validation_metric") if row.get("std_validation_metric") is not None else float("inf")),
        ),
        reverse=True,
    )
    for index, row in enumerate(rows, start=1):
        row["stage6_rank"] = index
        row["selected_for_final"] = index == 1
    return rows


def _paired_fold_differences(
    records: list[dict[str, Any]],
    candidate_a: str,
    candidate_b: str,
    selection_metric: str,
) -> tuple[list[float], int]:
    values: dict[str, dict[tuple[str, int], float]] = {candidate_a: {}, candidate_b: {}}
    for record in records:
        if record.get("status") not in {"completed", "existing"}:
            continue
        candidate_id = str(record.get("candidate_id") or "")
        if candidate_id not in values:
            continue
        metric_value = _numeric(record.get("selected_best_validation_metric_value"))
        if metric_value is None:
            metric_value = _numeric(record.get(selection_metric))
        if metric_value is None:
            continue
        fold = str(record.get("fold_unit") or record.get("validation_unit") or "")
        seed = record.get("model_seed")
        try:
            seed_int = int(seed)
        except Exception:
            seed_int = 0
        values[candidate_id][(fold, seed_int)] = float(metric_value)

    common_units = sorted(set(values[candidate_a]) & set(values[candidate_b]))
    by_fold: dict[str, list[float]] = {}
    for fold, seed in common_units:
        by_fold.setdefault(fold, []).append(values[candidate_a][(fold, seed)] - values[candidate_b][(fold, seed)])
    fold_diffs = [statistics.mean(diffs) for fold, diffs in sorted(by_fold.items()) if diffs]
    return fold_diffs, len(common_units)


def _bootstrap_mean_ci(values: list[float], *, n_bootstrap: int = 10000, seed: int = 20260522) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    mean_value = statistics.mean(values)
    if len(values) == 1:
        return mean_value, mean_value, mean_value
    rng = random.Random(seed)
    count = len(values)
    means = []
    for _ in range(n_bootstrap):
        means.append(statistics.mean(values[rng.randrange(count)] for _ in range(count)))
    means.sort()
    lower_idx = max(0, min(len(means) - 1, int(0.025 * (len(means) - 1))))
    upper_idx = max(0, min(len(means) - 1, int(0.975 * (len(means) - 1))))
    return mean_value, means[lower_idx], means[upper_idx]


def _pairwise_rows(
    records: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    selection_metric: str,
    raw_improvement_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ordered_ids = [str(row.get("candidate_id") or "") for row in summary_rows if row.get("candidate_id")]
    for i, candidate_a in enumerate(ordered_ids):
        for candidate_b in ordered_ids[i + 1 :]:
            fold_diffs, n_unit_pairs = _paired_fold_differences(records, candidate_a, candidate_b, selection_metric)
            mean_diff, ci_lower, ci_upper = _bootstrap_mean_ci(fold_diffs)
            rows.append(
                {
                    "candidate_a": candidate_a,
                    "candidate_b": candidate_b,
                    "selection_metric": selection_metric,
                    "difference": "candidate_a_minus_candidate_b",
                    "n_common_fold_seed_pairs": n_unit_pairs,
                    "n_common_folds": len(fold_diffs),
                    "mean_difference": mean_diff,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "raw_improvement_threshold": raw_improvement_threshold,
                    "passes_raw_threshold": (mean_diff is not None and mean_diff >= raw_improvement_threshold),
                    "passes_paired_bootstrap": (ci_lower is not None and ci_lower > raw_improvement_threshold),
                    "pairing_rule": "mean common seeds within each shared fold, then bootstrap fold-level differences",
                }
            )
    return rows


def run_stage6_standalone(
    *,
    existing_runs_dir: str | Path,
    output_runs_dir: str | Path | None = None,
    output_study_name: str | None = None,
    top_k: str | int = "auto",
    reevaluation_mode: str = "group_kfold_seed_repeat",
    repeat_seeds: str = "42",
    n_folds: int = 5,
    split_seed: int = 42,
    epochs: int | None = 50,
    device: str = "auto",
    selection_metric: str = "val_metal_balanced_acc",
    raw_improvement_threshold: float = 0.0,
    skip_existing_runs: bool = True,
    overwrite_output: bool = False,
    parallel_cross_validation_processes: int = 1,
    write_partial_progress: bool = False,
    launch: bool = False,
    repo_dir: str | Path | None = None,
) -> dict[str, Any]:
    repo = Path(repo_dir or Path.cwd()).expanduser().resolve()
    existing_dir = Path(existing_runs_dir).expanduser()
    if not existing_dir.exists() or not existing_dir.is_dir():
        raise RuntimeError(f"Existing HPO/training directory does not exist: {existing_dir}")
    if output_runs_dir:
        runs_dir = Path(output_runs_dir).expanduser()
    else:
        runs_dir = existing_dir.parent / f"{existing_dir.name}_stage6"
    if _path_contains(runs_dir, existing_dir):
        raise RuntimeError(
            "Standalone Stage 6 output directory must not be the existing HPO source directory "
            f"or a parent directory containing it. source={existing_dir}; output={runs_dir}"
        )
    if runs_dir.exists() and not runs_dir.is_dir():
        raise RuntimeError(f"Standalone Stage 6 output path exists but is not a directory: {runs_dir}")
    runs_dir_preexisting = runs_dir.exists()
    study_name = _slug(output_study_name or runs_dir.name or f"{existing_dir.name}_stage6")
    optuna_dir = runs_dir / "optuna" / study_name
    parallel_processes = int(parallel_cross_validation_processes or 1)
    if parallel_processes < 1:
        raise ValueError("parallel_cross_validation_processes must be >= 1.")
    requested_device = str(device)
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    seeds = _parse_ints(repeat_seeds, "REPEAT_SEEDS")
    units = _stage6_units(reevaluation_mode, seeds, int(n_folds), int(split_seed))
    candidates, report_rows = _discover_candidates(existing_dir, selection_metric)
    ranked = _rank_candidates(candidates, selection_metric)
    chosen_k = _resolve_top_k(top_k, len(ranked))
    selected = ranked[:chosen_k]
    selected_ids = {candidate.candidate_id for candidate in selected}
    for row in report_rows:
        if row["candidate_id"] in selected_ids and row["status"] == "compatible":
            row["status"] = "selected_for_stage6"

    if not selected:
        optuna_dir.mkdir(parents=True, exist_ok=True)
        _write_rows_csv(optuna_dir / "stage6_existing_trials_import_report.csv", report_rows)
        raise RuntimeError(f"No compatible completed validation-only candidates found in {existing_dir}.")

    manifest = _stage6_manifest_payload(
        existing_dir=existing_dir,
        runs_dir=runs_dir,
        optuna_dir=optuna_dir,
        study_name=study_name,
        top_k_requested=top_k,
        top_k_resolved=chosen_k,
        reevaluation_mode=reevaluation_mode,
        repeat_seeds=seeds,
        n_folds=int(n_folds),
        split_seed=int(split_seed),
        epochs=epochs,
        device=requested_device,
        selection_metric=selection_metric,
        raw_improvement_threshold=float(raw_improvement_threshold),
        selected=selected,
    )
    root_manifest_path = runs_dir / "stage6_manifest.json"
    optuna_manifest_path = optuna_dir / "stage6_manifest.json"
    existing_manifest = _read_json(root_manifest_path)
    if not existing_manifest:
        existing_manifest = _read_json(optuna_manifest_path)
    output_has_contents = False
    if runs_dir_preexisting:
        try:
            output_has_contents = any(runs_dir.iterdir())
        except Exception:
            output_has_contents = True
    mismatches: list[dict[str, Any]] = []
    if runs_dir_preexisting and existing_manifest:
        mismatches = _stage6_manifest_mismatches(existing_manifest, manifest)
    elif runs_dir_preexisting and output_has_contents:
        mismatches = [
            {
                "field": "stage6_manifest.json",
                "existing": "missing",
                "requested": "required for continuing a non-empty standalone Stage 6 output directory",
            }
        ]
    if mismatches:
        mismatch_text = _format_stage6_manifest_mismatches(mismatches)
        if not overwrite_output:
            raise RuntimeError(
                "Existing standalone Stage 6 output is not compatible with the requested settings. "
                "Set STAGE6_OVERWRITE_OUTPUT=True only if you intentionally want to replace it.\n"
                + mismatch_text
            )
        print("STAGE6_OVERWRITE_OUTPUT=True. Replacing incompatible standalone Stage 6 output directory:", runs_dir)
        print(mismatch_text)
        shutil.rmtree(runs_dir)
        runs_dir_preexisting = False
    elif runs_dir_preexisting and existing_manifest:
        print("Existing standalone Stage 6 output manifest matches requested settings; continuing:", runs_dir)

    runs_dir.mkdir(parents=True, exist_ok=True)
    optuna_dir.mkdir(parents=True, exist_ok=True)
    root_manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8")
    optuna_manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8")

    reruns = _build_rerun_configs(
        selected,
        units,
        output_runs_dir=runs_dir,
        repo_dir=repo,
        epochs=epochs,
        device=str(device),
        selection_metric=selection_metric,
    )
    top_rows = [
        {
            "rank": index,
            "candidate_id": f"imported_top{index}_{_slug(candidate.candidate_id)}",
            "imported_candidate_id": candidate.candidate_id,
            "run_dir": str(candidate.run_dir),
            "selection_metric": selection_metric,
            "validation_metric": _ranking_value(candidate, selection_metric),
            "val_metal_balanced_acc": candidate.val_metal_balanced_acc,
            "val_metal_min_recall": candidate.val_metal_min_recall,
            "selected_checkpoint": candidate.selected_checkpoint,
        }
        for index, candidate in enumerate(selected, start=1)
    ]
    commands = [
        f"# Validation-only standalone Stage 6 {reevaluation_mode} commands generated from {existing_dir}",
        "# These commands intentionally omit held-out test evaluation.",
        "",
    ] + [str(config["shell_command"]) for config in reruns]

    import_report_csv = optuna_dir / "stage6_existing_trials_import_report.csv"
    import_report_json = optuna_dir / "stage6_existing_trials_import_report.json"
    top_trials_csv = optuna_dir / "top_trials.csv"
    top_trial_configs_json = optuna_dir / "top_trial_configs.json"
    commands_txt = optuna_dir / "top_reevaluation_commands.txt"
    _write_rows_csv(import_report_csv, report_rows)
    import_report_json.write_text(
        json.dumps(
            _json_safe(
                {
                    "created_by": "stage6_standalone.py",
                    "existing_runs_dir": str(existing_dir),
                    "output_runs_dir": str(runs_dir),
                    "stage6_manifest_json": str(root_manifest_path),
                    "n_candidates_discovered": len(report_rows),
                    "n_compatible_completed_candidates": len(candidates),
                    "top_k_selected_for_stage6": chosen_k,
                    "selection_metric": selection_metric,
                    "held_out_test_policy": "Held-out test artifacts are skipped and test metrics are not used for ranking.",
                    "rows": report_rows,
                }
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_rows_csv(top_trials_csv, top_rows)
    top_trial_configs_json.write_text(
        json.dumps(
            _json_safe(
                [
                    {
                        "rank": index,
                        "candidate_id": f"imported_top{index}_{_slug(candidate.candidate_id)}",
                        "original_source_run_dir": str(candidate.run_dir),
                        "full_config": candidate.config,
                    }
                    for index, candidate in enumerate(selected, start=1)
                ]
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    commands_txt.write_text("\n".join(commands) + "\n", encoding="utf-8")

    def _run_one_stage6_config(config: dict[str, Any], *, print_to_console: bool) -> dict[str, Any]:
        run_dir = Path(config["run_dir"])
        status = "planned"
        return_code = None
        error_tail = ""
        if (skip_existing_runs or not launch) and (run_dir / "run_metadata.json").exists():
            status = "existing"
            print("Existing Stage 6 run:", run_dir)
        else:
            print("=" * 80)
            print(f"[Standalone Stage 6 {config['candidate_id']} {config['validation_unit']}]")
            print(config["shell_command"])
            _write_active_snapshot(
                run_dir,
                config,
                {
                    "reevaluation_mode": reevaluation_mode,
                    "existing_runs_dir": str(existing_dir),
                    "stage6_manifest_json": str(root_manifest_path),
                    "original_source_run_dir": config.get("source_run_dir"),
                    "stage6_parallel_cross_validation_processes": parallel_processes,
                    "stage6_parallel_scope": "within one ranked candidate; next candidate waits for current candidate's units",
                },
            )
            return_code, error_tail = _stream_command(
                config["command"],
                cwd=repo,
                stdout_log=Path(config["stdout_log_path"]),
                stderr_log=Path(config["stderr_log_path"]),
                print_to_console=print_to_console,
            )
            status = "completed" if return_code == 0 else "failed"
        candidate_metrics = _candidate_from_run_dir(run_dir, selection_metric) if status in {"completed", "existing"} else None
        return {
            "run_tag": f"{reevaluation_mode}_{config['candidate_id']}_{config['validation_unit']}",
            "candidate_id": config["candidate_id"],
            "imported_candidate_id": config["imported_candidate_id"],
            "validation_unit": config["validation_unit"],
            "fold_unit": config["fold_unit"],
            "model_seed": config["seed"],
            "split_seed": config["split_seed"],
            "n_folds": config["n_folds"],
            "fold_index": config["fold_index"],
            "status": status,
            "return_code": return_code,
            "error_message": "" if status in {"completed", "existing"} else error_tail,
            "selection_metric": selection_metric,
            "selected_best_validation_metric_value": candidate_metrics.selected_metric_value if candidate_metrics else None,
            "val_metal_balanced_acc": candidate_metrics.val_metal_balanced_acc if candidate_metrics else None,
            "val_metal_min_recall": candidate_metrics.val_metal_min_recall if candidate_metrics else None,
            "selected_checkpoint": candidate_metrics.selected_checkpoint if candidate_metrics else None,
            "run_name": config["run_name"],
            "run_dir": config["run_dir"],
            "source_run_dir": config.get("source_run_dir"),
            "top_config_reevaluation_mode": reevaluation_mode,
            "model_architecture": config.get("model_architecture"),
            "fusion_mode": config.get("fusion_mode"),
            "model_preset": config.get("model_preset"),
            "learning_rate": config.get("learning_rate"),
            "weight_decay": config.get("weight_decay"),
            "batch_size": config.get("batch_size"),
            "stdout_log_path": config["stdout_log_path"],
            "stderr_log_path": config["stderr_log_path"],
            "stage6_parallel_cross_validation_processes": parallel_processes,
            "stage6_parallel_scope": "candidate",
        }

    records = []
    if launch:
        print("Running standalone Stage 6 grouped-fold confirmation")
        print(
            "Stage 6 parallel cross-validation processes:",
            parallel_processes,
            "(within each ranked candidate; the next candidate waits for the current candidate batch)",
        )
        for batch in _stage6_candidate_batches(reruns):
            candidate_id = str(batch[0].get("candidate_id") or "")
            workers = min(parallel_processes, int(n_folds), len(batch))
            print(f"Starting Stage 6 candidate batch {candidate_id}: {len(batch)} validation unit(s), workers={workers}")
            if workers == 1:
                for config in batch:
                    records.append(_run_one_stage6_config(config, print_to_console=True))
            else:
                batch_records: list[dict[str, Any]] = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(_run_one_stage6_config, config, print_to_console=False): config
                        for config in batch
                    }
                    for future in concurrent.futures.as_completed(futures):
                        config = futures[future]
                        record = future.result()
                        batch_records.append(record)
                        print(
                            "Finished Stage 6 unit:",
                            config["candidate_id"],
                            config["validation_unit"],
                            "status=",
                            record.get("status"),
                            "metric=",
                            record.get("selected_best_validation_metric_value"),
                            "log=",
                            config.get("stdout_log_path"),
                        )
                batch_records.sort(key=lambda row: (int(row.get("fold_index") or 0), int(row.get("model_seed") or 0), str(row.get("validation_unit") or "")))
                records.extend(batch_records)
            print(f"Finished Stage 6 candidate batch {candidate_id}")
    else:
        print("Standalone Stage 6 preview only. Commands were written but no runs were launched.")
        for config in reruns:
            run_dir = Path(config["run_dir"])
            if (run_dir / "run_metadata.json").exists():
                records.append(_run_one_stage6_config(config, print_to_console=False))
        if records:
            print(f"Collected {len(records)} existing Stage 6 run(s) for partial progress reporting.")

    results_csv = optuna_dir / "seed_repeat_results.csv"
    summary_csv = optuna_dir / "seed_repeat_summary.csv"
    summary_json = optuna_dir / "seed_repeat_summary.json"
    pairwise_csv = optuna_dir / "seed_repeat_pairwise_bootstrap.csv"
    pairwise_json = optuna_dir / "seed_repeat_pairwise_bootstrap.json"
    ranked_csv = optuna_dir / "stage6_ranked_candidates.csv"
    selected_json = optuna_dir / "stage6_selected_final_candidate.json"
    partial_outputs: dict[str, Path] = {}
    if records:
        summary_rows = _summary_rows(records, selection_metric, units)
        completion = _stage6_completion_status(summary_rows, expected_candidate_count=chosen_k, units=units)
        summary_rows = completion["rows"]
        pairwise_rows = _pairwise_rows(records, summary_rows, selection_metric, float(raw_improvement_threshold))
        canonical_complete = bool(completion.get("stage6_complete"))
        if write_partial_progress:
            partial_outputs = _write_stage6_partial_outputs(
                optuna_dir=optuna_dir,
                records=records,
                summary_rows=summary_rows,
                pairwise_rows=pairwise_rows,
                completion=completion,
                manifest=manifest,
                canonical_outputs_written=canonical_complete,
            )
        if canonical_complete:
            _write_rows_csv(results_csv, records)
            _write_rows_csv(summary_csv, summary_rows)
            _write_rows_csv(ranked_csv, summary_rows)
            summary_json.write_text(json.dumps(_json_safe(summary_rows), indent=2, sort_keys=True), encoding="utf-8")
            _write_rows_csv(pairwise_csv, pairwise_rows)
            pairwise_json.write_text(json.dumps(_json_safe(pairwise_rows), indent=2, sort_keys=True), encoding="utf-8")
            selected_payload = {
                "created_by": "stage6_standalone.py",
                "protocol_stage": "Stage 6",
                "selection_basis": "validation/CV metrics only; held-out test metrics were not used",
                "selected_config_id": summary_rows[0]["candidate_id"] if summary_rows else None,
                "primary_source_run_dir": summary_rows[0]["primary_source_run_dir"] if summary_rows else "",
                "primary_source_checkpoint": summary_rows[0]["primary_source_checkpoint"] if summary_rows else "",
                "selected_ranking_metrics": summary_rows[0] if summary_rows else {},
                "import_report_csv": str(import_report_csv),
                "final_test_policy": {
                    "primary_final_test": "Evaluate only this frozen Stage-6-selected candidate after Stage 6B final refit.",
                    "no_test_selection": "Do not choose among candidates based on held-out test performance.",
                },
            }
            selected_json.write_text(json.dumps(_json_safe(selected_payload), indent=2, sort_keys=True), encoding="utf-8")
        else:
            print("Stage 6 is not complete; canonical Stage 6 selection files were not written.")
            if partial_outputs:
                print("Partial Stage 6 progress report:", partial_outputs.get("partial_report_md"))
            else:
                print("Partial Stage 6 progress report writing is disabled; Stage 6B can reconstruct preview-only partial artifacts if needed.")

    return {
        "mode": "standalone",
        "existing_runs_dir": str(existing_dir),
        "output_runs_dir": str(runs_dir),
        "optuna_dir": str(optuna_dir),
        "stage6_manifest_json": str(root_manifest_path),
        "import_report_csv": str(import_report_csv),
        "top_reevaluation_commands_txt": str(commands_txt),
        "n_ranked_candidates": len(ranked),
        "top_k": chosen_k,
        "n_stage6_runs": len(reruns),
        "overwrite_output": bool(overwrite_output),
        "parallel_cross_validation_processes": parallel_processes,
        "parallel_scope": "within each ranked candidate; candidates run sequentially by Stage 6 rank",
        "launched": bool(launch),
        "write_partial_progress": bool(write_partial_progress),
        "partial_outputs": {key: str(value) for key, value in partial_outputs.items()},
        "records": records,
    }
