"""Campaign contracts enforced inside a training run.

- frozen fold membership: the computed k-fold split must equal ``fold_membership.csv``;
- selected-checkpoint validation replay: native and common-four predictions from the
  one selected checkpoint, keyed by stable source identity;
- readout diagnostics: learned first-shell biases and empty-shell support.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from data_structures import PocketRecord
from label_schemes import COLLAPSED_METAL_LABELS, METAL_TARGET_LABELS
from training.final_test_reporting import collapse_metal_probabilities, collapse_metal_targets, metal_metrics_from_probabilities
from training.source_cohort import sha256_file

FOLD_MEMBERSHIP_COLUMNS = ("source_uid", "physical_ion_id", "group_id", "native_element", "fold")
RECONCILIATION_TOLERANCE = 1.0e-9


class CampaignContractError(RuntimeError):
    """A frozen campaign contract was violated inside a run."""


def read_fold_membership(path: Path, expected_sha256: str) -> dict[str, dict[str, str]]:
    actual = sha256_file(Path(path))
    if actual != expected_sha256.lower():
        raise CampaignContractError(f"Fold membership {path} has SHA256 {actual}, expected {expected_sha256}")
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in FOLD_MEMBERSHIP_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise CampaignContractError(f"Fold membership {path} lacks columns {missing}")
        rows = {row["source_uid"]: row for row in reader}
    return rows


def membership_identity_sha256(rows: list[dict[str, Any]]) -> str:
    """Hash of ion identity and fold only (never labels of a scheme or model seeds)."""
    payload = "\n".join(
        f"{row['source_uid']}\t{row['physical_ion_id']}\t{row['group_id']}\t{int(row['fold'])}"
        for row in sorted(rows, key=lambda item: item["source_uid"])
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_fold_membership(
    *,
    train_pockets: list[PocketRecord],
    val_pockets: list[PocketRecord],
    fold_index: int,
    membership: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Fail unless this run's split is exactly the frozen fold ``fold_index``."""
    def uid(pocket: PocketRecord) -> str:
        value = pocket.metadata.get("source_uid")
        if value is None:
            raise CampaignContractError(f"Example {pocket.pocket_id} lacks source_uid")
        return str(value)

    train_uids = [uid(pocket) for pocket in train_pockets]
    val_uids = [uid(pocket) for pocket in val_pockets]
    loaded = set(train_uids) | set(val_uids)
    if len(loaded) != len(train_uids) + len(val_uids):
        raise CampaignContractError("An example occurs in both partitions or twice")
    if loaded != set(membership):
        raise CampaignContractError(
            f"Loaded examples differ from frozen membership ({len(loaded)} vs {len(membership)})"
        )
    expected_val = {key for key, row in membership.items() if int(row["fold"]) == int(fold_index)}
    if set(val_uids) != expected_val:
        raise CampaignContractError(
            f"Computed validation fold {fold_index} differs from frozen membership: "
            f"{len(set(val_uids) - expected_val)} unexpected, {len(expected_val - set(val_uids))} missing"
        )
    for pocket in [*train_pockets, *val_pockets]:
        row = membership[uid(pocket)]
        if row["physical_ion_id"] != pocket.metadata.get("physical_ion_id"):
            raise CampaignContractError(f"Physical ion differs for {row['source_uid']}")
        if row["native_element"].upper() != str(pocket.metal_element).upper():
            raise CampaignContractError(f"Native element differs for {row['source_uid']}")
    return {
        "fold_index": int(fold_index),
        "n_train": len(train_uids),
        "n_val": len(val_uids),
        "membership_identity_sha256": membership_identity_sha256(list(membership.values())),
        "verified": True,
    }


def binding_bias_state(model: torch.nn.Module) -> dict[str, float]:
    """Current learned first-shell readout biases, flattened for epoch logs."""
    state: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if "binding_bias." in name:
            state["readout_bias." + name.replace("binding_bias.", "")] = float(parameter.detach().cpu().item())
    return state


def first_shell_support(graphs: list[Any] | None) -> dict[str, int]:
    if not graphs:
        return {"n_graphs": 0, "n_empty_first_shell": 0}
    roles = [getattr(graph, "x_role", None) for graph in graphs]
    if any(role is None for role in roles):
        return {"n_graphs": len(graphs), "n_empty_first_shell": None}
    empty = sum(1 for role in roles if not bool((role[:, 0] > 0.5).any().item()))
    return {"n_graphs": len(graphs), "n_empty_first_shell": int(empty)}


def _selected_record(history: list[dict[str, Any]], epoch: int) -> dict[str, Any]:
    rows = [row for row in history if int(row["epoch"]) == int(epoch)]
    if len(rows) != 1:
        raise CampaignContractError(f"Selected epoch {epoch} is missing or duplicated in the history")
    return rows[0]


def export_selected_validation_predictions(
    *,
    model: torch.nn.Module,
    val_loader,
    val_pockets: list[PocketRecord],
    best_checkpoint: dict[str, Any],
    checkpoint_path: Path,
    history: list[dict[str, Any]],
    config_payload: dict[str, Any],
    run_dir: Path,
    device: str,
) -> dict[str, Any]:
    """Replay the selected checkpoint on validation; write UID-keyed predictions.

    Native and common-four metrics come from the same selected checkpoint. The
    common-four view sums six-class probabilities before the argmax.
    """
    from training.loop import evaluate_epoch_with_predictions

    model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
    predictions = evaluate_epoch_with_predictions(model, val_loader, device=device)
    logits = predictions["metal_logits"].float()
    targets = predictions["metal_y"].long()
    if not bool(torch.isfinite(logits).all()):
        raise CampaignContractError("Selected-checkpoint replay produced nonfinite logits")
    if targets.tolist() != [int(pocket.y_metal) for pocket in val_pockets]:
        raise CampaignContractError("Replayed validation targets or ordering differ from the split")
    probabilities = torch.softmax(logits, dim=-1)
    metrics = metal_metrics_from_probabilities(probabilities, targets, prefix="val")
    selected = _selected_record(history, int(best_checkpoint["epoch"]))
    reconciliation = {}
    reconciliation_status = "match"
    for key in ("val_metal_balanced_acc", "val_metal_collapsed4_balanced_acc"):
        replayed, recorded = float(metrics[key]), float(selected[key])
        reconciliation[key] = {"replayed": replayed, "recorded": recorded, "abs_difference": abs(replayed - recorded)}
        if not (abs(replayed - recorded) <= RECONCILIATION_TOLERANCE):
            reconciliation_status = "mismatch"
    from training.loop import classification_metrics_from_logits

    matrices = {
        "val_metal_confusion_matrix": classification_metrics_from_logits(logits, targets)["confusion_matrix"],
        "val_metal_collapsed4_confusion_matrix": classification_metrics_from_logits(
            collapse_metal_probabilities(probabilities).clamp_min(1e-30).log(),
            collapse_metal_targets(targets),
        )["confusion_matrix"],
    }
    for key, matrix in matrices.items():
        if key not in selected or matrix != selected[key]:
            reconciliation[key] = {"replayed": matrix, "recorded": selected.get(key)}
            reconciliation_status = "mismatch"
    collapsed_probabilities = collapse_metal_probabilities(probabilities)
    collapsed_targets = collapse_metal_targets(targets)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    native_labels = [METAL_TARGET_LABELS[index] for index in range(len(METAL_TARGET_LABELS))]
    four_labels = [COLLAPSED_METAL_LABELS[index] for index in range(len(COLLAPSED_METAL_LABELS))]
    rows = []
    for index, pocket in enumerate(val_pockets):
        row = {
            "source_uid": pocket.metadata["source_uid"],
            "alias_source_uids": ";".join(pocket.metadata.get("alias_source_uids", [])),
            "physical_ion_id": pocket.metadata.get("physical_ion_id"),
            "group_id": pocket.metadata.get("cohort_group_id"),
            "structure_id": pocket.structure_id,
            "example_id": pocket.pocket_id,
            "parent_pocket_id": pocket.metadata.get("parent_pocket_id"),
            "native_element": pocket.metal_element,
            "fold": config_payload.get("fold_index"),
            "model_seed": config_payload.get("seed"),
            "metal_label_scheme": config_payload.get("metal_label_scheme"),
            "binding_residue_pooling": config_payload.get("binding_residue_pooling", "none"),
            "selected_epoch": int(best_checkpoint["epoch"]),
            "checkpoint_sha256": checkpoint_sha256,
            "y_native": int(targets[index]),
            "pred_native": int(probabilities[index].argmax()),
            "y_common4": int(collapsed_targets[index]),
            "pred_common4": int(collapsed_probabilities[index].argmax()),
        }
        for label_index, label in enumerate(native_labels):
            row[f"p_native_{label.replace(' ', '_')}"] = f"{float(probabilities[index, label_index]):.8f}"
        for label_index, label in enumerate(four_labels):
            row[f"p_common4_{label.replace(' ', '_')}"] = f"{float(collapsed_probabilities[index, label_index]):.8f}"
        rows.append(row)
    predictions_path = run_dir / "val_predictions.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    receipt = {
        "fit_status": ("replay_failed" if reconciliation_status != "match" else
                       "completed" if len(history) == int(config_payload["epochs"]) else "incomplete"),
        "completed_epochs": len(history),
        "planned_epochs": int(config_payload["epochs"]),
        "selected_checkpoint": checkpoint_path.name,
        "selected_checkpoint_sha256": checkpoint_sha256,
        "selected_epoch": int(best_checkpoint["epoch"]),
        "selection_metric": best_checkpoint.get("selection_metric"),
        "selection_metric_value": best_checkpoint.get("selection_metric_value"),
        "tie_rule": "earliest epoch (strictly greater metric required to replace)",
        "campaign_run_identity": json.loads(config_payload["campaign_run_identity"])
        if config_payload.get("campaign_run_identity") else None,
        "validation_predictions": {"path": predictions_path.name, "sha256": sha256_file(predictions_path),
                                   "n_rows": len(rows)},
        "metrics": {key: value for key, value in metrics.items()},
        "reconciliation": reconciliation,
        "reconciliation_status": reconciliation_status,
        "common_four_rule": "sum native probabilities into [Mn, Cu, Zn, Fe+Co+Ni] before argmax",
    }
    (run_dir / "selected_checkpoint.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8",
    )
    if reconciliation_status != "match":
        raise CampaignContractError("Selected-checkpoint replay does not reconcile; artifacts preserved, fit not certified")
    return receipt


def load_campaign_prediction_components(checkpoint: dict[str, Any], *, device: str = "cpu"):
    """Rebuild saved model/normalization/graph options without training preparation."""
    from export_validation_predictions import factory_kwargs
    from label_schemes import configure_active_metal_label_scheme
    from model_variants import build_pocket_classifier
    from training.run import normalization_stats_from_payload

    config = checkpoint["config"]
    configure_active_metal_label_scheme(config["metal_label_scheme"])
    metal_labels = {int(key): value for key, value in checkpoint["metal_labels"].items()}
    ec_labels = {int(key): value for key, value in checkpoint["ec_labels"].items()}
    if metal_labels != METAL_TARGET_LABELS:
        raise CampaignContractError("Checkpoint vocabulary differs from its saved target scheme")
    kwargs = factory_kwargs(Path(__file__).with_name("run.py"), config, checkpoint, metal_labels, ec_labels)
    model = build_pocket_classifier(**kwargs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    options = {key: config[key] for key in (
        "esm_dim", "edge_radius", "use_ring_edges", "require_ring_edges", "shell_role_source",
        "node_feature_set", "omit_node_features", "metal_node_mode",
    )}
    return model, normalization_stats_from_payload(checkpoint["normalization_stats"]), options


def replay_campaign_run(run_dir: Path, *, device: str = "cpu", path_map: dict[str, str] | None = None,
                        output_dir: Path | None = None) -> dict[str, Any]:
    """Independently reload a campaign's selected checkpoint and frozen validation ions.

    Only validation structures are parsed. Saved normalization/class weights are
    reused; no optimizer, data preparation or training/refit routine is called.
    """
    from export_validation_predictions import remap_path, verify_membership
    from torch_geometric.loader import DataLoader
    from training.access_guard import install_forbidden_read_guard
    from training.data import _cohort_structure_files
    from training.graph_dataset import PocketGraphDataset, build_graph_data_list
    from training.source_cohort import read_cohort_csv, cohort_load_payload
    from training.structure_loading import load_structure_pockets
    from training.run import to_jsonable

    run_dir = Path(run_dir)
    payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    config = payload["config"]
    if (config.get("task") != "metal" or config.get("metal_example_unit") != "ion"
            or not config.get("source_cohort_csv") or not config.get("fold_membership_csv")
            or config.get("run_test_eval") or config.get("test_structure_dir") or config.get("test_summary_csv")
            or config.get("allow_final_refit_test_eval") or config.get("allow_train_loss_test_eval_debug")):
        raise CampaignContractError("Independent campaign replay requires validation-only frozen source-cohort configuration")
    mapped = {key: remap_path(config.get(key), path_map or {}) for key in (
        "structure_dir", "source_cohort_csv", "fold_membership_csv", "esm_embeddings_dir",
        "external_features_root_dir", "ring_features_dir",
    )}
    train_dir = mapped["structure_dir"]
    from benchmarking.pmm_ion_campaign import forbidden_read_roots

    install_forbidden_read_guard(forbidden_read_roots(train_dir))
    checkpoint_path = run_dir / "best_model_checkpoint.pt"
    selected_receipt = json.loads((run_dir / "selected_checkpoint.json").read_text(encoding="utf-8"))
    if sha256_file(checkpoint_path) != selected_receipt["selected_checkpoint_sha256"]:
        raise CampaignContractError("Selected checkpoint content changed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if to_jsonable(checkpoint["config"]) != config:
        raise CampaignContractError("Selected checkpoint and saved run configuration differ")
    saved_prediction_path = run_dir / selected_receipt["validation_predictions"]["path"]
    if sha256_file(saved_prediction_path) != selected_receipt["validation_predictions"]["sha256"]:
        raise CampaignContractError("Saved validation prediction content changed")
    model, normalization, graph_options = load_campaign_prediction_components(checkpoint, device=device)
    bindings = read_cohort_csv(mapped["source_cohort_csv"], config["source_cohort_sha256"])
    membership = read_fold_membership(mapped["fold_membership_csv"], config["fold_membership_sha256"])
    wanted = {uid for uid, row in membership.items() if int(row["fold"]) == int(config["fold_index"])}
    bindings = [binding for binding in bindings if binding.source_uid in wanted]
    if {binding.source_uid for binding in bindings} != wanted:
        raise CampaignContractError("Frozen validation UIDs are not present in the source cohort")
    structure_files = _cohort_structure_files(train_dir, bindings)
    binding_payload = cohort_load_payload(bindings)
    pockets = []
    for path in structure_files:
        loaded, _, _ = load_structure_pockets(
            structure_path=path, structure_root=train_dir, allowed_site_metal_labels=None,
            esm_dim=config["esm_dim"], embeddings_dir=mapped["esm_embeddings_dir"],
            require_esm_embeddings=config["require_esm_embeddings"], ring_features_dir=mapped["ring_features_dir"],
            feature_root_dir=mapped["external_features_root_dir"],
            external_feature_source=config["external_feature_source"],
            require_external_features=config["require_external_features"],
            unsupported_metal_policy=config["unsupported_metal_policy"], ec_label_depth=config["ec_label_depth"],
            metal_example_unit="ion", cohort_bindings=binding_payload,
        )
        pockets.extend(loaded)
    saved_membership = verify_membership(checkpoint["dataset_summary"])["validation"]["examples"]
    by_example = {(pocket.structure_id, pocket.pocket_id): pocket for pocket in pockets}
    keys = [(row["structure_id"], row["pocket_id"]) for row in saved_membership]
    if len(by_example) != len(pockets) or set(keys) != set(by_example):
        raise CampaignContractError("Reloaded validation examples differ from saved membership")
    pockets = [by_example[key] for key in keys]
    for pocket in pockets:
        row = membership[pocket.metadata["source_uid"]]
        if (row["physical_ion_id"] != pocket.metadata["physical_ion_id"]
                or row["group_id"] != pocket.metadata["cohort_group_id"]
                or row["native_element"] != pocket.metal_element):
            raise CampaignContractError("Reloaded physical ion identity differs from frozen membership")
    graphs = build_graph_data_list(pockets, **graph_options)
    loader = DataLoader(PocketGraphDataset(pockets, precomputed_data=graphs,
                        normalization_stats=normalization, **graph_options),
                        batch_size=config["batch_size"], shuffle=False, num_workers=0,
                        generator=torch.Generator().manual_seed(int(config["seed"]) + 2))
    output_dir = Path(output_dir) if output_dir is not None else run_dir / "independent_validation_replay"
    output_dir.mkdir(parents=True, exist_ok=False)
    receipt = export_selected_validation_predictions(
        model=model, val_loader=loader, val_pockets=pockets, best_checkpoint=checkpoint,
        checkpoint_path=checkpoint_path, history=payload["history"], config_payload=config,
        run_dir=output_dir, device=device,
    )
    with saved_prediction_path.open(encoding="utf-8", newline="") as handle:
        saved = {row["source_uid"]: row for row in csv.DictReader(handle)}
    with (output_dir / "val_predictions.csv").open(encoding="utf-8", newline="") as handle:
        replayed = {row["source_uid"]: row for row in csv.DictReader(handle)}
    if saved.keys() != replayed.keys():
        raise CampaignContractError("Independent replay UID set differs from saved predictions")
    for uid, row in replayed.items():
        for key, value in row.items():
            if key.startswith("p_"):
                matches = abs(float(value) - float(saved[uid][key])) <= 1e-6
            else:
                matches = value == saved[uid].get(key)
            if not matches:
                raise CampaignContractError(f"Independent replay differs at {uid}/{key}")
    receipt.update(independent_replay=True, prediction_rows_verified=True, source_run_dir=str(run_dir),
                   normalization_refitted=False, training_performed=False)
    (output_dir / "replay_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt
