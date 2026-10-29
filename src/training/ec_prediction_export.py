"""EC-class prediction export, independent replay and the label-free test pass (contract C3).

In-run export (``--export-ec-predictions``)
  For ``--task ec --ec-class-vocabulary fixed_seven --checkpoint-rule terminal`` runs. After
  training, ``run_training`` reloads the terminal checkpoint strictly and predicts one of two
  pocket sets:

  - validation mode (EC fold membership split): every validation pocket. Files
    ``ec_val_pocket_predictions.csv`` and ``ec_val_protein_predictions.csv``;
  - probe mode (refits without validation, ``--val-fraction 0``): every pocket of a fixed list
    of training structures given by ``--ec-probe-structure-ids FILE`` (one structure ID per
    line; ``--ec-probe-structure-ids-sha256`` must match the file). The predictions exist only
    so that an independent replay can show the refit reproduces; there are no metrics. Files
    ``ec_probe_pocket_predictions.csv`` and ``ec_probe_protein_predictions.csv``.

  Pocket rows follow the loader order. Protein rows are the trainer's EC metric groups (a
  protein under ``--ec-group-weighting structure_id``), sorted by group key; their logits are
  the mean of the group's pocket logits, exactly as ``ec_group_metrics_from_logits`` takes it.
  Groups whose pockets disagree on the label get no protein row and are counted. Logits are in
  the fixed class order 1..7 and written with ``repr`` (full float precision); ``y`` and
  ``pred`` are class indices 0..6. Every file is written once (mode ``x``).

  ``ec_prediction_receipt.json`` records the run, fold, seed, model family, vocabulary,
  checkpoint, run configuration, membership or probe file and normalization hashes, the
  training protein counts per class (for logit-adjustment priors), the planned and realised
  optimizer steps of ``optimizer_groups.json``, the selected-site pocket grouping when it is
  not the default ``cluster`` (under ``structure`` every protein has one pocket, so its protein
  row holds that pocket's logits) and, in validation mode, a reconciliation of
  every ``val_ec_*`` metric of the terminal history record recomputed from the written CSVs
  with ``training.ec_class_metrics``: floats within 1e-9, counts and confusion matrices exact.
  The trainer takes accuracy as a float32 mean, so accuracy is recomputed as the float32
  rounding of the exact quotient. A mismatch raises after the files are written, so they stay
  for inspection.

Replay (run it in a fresh process)
  ``PYTHONPATH=src python -m training.ec_prediction_export replay --run-dir RUN``
  rebuilds the model from the checkpoint (frozen factory call plus the recorded EC auxiliary
  attachments, strict state-dict load), reloads only the exported structures with the run's
  recorded settings (membership or probe list, site extraction and pocket grouping, whole-protein vectors and
  their standardization, normalization statistics; no parse or graph cache, serial parsing),
  predicts again and compares with the saved CSVs: identities and labels exactly, logits
  within 1e-6, predicted classes exactly unless the saved top-two logit margin is at most
  twice the logit tolerance (a near tie, counted in the replay receipt; outside a near tie a
  changed class already implies a logit change beyond the tolerance). It refuses a run whose
  export receipt did not reconcile. Before any scientific input is opened it installs the
  held-out guard of ``DEEPMZYME_FORBIDDEN_READ_ROOTS`` and a recorded-input guard: for the
  rest of the process, files are read only from the run's recorded inputs, the run and output
  directories, the Python installation, this source tree and system directories, and written
  only in the output directory. Relative input paths in ``run_config.json`` resolve against
  the replay's working directory, so start it where the run was started. Outputs go to
  ``RUN/ec_prediction_replay`` unless ``--output-dir`` is given.

Label-free test pass (``export_ec_test_predictions``; decisions D7 and D12)
  Placeholder convention (D7): a label-blind test structure is named
  ``<accession>__chain_<chain>__EC_0.0.0.0``. The placeholder keeps the structure-ID grammar
  without carrying a label: ``0`` is not an EC class, so the strict class parser would refuse
  it, and the pass bypasses that parser and never assigns a target. Its selected sites come
  from an accession-keyed site table with exactly the columns ``accession``, ``chain_resi`` and
  ``metaltype`` (no EC column is accepted). Structures load with ``site_extraction=
  selected_sites`` and the checkpoint's ``--selected-site-pocket-grouping`` (recorded in the
  receipt when it is not ``cluster``), so only checkpoints trained with ``--site-extraction
  selected_sites`` are accepted; per-protein logits are keyed by accession.

  Enforcement (D12) cannot rest on who calls the function, so the pass refuses unless the
  Freeze-2 manifest file hashes to the expected SHA-256, lists the checkpoint, pins the live
  code of the prediction path, and names the output root; the pass ID was never used there
  (``ec_test_pass_ledger.jsonl``); and its outputs do not exist yet. The pass directory is
  created atomically before the claim is logged, and every output is written once (mode
  ``x``). The Freeze-2 manifest this pass reads is JSON with:

  - ``schema_version`` 1;
  - ``checkpoint_sha256s``: list of checkpoint SHA-256s;
  - ``code_sha256``: repository path (``src/...py``) -> SHA-256. It must cover every file of
    ``FREEZE2_REQUIRED_CODE_PATHS`` (globs expanded on the live tree), and every listed file
    must hash to its entry in the live tree;
  - ``test_pass_output_root``: absolute path of the one folder that holds the pass ledger
    and pass outputs, so a pass ID cannot be reused by pointing at another folder.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import site
import sys
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from data_structures import PocketRecord
from export_validation_predictions import factory_kwargs, fingerprint, install_read_only_guard
from graph.construction import pocket_to_pyg_data
from label_schemes import METAL_TARGET_LABELS, configure_active_metal_label_scheme
from model_variants import build_pocket_classifier
from model_variants.factory import MODEL_ARCHITECTURE_ALIASES
from training.campaign_runtime import RECONCILIATION_TOLERANCE, CampaignContractError
from training.config import DEFAULT_SELECTED_SITE_POCKET_GROUPING, TrainConfig, required_targets_for_task
from training.ec_class_metrics import N_EC_CLASSES, ec_class_metrics_from_logits
from training.ec_classes import (
    EC_CLASS_TOKEN_TO_INDEX, EC_CLASS_TOKENS, STRUCTURE_ID_RE, accession_from_structure_id, complete_ec_prefix,
    ec_class_split_support,
)
from training.graph_dataset import PocketGraphDataset
from training.loop import predict_logits
from training.site_filter import parse_summary_chain_resi
from training.splits import assign_ec_group_metadata


SRC_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SRC_ROOT.parent
EXPORT_SOURCE_KEY = "src/training/ec_prediction_export.py"
EC_PREDICTION_SCHEMA_VERSION = 1

TERMINAL_CHECKPOINT_NAME = "terminal_model_checkpoint.pt"
EC_PREDICTION_RECEIPT_NAME = "ec_prediction_receipt.json"
EC_REPLAY_RECEIPT_NAME = "ec_prediction_replay_receipt.json"
DEFAULT_REPLAY_DIRNAME = "ec_prediction_replay"
MODE_VALIDATION = "validation"
MODE_PROBE = "probe"
PREDICTION_FILE_NAMES = {
    MODE_VALIDATION: ("ec_val_pocket_predictions.csv", "ec_val_protein_predictions.csv"),
    MODE_PROBE: ("ec_probe_pocket_predictions.csv", "ec_probe_protein_predictions.csv"),
}
AGGREGATION = "mean_pocket_logits"

LOGIT_COLUMNS = tuple(f"logit_{token}" for token in EC_CLASS_TOKENS)
IDENTITY_COLUMNS = ("run_id", "mode", "fold_index", "n_folds", "seed")
EC_POCKET_PREDICTION_COLUMNS = IDENTITY_COLUMNS + (
    "accession", "structure_id", "ec_group_key", "pocket_id", "site_ids", "y", "pred",
) + LOGIT_COLUMNS
EC_PROTEIN_PREDICTION_COLUMNS = IDENTITY_COLUMNS + (
    "accession", "ec_group_key", "structure_ids", "n_pockets", "y", "pred",
) + LOGIT_COLUMNS
POCKET_KEY_COLUMNS = ("structure_id", "pocket_id")
PROTEIN_KEY_COLUMNS = ("ec_group_key",)

REPLAY_LOGIT_ATOL = 1.0e-6
MAX_REPLAY_LOGIT_ATOL = 1.0e-5

TEST_PLACEHOLDER_EC = "0.0.0.0"
TEST_PLACEHOLDER_SUFFIX = f"__EC_{TEST_PLACEHOLDER_EC}"
TEST_SITE_TABLE_COLUMNS = ("accession", "chain_resi", "metaltype")
TEST_PROTEIN_PREDICTION_COLUMNS = ("pass_id", "accession", "n_pockets", "pred") + LOGIT_COLUMNS
TEST_PROTEIN_PREDICTIONS_NAME = "ec_test_protein_predictions.csv"
TEST_RECEIPT_NAME = "ec_test_prediction_receipt.json"
TEST_PASS_LEDGER_NAME = "ec_test_pass_ledger.jsonl"
FREEZE2_SCHEMA_VERSION = 1
# Code the test pass executes to rebuild the model, build pockets and graphs, and predict.
FREEZE2_REQUIRED_CODE_PATHS = (
    EXPORT_SOURCE_KEY,
    "src/export_validation_predictions.py",
    "src/training/run.py",
    "src/training/config.py",
    "src/training/loop.py",
    "src/training/graph_dataset.py",
    "src/training/structure_loading.py",
    "src/training/site_filter.py",
    "src/training/feature_paths.py",
    "src/training/feature_sources.py",
    "src/training/esm_feature_loading.py",
    "src/training/external_feature_loading.py",
    "src/training/global_esm_features.py",
    "src/training/ec_classes.py",
    "src/training/labels.py",
    "src/ec_auxiliary_objectives.py",
    "src/model.py",
    "src/metal_objectives.py",
    "src/featurization.py",
    "src/data_structures.py",
    "src/label_schemes.py",
    "src/structure_store.py",
    "src/graph/*.py",
    "src/model_variants/*.py",
)
_PASS_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

_SYSTEM_READ_ROOTS = ("/proc", "/sys", "/dev", "/etc", "/usr", "/lib", "/lib64")
_LISTING_EVENTS = frozenset({"os.listdir", "os.scandir", "glob.glob", "glob.glob/2"})


class EcPredictionExportError(CampaignContractError):
    """An EC prediction export, replay or test pass broke its contract."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise EcPredictionExportError(message)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def ec_class_vocabulary_sha256() -> str:
    """Identity of the fixed seven-class vocabulary (tokens in index order)."""
    return _canonical_sha256({"vocabulary": "fixed_seven", "tokens": list(EC_CLASS_TOKENS)})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonable(value: Any) -> Any:
    from training.run import to_jsonable

    return to_jsonable(value)


def _optional_path(value: Any) -> Path | None:
    """A saved path value; config payloads store a missing summary as the string 'None'."""
    if value is None or str(value) in {"", "None"}:
        return None
    return Path(str(value))


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> str:
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(dict(payload)), indent=2, sort_keys=True) + "\n")
    return _sha256_file(path)


def _write_csv_once(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        handle.write(buffer.getvalue())
    return {"path": Path(path).name, "sha256": _sha256_file(path), "n_rows": len(rows)}


def read_prediction_csv(path: Path, columns: Sequence[str]) -> list[dict[str, str]]:
    """Rows of an exported prediction CSV; the header must equal ``columns``."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require(list(reader.fieldnames or []) == list(columns), f"{Path(path).name} has unexpected columns")
        return list(reader)


def _format_logit(value: float) -> str:
    return repr(float(value))


def _logit_matrix(rows: Sequence[Mapping[str, str]]) -> np.ndarray:
    return np.array([[float(row[column]) for column in LOGIT_COLUMNS] for row in rows], dtype=np.float64)


def _targets(rows: Sequence[Mapping[str, str]]) -> np.ndarray:
    return np.array([int(row["y"]) for row in rows], dtype=np.int64)


def _site_ids(pocket: PocketRecord) -> str:
    selected = pocket.metadata.get("selected_site_keys")
    if selected:
        return ";".join(f"{chain}_{resseq}" for chain, resseq in selected)
    return ";".join(
        f"{str(chain).strip()}_{str(resseq).strip()}{str(icode).strip()}"
        for chain, resseq, icode in pocket.metadata.get("metal_site_ids", [])
    )


def _pocket_grouping(config: Mapping[str, Any]) -> str:
    """A run's selected-site pocket grouping (configurations from before the option hold the default)."""
    return config.get("selected_site_pocket_grouping", DEFAULT_SELECTED_SITE_POCKET_GROUPING)


def _pocket_grouping_record(config: Mapping[str, Any]) -> dict[str, str]:
    """A run's selected-site pocket grouping, recorded only when it is not the default ``cluster``."""
    grouping = _pocket_grouping(config)
    return {} if grouping == DEFAULT_SELECTED_SITE_POCKET_GROUPING else {"selected_site_pocket_grouping": grouping}


def _graph_options(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "esm_dim": int(config["esm_dim"]),
        "edge_radius": float(config["edge_radius"]),
        "use_ring_edges": bool(config["use_ring_edges"]),
        "require_ring_edges": bool(config["require_ring_edges"]),
        "node_feature_set": config["node_feature_set"],
        "omit_node_features": tuple(config["omit_node_features"]),
        "metal_node_mode": config["metal_node_mode"],
        "shell_role_source": config["shell_role_source"],
    }


def _prediction_loader(pockets: list[PocketRecord], config: Mapping[str, Any], normalization_stats) -> DataLoader:
    """Unshuffled loader over freshly built graphs (no graph cache), as an evaluation loader sees them."""
    options = _graph_options(config)
    graphs = [pocket_to_pyg_data(pocket, **options) for pocket in pockets]
    dataset = PocketGraphDataset(pockets, precomputed_data=graphs, normalization_stats=normalization_stats, **options)
    return DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=0)


def campaign_family(config: Mapping[str, Any]) -> str | None:
    """The campaign family label (F1-F5, C-G) of a saved configuration, or None for other models."""
    architecture = MODEL_ARCHITECTURE_ALIASES.get(str(config.get("model_architecture")), config.get("model_architecture"))
    with_global = bool(config.get("use_global_esm_branch", False))
    if architecture == "only_global":
        return "C-G"
    if architecture == "only_gvp":
        return None if with_global else "F1"
    if architecture == "only_esm":
        return "F5" if with_global else "F2"
    if (architecture == "gvp" and config.get("fusion_mode") == "late_fusion" and config.get("use_esm_branch", True)
            and not config.get("use_early_esm", False)):
        return "F4" if with_global else "F3"
    return None


def _model_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_architecture": config.get("model_architecture"),
        "fusion_mode": config.get("fusion_mode"),
        "use_global_esm_branch": bool(config.get("use_global_esm_branch", False)),
        "global_esm_layer": config.get("global_esm_layer") if config.get("use_global_esm_branch") else None,
        "ec_aux_levels": config.get("ec_aux_levels", ""),
        "ec_supcon_weight": config.get("ec_supcon_weight", 0.0),
        "campaign_family": campaign_family(config),
    }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _uses_ec_fold_membership(config: TrainConfig) -> bool:
    return config.fold_split_source == "membership" and config.task == "ec" and config.source_cohort_csv is None


def read_probe_structure_ids(path: Path, expected_sha256: str) -> tuple[str, ...]:
    """Probe structure IDs (sorted) from a one-ID-per-line file whose bytes must hash to ``expected_sha256``."""
    from training.global_esm_features import GlobalEsmCacheError, read_id_allowlist

    try:
        return read_id_allowlist(Path(path), expected_sha256=expected_sha256).ids
    except (GlobalEsmCacheError, OSError) as exc:
        raise EcPredictionExportError(f"Probe structure list {path}: {exc}") from None


def validate_ec_export_configuration(config: TrainConfig) -> str | None:
    """Refuse an EC prediction export the run cannot certify; returns the export mode (None when off)."""
    probe = config.ec_probe_structure_ids is not None
    if not config.export_ec_predictions:
        if probe or config.ec_probe_structure_ids_sha256 is not None:
            raise ValueError("--ec-probe-structure-ids requires --export-ec-predictions")
        return None
    if config.task != "ec" or config.ec_class_vocabulary != "fixed_seven" or config.checkpoint_rule != "terminal":
        raise ValueError(
            "--export-ec-predictions requires --task ec, --ec-class-vocabulary fixed_seven and --checkpoint-rule terminal"
        )
    if config.export_validation_predictions:
        raise ValueError("--export-ec-predictions cannot be combined with --export-validation-predictions")
    if config.run_test_eval or config.test_structure_dir is not None or config.test_summary_csv is not None:
        raise ValueError("--export-ec-predictions cannot be combined with a held-out test set")
    if config.explicit_membership_manifest or config.source_cohort_csv is not None:
        raise ValueError("--export-ec-predictions needs an EC fold membership split or a probe list, not another split")
    membership = _uses_ec_fold_membership(config)
    if membership == probe:
        raise ValueError(
            "--export-ec-predictions needs exactly one of an EC fold membership split (--fold-split-source "
            "membership) or --ec-probe-structure-ids for a run without validation"
        )
    if probe:
        if config.val_fraction > 0.0 or config.n_folds is not None:
            raise ValueError("--ec-probe-structure-ids is for runs without validation (--val-fraction 0, no folds)")
        if not config.ec_probe_structure_ids_sha256:
            raise ValueError("--ec-probe-structure-ids requires --ec-probe-structure-ids-sha256")
        read_probe_structure_ids(Path(config.ec_probe_structure_ids), str(config.ec_probe_structure_ids_sha256))
        return MODE_PROBE
    return MODE_VALIDATION


# ---------------------------------------------------------------------------
# Model rebuild
# ---------------------------------------------------------------------------

def replay_model_attachments(model: torch.nn.Module, attachments: Mapping[str, Any]) -> dict[str, Any]:
    """Re-attach the EC auxiliary modules recorded in ``dataset_summary['ec_auxiliary_objectives']``."""
    from ec_auxiliary_objectives import attach_ec_auxiliary_objectives

    levels = {int(level): int(size) for level, size in (attachments.get("ec_aux_levels") or {}).items()}
    supcon = attachments.get("ec_supcon") or None
    return attach_ec_auxiliary_objectives(
        model,
        aux_level_sizes=levels,
        aux_weight=float(attachments.get("ec_aux_weight", 0.0)),
        supcon_weight=float(supcon["weight"]) if supcon else 0.0,
        supcon_temperature=float(supcon["temperature"]) if supcon else 0.1,
        supcon_memory=int(supcon["memory"]) if supcon else 256,
        supcon_projection_dim=int(supcon["projection_dim"]) if supcon else 128,
    )


def rebuild_model_from_checkpoint(source: Path, *, device: str = "cpu") -> tuple[torch.nn.Module, dict[str, Any]]:
    """Model and checkpoint from a run directory (its terminal checkpoint) or a checkpoint file.

    The model comes from the frozen factory call of ``training/run.py`` (read, never run)
    plus the recorded EC auxiliary attachments, so the saved state dict loads strictly.
    """
    path = Path(source)
    if path.is_dir():
        path = path / TERMINAL_CHECKPOINT_NAME
    _require(path.is_file(), f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model_state_dict", "config", "ec_labels", "metal_labels"}
    _require(isinstance(checkpoint, dict) and required <= set(checkpoint), f"{path} is not a training checkpoint")
    config = checkpoint["config"]
    configure_active_metal_label_scheme(config["metal_label_scheme"])
    metal_labels = {int(key): str(value) for key, value in checkpoint["metal_labels"].items()}
    ec_labels = {int(key): str(value) for key, value in checkpoint["ec_labels"].items()}
    _require(metal_labels == METAL_TARGET_LABELS, "Checkpoint metal vocabulary differs from its saved scheme")
    kwargs = factory_kwargs(Path(__file__).with_name("run.py"), config, checkpoint, metal_labels, ec_labels)
    model = build_pocket_classifier(**kwargs)
    attachments = ((checkpoint.get("dataset_summary") or {}).get("ec_auxiliary_objectives") or {}).get("attachments")
    if attachments:
        replay_model_attachments(model, attachments)
    model = model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def _require_fixed_seven_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    config = checkpoint["config"]
    _require(config.get("task") == "ec" and config.get("ec_class_vocabulary") == "fixed_seven",
             "EC prediction export needs a --task ec fixed_seven checkpoint")
    _require({int(k): str(v) for k, v in checkpoint["ec_labels"].items()}
             == {index: token for token, index in EC_CLASS_TOKEN_TO_INDEX.items()},
             "Checkpoint EC labels are not the fixed seven-class vocabulary")
    _require(checkpoint.get("checkpoint_rule") == "terminal", "EC prediction export needs the terminal checkpoint")


# ---------------------------------------------------------------------------
# Prediction rows
# ---------------------------------------------------------------------------

def _predict(model: torch.nn.Module, loader: DataLoader, pockets: Sequence[PocketRecord], device: str) -> dict[str, Any]:
    predictions = predict_logits(model, loader, device=device)
    _require("ec_logits" in predictions, "The model returned no EC logits")
    logits = predictions["ec_logits"].float()
    _require(tuple(logits.shape) == (len(pockets), N_EC_CLASSES),
             f"Expected {len(pockets)} x {N_EC_CLASSES} EC logits, got {tuple(logits.shape)}")
    _require(bool(torch.isfinite(logits).all()), "Prediction produced non-finite EC logits")
    return {**predictions, "ec_logits": logits}


def build_prediction_rows(
    pockets: Sequence[PocketRecord],
    logits: torch.Tensor,
    identity: Mapping[str, str],
) -> tuple[list[dict[str, str]], list[dict[str, str]], int]:
    """Pocket rows (loader order), protein rows (sorted EC group keys) and the number of conflicting groups.

    A protein row holds the float32 mean of its pocket logits in loader order, the
    aggregation of ``training.run.ec_group_metrics_from_logits``.
    """
    pocket_rows: list[dict[str, str]] = []
    members: dict[str, list[int]] = {}
    for index, pocket in enumerate(pockets):
        group_key = pocket.metadata.get("ec_group_key")
        _require(group_key is not None and pocket.y_ec is not None, f"Pocket {pocket.pocket_id} has no EC group or label")
        pocket_rows.append({
            **identity,
            "accession": accession_from_structure_id(pocket.structure_id),
            "structure_id": pocket.structure_id,
            "ec_group_key": str(group_key),
            "pocket_id": pocket.pocket_id,
            "site_ids": _site_ids(pocket),
            "y": str(int(pocket.y_ec)),
            "pred": str(int(logits[index].argmax())),
            **{column: _format_logit(value) for column, value in zip(LOGIT_COLUMNS, logits[index].tolist())},
        })
        members.setdefault(str(group_key), []).append(index)
    protein_rows: list[dict[str, str]] = []
    conflicting = 0
    for group_key in sorted(members):
        indices = members[group_key]
        targets = {int(pockets[index].y_ec) for index in indices}
        if len(targets) != 1:
            conflicting += 1
            continue
        accessions = sorted({pocket_rows[index]["accession"] for index in indices})
        _require(len(accessions) == 1, f"EC group {group_key!r} spans accessions {accessions}")
        mean = torch.stack([logits[index] for index in indices], dim=0).mean(dim=0)
        protein_rows.append({
            **identity,
            "accession": accessions[0],
            "ec_group_key": group_key,
            "structure_ids": ";".join(sorted({pockets[index].structure_id for index in indices})),
            "n_pockets": str(len(indices)),
            "y": str(targets.pop()),
            "pred": str(int(mean.argmax())),
            **{column: _format_logit(value) for column, value in zip(LOGIT_COLUMNS, mean.tolist())},
        })
    return pocket_rows, protein_rows, conflicting


def _identity(run_id: str, mode: str, config: Mapping[str, Any]) -> dict[str, str]:
    fold_index = config.get("fold_index") if mode == MODE_VALIDATION else None
    n_folds = config.get("n_folds") if mode == MODE_VALIDATION else None
    return {"run_id": run_id, "mode": mode, "fold_index": "" if fold_index is None else str(int(fold_index)),
            "n_folds": "" if n_folds is None else str(int(n_folds)), "seed": str(int(config["seed"]))}


# ---------------------------------------------------------------------------
# Metric reconciliation
# ---------------------------------------------------------------------------

_FIXED_CLASS_KEYS = {
    "ec_classes_1_6_balanced_acc": "ec_classes_1_6_balanced_accuracy",
    "ec_classes_1_6_n_present": "ec_classes_1_6_n_present",
    "ec_classes_1_6_macro_f1": "ec_classes_1_6_macro_f1",
    "seven_class_balanced_acc": "seven_class_balanced_accuracy",
    "seven_class_n_present": "seven_class_n_present",
    "seven_class_macro_f1": "seven_class_macro_f1",
}


def _trainer_metric_view(metrics: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    """The trainer's legacy metric keys from ``ec_class_metrics`` output (depth 1, fixed seven classes).

    Over the seven classes the legacy balanced accuracy and macro-F1 average the classes
    with support, which are the seven-class values; the level-1 metrics equal them because a
    class token is its own first-level prefix.
    """
    support = metrics["per_class_support"]
    recall = metrics["per_class_recall"]
    accuracy = float(np.float32(metrics["accuracy"]))
    return {
        f"{prefix}_acc": accuracy,
        f"{prefix}_balanced_acc": metrics["seven_class_balanced_accuracy"],
        f"{prefix}_macro_f1": metrics["seven_class_macro_f1"],
        f"{prefix}_per_class_recall": {token: (float(recall[index]) if support[index] > 0 else None)
                                       for index, token in enumerate(EC_CLASS_TOKENS)},
        f"{prefix}_per_class_support": {token: int(round(support[index])) for index, token in enumerate(EC_CLASS_TOKENS)},
        f"{prefix}_level_1_acc": accuracy,
        f"{prefix}_level_1_balanced_acc": metrics["seven_class_balanced_accuracy"],
        f"{prefix}_level_1_macro_f1": metrics["seven_class_macro_f1"],
    }


def count_conflicting_groups(pocket_rows: Sequence[Mapping[str, str]]) -> int:
    labels: dict[str, set[str]] = {}
    for row in pocket_rows:
        labels.setdefault(row["ec_group_key"], set()).add(row["y"])
    return sum(1 for values in labels.values() if len(values) > 1)


def recompute_ec_validation_metrics(
    pocket_rows: Sequence[Mapping[str, str]],
    protein_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """Every ``val_ec_*`` key the trainer records for a fixed seven-class EC run, from exported rows."""
    _require(bool(pocket_rows), "No pocket rows to recompute metrics from")
    pocket_metrics = ec_class_metrics_from_logits(_logit_matrix(pocket_rows), _targets(pocket_rows))
    expected = _trainer_metric_view(pocket_metrics, "val_ec")
    expected["val_ec_group_n_groups"] = len(protein_rows)
    expected["val_ec_group_n_conflicting_groups"] = count_conflicting_groups(pocket_rows)
    if not protein_rows:
        return expected
    group = ec_class_metrics_from_logits(_logit_matrix(protein_rows), _targets(protein_rows))
    expected.update(_trainer_metric_view(group, "val_ec_group"))
    present = [index for index, value in enumerate(group["per_class_support"]) if value > 0]
    expected["val_ec_group_min_recall"] = min(float(group["per_class_recall"][index]) for index in present)
    expected["val_ec_group_confusion_matrix"] = [[int(round(value)) for value in row] for row in group["confusion_matrix"]]
    for trainer_key, metric_key in _FIXED_CLASS_KEYS.items():
        expected[f"val_ec_group_{trainer_key}"] = group[metric_key]
    return expected


def _metric_matches(recorded: Any, recomputed: Any, tolerance: float) -> tuple[bool, float]:
    """(match, largest absolute difference); integers and booleans compare exactly, floats within ``tolerance``."""
    if isinstance(recorded, Mapping) and isinstance(recomputed, Mapping):
        left = {str(key): value for key, value in recorded.items()}
        right = {str(key): value for key, value in recomputed.items()}
        if left.keys() != right.keys():
            return False, math.inf
        results = [_metric_matches(left[key], right[key], tolerance) for key in left]
        return all(ok for ok, _ in results), max((diff for _, diff in results), default=0.0)
    if isinstance(recorded, (list, tuple)) and isinstance(recomputed, (list, tuple)):
        if len(recorded) != len(recomputed):
            return False, math.inf
        results = [_metric_matches(a, b, tolerance) for a, b in zip(recorded, recomputed)]
        return all(ok for ok, _ in results), max((diff for _, diff in results), default=0.0)
    if recorded is None or recomputed is None:
        return recorded is None and recomputed is None, 0.0
    if isinstance(recorded, bool) or isinstance(recomputed, bool):
        return recorded == recomputed, 0.0
    if isinstance(recorded, (int, np.integer)) and isinstance(recomputed, (int, np.integer)):
        return int(recorded) == int(recomputed), float(abs(int(recorded) - int(recomputed)))
    if isinstance(recorded, (int, float, np.floating)) and isinstance(recomputed, (int, float, np.floating)):
        left, right = float(recorded), float(recomputed)
        if math.isnan(left) and math.isnan(right):
            return True, 0.0
        if not (math.isfinite(left) and math.isfinite(right)):
            return left == right, 0.0 if left == right else math.inf
        difference = abs(left - right)
        return difference <= tolerance, difference
    return recorded == recomputed, 0.0


def reconcile_ec_validation_metrics(
    record: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    tolerance: float = RECONCILIATION_TOLERANCE,
) -> tuple[dict[str, Any], str]:
    """Compare every ``val_ec_*`` key of a history record with its recomputation; returns (details, status)."""
    details: dict[str, Any] = {}
    status = "match"
    for key in sorted({key for key in record if key.startswith("val_ec_")} | set(expected)):
        if key not in record:
            entry = {"recorded": None, "recomputed": expected[key], "match": False,
                     "reason": "absent from the terminal history record"}
        elif key not in expected:
            entry = {"recorded": record[key], "recomputed": None, "match": False,
                     "reason": "not recomputed by the EC prediction export"}
        else:
            match, difference = _metric_matches(record[key], expected[key], tolerance)
            entry = {"recorded": record[key], "recomputed": expected[key], "match": match,
                     "max_abs_difference": difference}
        details[key] = entry
        if not entry["match"]:
            status = "mismatch"
    return details, status


# ---------------------------------------------------------------------------
# In-run export
# ---------------------------------------------------------------------------

def _probe_pockets_and_loader(prepared, config: TrainConfig, probe_ids: Sequence[str]):
    eval_dataset = prepared.train_eval_loader.dataset
    wanted = set(probe_ids)
    indices = [index for index, pocket in enumerate(eval_dataset.pockets) if pocket.structure_id in wanted]
    found = {eval_dataset.pockets[index].structure_id for index in indices}
    missing = sorted(wanted - found)
    _require(not missing, f"{len(missing)} probe structure(s) are not training structures of this run, e.g. {missing[:3]}")
    pockets = [eval_dataset.pockets[index] for index in indices]
    dataset = PocketGraphDataset(
        pockets, esm_dim=config.esm_dim, edge_radius=config.edge_radius,
        normalization_stats=prepared.normalization_stats, use_ring_edges=config.use_ring_edges,
        require_ring_edges=config.require_ring_edges, shell_role_source=config.shell_role_source,
        precomputed_data=[eval_dataset.precomputed_data[index] for index in indices],
        node_feature_set=config.node_feature_set, omit_node_features=config.omit_node_features,
        metal_node_mode=config.metal_node_mode,
    )
    return pockets, DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)


def export_ec_run_predictions(
    prepared,
    config: TrainConfig,
    *,
    history: list[dict[str, Any]],
    checkpoint: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Write the C3 prediction files and receipt of a finished run; raises after writing on a mismatch."""
    from training.ec_membership import read_ec_fold_membership
    from training.run import normalization_stats_payload

    mode = validate_ec_export_configuration(config)
    _require(mode is not None, "EC prediction export is not enabled for this run")
    run_dir = Path(prepared.run_dir)
    checkpoint_path = run_dir / TERMINAL_CHECKPOINT_NAME
    _require(checkpoint is not None and checkpoint.get("checkpoint_rule") == "terminal" and checkpoint_path.is_file(),
             "EC prediction export needs the saved terminal checkpoint")
    _require(bool(history) and int(checkpoint["epoch"]) == len(history) == int(history[-1]["epoch"]),
             "The terminal checkpoint is not the last epoch of the history")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _require_fixed_seven_checkpoint(saved)
    _require(int(saved["epoch"]) == int(checkpoint["epoch"]),
             f"{TERMINAL_CHECKPOINT_NAME} holds epoch {saved['epoch']}, not the terminal epoch {checkpoint['epoch']}")
    prepared.model.load_state_dict(saved["model_state_dict"], strict=True)
    config_payload = prepared.config_payload
    identity = _identity(run_dir.name, mode, config_payload)

    membership_record = probe_record = None
    if mode == MODE_VALIDATION:
        _require(prepared.val_loader is not None and bool(prepared.split.val_pockets), "No validation pockets to export")
        pockets, loader = list(prepared.split.val_pockets), prepared.val_loader
        membership = read_ec_fold_membership(Path(config.fold_membership_csv), str(config.fold_membership_sha256))
        expected_ids = {row["structure_id"] for row in membership.rows if int(row["fold"]) == int(config.fold_index)}
        _require({pocket.structure_id for pocket in pockets} == expected_ids,
                 "Exported validation structures differ from the membership's validation fold")
        membership_record = {"path": str(config.fold_membership_csv), "file_sha256": membership.file_sha256,
                             "identity_sha256": membership.identity_sha256, "n_validation_structures": len(expected_ids)}
    else:
        probe_ids = read_probe_structure_ids(Path(config.ec_probe_structure_ids),
                                             str(config.ec_probe_structure_ids_sha256))
        pockets, loader = _probe_pockets_and_loader(prepared, config, probe_ids)
        probe_record = {"path": str(config.ec_probe_structure_ids), "file_sha256": config.ec_probe_structure_ids_sha256,
                        "n_structures": len(probe_ids)}

    predictions = _predict(prepared.model, loader, pockets, config.device)
    if mode == MODE_VALIDATION and "ec_group_id" in predictions:
        pairs = {(int(group_id), pocket.metadata["ec_group_key"])
                 for group_id, pocket in zip(predictions["ec_group_id"].tolist(), pockets)}
        _require(len(pairs) == len({key for _, key in pairs}) == len({group_id for group_id, _ in pairs}),
                 "Graph EC group IDs do not map one-to-one to EC group keys")
    pocket_rows, protein_rows, conflicting = build_prediction_rows(pockets, predictions["ec_logits"], identity)
    pocket_name, protein_name = PREDICTION_FILE_NAMES[mode]
    pocket_file = _write_csv_once(run_dir / pocket_name, EC_POCKET_PREDICTION_COLUMNS, pocket_rows)
    protein_file = _write_csv_once(run_dir / protein_name, EC_PROTEIN_PREDICTION_COLUMNS, protein_rows)

    reconciliation: dict[str, Any] = {}
    status = "not_applicable"
    if mode == MODE_VALIDATION:
        expected = recompute_ec_validation_metrics(
            read_prediction_csv(run_dir / pocket_name, EC_POCKET_PREDICTION_COLUMNS),
            read_prediction_csv(run_dir / protein_name, EC_PROTEIN_PREDICTION_COLUMNS),
        )
        reconciliation, status = reconcile_ec_validation_metrics(history[-1], expected)

    optimizer_groups = json.loads((run_dir / "optimizer_groups.json").read_text(encoding="utf-8"))
    global_esm = prepared.dataset_summary.get("global_esm")
    receipt = {
        "schema_version": EC_PREDICTION_SCHEMA_VERSION,
        "mode": mode,
        "run_id": run_dir.name,
        "campaign_run_identity": (json.loads(config_payload["campaign_run_identity"])
                                  if config_payload.get("campaign_run_identity") else None),
        "fold_index": config_payload.get("fold_index") if mode == MODE_VALIDATION else None,
        "n_folds": config_payload.get("n_folds") if mode == MODE_VALIDATION else None,
        "seed": int(config.seed),
        "model": _model_identity(config_payload),
        "ec_class_tokens": list(EC_CLASS_TOKENS),
        "vocabulary_sha256": ec_class_vocabulary_sha256(),
        "aggregation": AGGREGATION,
        "checkpoint": {"path": TERMINAL_CHECKPOINT_NAME, "sha256": _sha256_file(checkpoint_path),
                       "epoch": int(checkpoint["epoch"]), "checkpoint_rule": "terminal"},
        "run_config_sha256": _sha256_file(run_dir / "run_config.json"),
        "membership": membership_record,
        "probe": probe_record,
        "structure_ids_sha256": _canonical_sha256(sorted({pocket.structure_id for pocket in pockets})),
        "normalization_sha256": fingerprint(_jsonable(normalization_stats_payload(prepared.normalization_stats))),
        "global_esm": None if global_esm is None else {
            "manifest_sha256": global_esm.get("manifest_sha256"), "layer": global_esm.get("layer"),
            "standardization": (global_esm.get("standardization") or {}).get("mode", "none"),
        },
        "training_protein_class_counts": ec_class_split_support(prepared.split.train_pockets, [])[
            "train_proteins_by_ec_class"],
        "epochs": {"planned": int(config.epochs), "completed": len(history)},
        "optimizer_steps": {key: optimizer_groups.get(key) for key in (
            "optimizer_steps_per_epoch", "planned_total_steps", "planned_steps_for_epochs_run",
            "realised_total_steps", "realised_steps_match")},
        "pocket_predictions": pocket_file,
        "protein_predictions": protein_file,
        "n_conflicting_groups": conflicting,
        "reconciliation_tolerance": RECONCILIATION_TOLERANCE,
        "reconciliation": reconciliation,
        "reconciliation_status": status,
        "export_source_sha256": _sha256_file(Path(__file__)),
        "created_utc": _utc_now(),
        **_pocket_grouping_record(config_payload),
    }
    _write_json_once(run_dir / EC_PREDICTION_RECEIPT_NAME, receipt)
    if status == "mismatch":
        failed = sorted(key for key, entry in reconciliation.items() if not entry["match"])
        raise EcPredictionExportError(
            f"EC predictions do not reconcile with the terminal validation metrics ({failed[:5]}); "
            f"files kept in {run_dir}"
        )
    return receipt


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def _validate_replay_config(config: Mapping[str, Any]) -> None:
    _require(config.get("task") == "ec" and config.get("ec_class_vocabulary") == "fixed_seven"
             and config.get("checkpoint_rule") == "terminal" and config.get("export_ec_predictions"),
             "Replay needs a fixed seven-class EC run with a terminal checkpoint and an EC prediction export")
    for name in ("run_test_eval", "allow_train_loss_test_eval_debug", "allow_final_refit_test_eval"):
        _require(not config.get(name), f"Replay refuses a run with {name}")
    for name in ("test_structure_dir", "test_summary_csv", "source_cohort_csv", "explicit_membership_manifest"):
        _require(_optional_path(config.get(name)) is None, f"Replay refuses a run with {name}")


def _normalize_both(path: str | os.PathLike) -> set[str]:
    text = os.fsdecode(path)
    return {os.path.normcase(os.path.abspath(text)), os.path.normcase(os.path.realpath(text))}


def _is_under(path: str, roots: Iterable[str]) -> bool:
    return any(path == root or path.startswith(root.rstrip(os.sep) + os.sep) for root in roots)


def _python_read_roots() -> list[str]:
    roots = {sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix, str(SRC_ROOT)}
    roots.update(getattr(site, "getsitepackages", lambda: [])())
    user_site = site.getusersitepackages() if site.ENABLE_USER_SITE else None
    if user_site and Path(user_site).is_dir():
        roots.add(user_site)
    return sorted(roots)


def install_recorded_input_guard(read_roots: Iterable[str | os.PathLike], *, output_dir: Path) -> tuple[list, set]:
    """Keep this process inside its recorded inputs for the rest of its life (audit hooks stay).

    Files may be read, listed or globbed only under ``read_roots``, ``output_dir``, the Python
    installation, this source tree and system directories. Writes and other filesystem
    changes are allowed only under ``output_dir``, and subprocesses and sockets are refused
    (``export_validation_predictions.install_read_only_guard``). Returns the data roots and the
    set that collects every file read under them.
    """
    output_dir = Path(output_dir).resolve()
    read_roots = [Path(root) for root in read_roots]
    data_roots, data_reads = install_read_only_guard(output_dir)
    data_roots.extend(root.resolve() for root in read_roots)
    allowed: set[str] = set()
    for root in [*read_roots, output_dir, *_python_read_roots(), *_SYSTEM_READ_ROOTS]:
        allowed |= _normalize_both(root)
    allowed_roots = sorted(allowed)

    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND

    def hook(event: str, args: tuple) -> None:
        if event == "open" and args:
            mode = str(args[1] or "") if len(args) > 1 else ""
            flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
            if any(token in mode for token in "wax+") or flags & write_flags:
                return  # writes are checked by install_read_only_guard
        elif event not in _LISTING_EVENTS or not args:
            return
        target = args[0]
        if not isinstance(target, (str, bytes, os.PathLike)):
            return
        if not any(_is_under(form, allowed_roots) for form in _normalize_both(target)):
            raise PermissionError(f"EC prediction replay may read only the run's recorded inputs: {os.fsdecode(target)}")

    sys.addaudithook(hook)
    return data_roots, data_reads


def _recorded_read_roots(config: Mapping[str, Any]) -> list[Path]:
    """Every input path the run recorded, resolved as its loader resolved them."""
    from graph.ring_edges import default_ring_output_dir
    from project_paths import resolve_ring_features_dir
    from training.feature_paths import resolve_runtime_feature_paths

    structure_dir = Path(config["structure_dir"])
    embeddings_dir, feature_root = resolve_runtime_feature_paths(
        structure_dir=structure_dir, esm_embeddings_dir=_optional_path(config.get("esm_embeddings_dir")),
        external_features_root_dir=_optional_path(config.get("external_features_root_dir")),
        external_feature_source=config["external_feature_source"],
    )
    roots = [structure_dir, embeddings_dir, feature_root]
    if config.get("use_ring_edges") or config.get("require_ring_edges"):
        # RING files are opened only when the graphs use RING edges.
        ring = _optional_path(config.get("ring_features_dir"))
        roots.append(resolve_ring_features_dir(str(ring), create=False) if ring is not None else default_ring_output_dir())
    for key in ("summary_csv", "fold_membership_csv", "ec_probe_structure_ids", "global_esm_cache_dir"):
        value = _optional_path(config.get(key))
        if value is not None:
            roots.append(value)
    return roots


def _replay_auxiliary_targets(
    pockets: Sequence[PocketRecord], vocabularies: Mapping[str, Any], *, training_side: bool,
) -> None:
    """Training-only EC level targets as the run set them (they never enter predictions)."""
    for level, vocabulary in vocabularies.items():
        index = {token: position for position, token in enumerate(vocabulary["tokens"])}
        for pocket in pockets:
            value = -1
            if training_side:
                numbers = pocket.metadata.get("ec_numbers") or ()
                numbers = numbers.split(";") if isinstance(numbers, str) else numbers
                prefix = complete_ec_prefix([str(number) for number in numbers], int(level))
                value = index.get(prefix, -1) if prefix is not None else -1
            pocket.metadata[f"ec_aux_l{level}"] = value


def _attach_recorded_global_vectors(pockets: Sequence[PocketRecord], config: Mapping[str, Any],
                                    global_summary: Mapping[str, Any]) -> None:
    from training.global_esm_features import GLOBAL_ESM_METADATA_KEY, attach_global_esm_to_pockets, load_global_esm_vectors

    structure_ids = sorted({pocket.structure_id for pocket in pockets})
    vectors = load_global_esm_vectors(Path(config["global_esm_cache_dir"]), str(config["global_esm_manifest_sha256"]),
                                      structure_ids, expected_layer=config["global_esm_layer"],
                                      expected_dim=int(config["global_esm_dim"]))
    attach_global_esm_to_pockets(pockets, vectors)
    standardization = global_summary.get("standardization")
    if standardization:
        mean = torch.tensor(standardization["mean"], dtype=torch.float32)
        std = torch.tensor(standardization["std"], dtype=torch.float32)
        for pocket in pockets:
            pocket.metadata[GLOBAL_ESM_METADATA_KEY] = (pocket.metadata[GLOBAL_ESM_METADATA_KEY] - mean) / std


def _reload_exported_pockets(config: Mapping[str, Any], checkpoint: Mapping[str, Any],
                             structure_ids: set[str], *, training_side: bool) -> list[PocketRecord]:
    """Load exactly ``structure_ids`` with the run's recorded settings and rebuild their run-time metadata."""
    from training.data import load_training_pockets_with_report_from_dir

    result = load_training_pockets_with_report_from_dir(
        structure_dir=Path(config["structure_dir"]),
        require_full_labels=True,
        required_targets=required_targets_for_task(config["task"]),
        summary_csv=_optional_path(config.get("summary_csv")),
        esm_dim=int(config["esm_dim"]),
        esm_embeddings_dir=_optional_path(config.get("esm_embeddings_dir")),
        require_esm_embeddings=bool(config["require_esm_embeddings"]),
        ring_features_dir=_optional_path(config.get("ring_features_dir")),
        external_features_root_dir=_optional_path(config.get("external_features_root_dir")),
        external_feature_source=config["external_feature_source"],
        require_external_features=bool(config["require_external_features"]),
        unsupported_metal_policy=config["unsupported_metal_policy"],
        invalid_structure_policy=config["invalid_structure_policy"],
        ec_label_depth=int(config["ec_label_depth"]),
        metal_eligibility_scheme=config["metal_eligibility_scheme"],
        metal_example_unit=config["metal_example_unit"],
        allowed_structure_ids=set(structure_ids),
        load_workers=1,
        ec_class_vocabulary=config["ec_class_vocabulary"],
        site_extraction=config.get("site_extraction", "component"),
        selected_site_pocket_grouping=_pocket_grouping(config),
    )
    pockets = list(result.pockets)
    assign_ec_group_metadata(pockets, weighting_mode=config["ec_group_weighting"])
    summary = checkpoint.get("dataset_summary") or {}
    vocabularies = (summary.get("ec_auxiliary_objectives") or {}).get("vocabularies") or {}
    if vocabularies:
        _replay_auxiliary_targets(pockets, vocabularies, training_side=training_side)
    if config.get("use_global_esm_branch"):
        _attach_recorded_global_vectors(pockets, config, summary.get("global_esm") or {})
    return pockets


def compare_prediction_rows(
    saved: Sequence[Mapping[str, str]],
    replayed: Sequence[Mapping[str, str]],
    *,
    key_columns: Sequence[str],
    logit_atol: float,
) -> tuple[float, list[str], int]:
    """(largest logit difference, problems, near-tie class changes) of saved against replayed rows.

    Rows must agree in order and in every field except the logits (within ``logit_atol``) and
    ``pred``: a changed predicted class counts as a near tie, not a problem, when the saved
    top-two margin is at most ``2 * logit_atol``. A wider margin cannot flip while every
    logit stays within the tolerance.
    """
    problems: list[str] = []
    near_ties = 0
    saved_keys = [tuple(row[column] for column in key_columns) for row in saved]
    replayed_keys = [tuple(row[column] for column in key_columns) for row in replayed]
    if saved_keys != replayed_keys:
        return math.inf, [f"row keys or order differ ({len(saved_keys)} saved, {len(replayed_keys)} replayed)"], 0
    largest = 0.0
    for key, left, right in zip(saved_keys, saved, replayed):
        if left.keys() != right.keys():
            problems.append(f"{key}: columns differ")
            continue
        for column, value in left.items():
            if column in LOGIT_COLUMNS:
                difference = abs(float(value) - float(right[column]))
                largest = max(largest, difference)
                if not difference <= logit_atol:
                    problems.append(f"{key}/{column}: logit differs by {difference:.3g}")
            elif value != right[column]:
                if column == "pred" and _top_two_margin(left) <= 2.0 * logit_atol:
                    near_ties += 1
                    continue
                problems.append(f"{key}/{column}: {value!r} != {right[column]!r}")
    return largest, problems, near_ties


def _top_two_margin(row: Mapping[str, str]) -> float:
    first, second = sorted((float(row[column]) for column in LOGIT_COLUMNS), reverse=True)[:2]
    return first - second


def replay_ec_predictions(
    run_dir: Path,
    *,
    output_dir: Path | None = None,
    device: str = "cpu",
    logit_atol: float = REPLAY_LOGIT_ATOL,
    forbidden_read_roots: Sequence[str | os.PathLike] = (),
    install_guards: bool = True,
) -> dict[str, Any]:
    """Independently rebuild the model and the exported pockets, predict again and compare.

    Meant for a fresh process: the guards installed here last for the life of the process.
    ``install_guards=False`` is only for callers already running under equivalent guards.
    Writes the replayed CSVs and ``ec_prediction_replay_receipt.json`` to ``output_dir`` and
    raises after writing them when anything differs.
    """
    from training.access_guard import install_forbidden_read_guard, install_guard_from_environment
    from training.ec_membership import read_ec_fold_membership
    from training.parallel_loading import PARSE_CACHE_ENV
    from training.raw_graph_cache import GRAPH_CACHE_ENV
    from training.run import normalization_stats_from_payload, to_jsonable
    from training.structure_loading import find_structure_files

    _require(0.0 < float(logit_atol) <= MAX_REPLAY_LOGIT_ATOL, f"logit_atol must be in (0, {MAX_REPLAY_LOGIT_ATOL}]")
    if install_guards:
        # Held-out roots are guarded before the first read; caches stay off so every input is re-derived.
        sys.dont_write_bytecode = True
        for name in (PARSE_CACHE_ENV, GRAPH_CACHE_ENV):
            os.environ.pop(name, None)
        install_guard_from_environment()
        install_forbidden_read_guard(forbidden_read_roots)
    run_dir = Path(run_dir).resolve()
    receipt = json.loads((run_dir / EC_PREDICTION_RECEIPT_NAME).read_text(encoding="utf-8"))
    payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    config = payload["config"]
    _validate_replay_config(config)
    mode = receipt["mode"]
    _require(mode in PREDICTION_FILE_NAMES, f"Unknown export mode {mode!r}")
    expected_status = "match" if mode == MODE_VALIDATION else "not_applicable"
    _require(receipt.get("reconciliation_status") == expected_status,
             f"Replay refuses an export whose reconciliation status is {receipt.get('reconciliation_status')!r} "
             f"(expected {expected_status!r})")
    checkpoint_path = run_dir / receipt["checkpoint"]["path"]
    _require(_sha256_file(checkpoint_path) == receipt["checkpoint"]["sha256"],
             "Checkpoint content changed since the export")
    _require(_sha256_file(run_dir / "run_config.json") == receipt["run_config_sha256"],
             "run_config.json changed since the export")
    pocket_name, protein_name = PREDICTION_FILE_NAMES[mode]
    for name, record in ((pocket_name, receipt["pocket_predictions"]), (protein_name, receipt["protein_predictions"])):
        _require(record["path"] == name and _sha256_file(run_dir / name) == record["sha256"],
                 f"Saved prediction file {name} changed since the export")

    if mode == MODE_VALIDATION:
        membership = read_ec_fold_membership(Path(config["fold_membership_csv"]), str(config["fold_membership_sha256"]))
        structure_ids = {row["structure_id"] for row in membership.rows if int(row["fold"]) == int(config["fold_index"])}
    else:
        structure_ids = set(read_probe_structure_ids(Path(config["ec_probe_structure_ids"]),
                                                     str(config["ec_probe_structure_ids_sha256"])))
    _require(_canonical_sha256(sorted(structure_ids)) == receipt["structure_ids_sha256"],
             "Replay structures differ from the exported structures")
    structure_files = [path for path in find_structure_files(Path(config["structure_dir"])) if path.stem in structure_ids]
    missing = structure_ids - {path.stem for path in structure_files}
    _require(not missing, f"{len(missing)} exported structure(s) were not found in the recorded structure directory")
    output_dir = Path(output_dir).resolve() if output_dir is not None else run_dir / DEFAULT_REPLAY_DIRNAME
    _require(not output_dir.exists(), f"Replay output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    data_reads: set = set()
    if install_guards:
        _roots, data_reads = install_recorded_input_guard(
            [*_recorded_read_roots(config), *structure_files, run_dir], output_dir=output_dir,
        )

    model, checkpoint = rebuild_model_from_checkpoint(checkpoint_path, device=device)
    _require(to_jsonable(checkpoint["config"]) == config, "Checkpoint configuration differs from run_config.json")
    _require_fixed_seven_checkpoint(checkpoint)
    pockets = _reload_exported_pockets(config, checkpoint, structure_ids, training_side=mode == MODE_PROBE)
    loader = _prediction_loader(pockets, config, normalization_stats_from_payload(checkpoint["normalization_stats"]))
    predictions = _predict(model, loader, pockets, device)
    identity = _identity(receipt["run_id"], mode, config)
    pocket_rows, protein_rows, conflicting = build_prediction_rows(pockets, predictions["ec_logits"], identity)
    replay_files = {
        "pocket_predictions": _write_csv_once(output_dir / pocket_name, EC_POCKET_PREDICTION_COLUMNS, pocket_rows),
        "protein_predictions": _write_csv_once(output_dir / protein_name, EC_PROTEIN_PREDICTION_COLUMNS, protein_rows),
    }
    pocket_largest, pocket_problems, pocket_near_ties = compare_prediction_rows(
        read_prediction_csv(run_dir / pocket_name, EC_POCKET_PREDICTION_COLUMNS), pocket_rows,
        key_columns=POCKET_KEY_COLUMNS, logit_atol=logit_atol)
    protein_largest, protein_problems, protein_near_ties = compare_prediction_rows(
        read_prediction_csv(run_dir / protein_name, EC_PROTEIN_PREDICTION_COLUMNS), protein_rows,
        key_columns=PROTEIN_KEY_COLUMNS, logit_atol=logit_atol)
    problems = pocket_problems + protein_problems
    if conflicting != receipt["n_conflicting_groups"]:
        problems.append(f"conflicting groups {conflicting} != {receipt['n_conflicting_groups']}")
    replay_receipt = {
        "schema_version": EC_PREDICTION_SCHEMA_VERSION,
        "mode": mode,
        "run_id": receipt["run_id"],
        "source_run_dir": str(run_dir),
        "source_receipt_sha256": _sha256_file(run_dir / EC_PREDICTION_RECEIPT_NAME),
        "source_reconciliation_status": receipt["reconciliation_status"],
        "checkpoint_sha256": receipt["checkpoint"]["sha256"],
        "device": device,
        "logit_atol": float(logit_atol),
        "max_pocket_logit_abs_difference": pocket_largest,
        "max_protein_logit_abs_difference": protein_largest,
        "near_tie_pred_changes": {"pocket": pocket_near_ties, "protein": protein_near_ties,
                                  "margin_limit": 2.0 * float(logit_atol)},
        "problems": problems[:50],
        "n_problems": len(problems),
        "replay_status": "match" if not problems else "mismatch",
        "replay_files": replay_files,
        "guards_installed": bool(install_guards),
        "training_performed": False,
        "optimizer_created": False,
        "normalization_refitted": False,
        "input_file_hashes": [{"path": str(path), "sha256": _sha256_file(path)}
                              for path in sorted(data_reads) if Path(path).is_file()],
        "export_source_sha256": _sha256_file(Path(__file__)),
        "created_utc": _utc_now(),
    }
    _write_json_once(output_dir / EC_REPLAY_RECEIPT_NAME, replay_receipt)
    if problems:
        raise EcPredictionExportError(f"EC prediction replay differs from the export: {problems[:5]}")
    return replay_receipt


# ---------------------------------------------------------------------------
# Label-free test pass
# ---------------------------------------------------------------------------

def is_placeholder_structure_id(structure_id: str) -> bool:
    match = STRUCTURE_ID_RE.match(structure_id)
    return match is not None and match.group("ec") == TEST_PLACEHOLDER_EC


def _label_free_name(structure_id: str) -> str:
    """A structure name for error messages that never repeats an EC label."""
    match = STRUCTURE_ID_RE.match(structure_id)
    return f"{match.group('pdbid')} (chain {match.group('chain')})" if match else "an unparseable structure name"


def _read_freeze2_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    _require(Path(path).is_file(), f"Freeze-2 manifest not found: {path}")
    data = Path(path).read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    _require(actual == str(expected_sha256).strip().lower(),
             f"Freeze-2 manifest {path} has SHA-256 {actual}, expected {expected_sha256}")
    manifest = json.loads(data.decode("utf-8"))
    _require(isinstance(manifest, dict) and manifest.get("schema_version") == FREEZE2_SCHEMA_VERSION
             and isinstance(manifest.get("checkpoint_sha256s"), list) and isinstance(manifest.get("code_sha256"), dict)
             and isinstance(manifest.get("test_pass_output_root"), str),
             "Freeze-2 manifest lacks schema_version 1, checkpoint_sha256s, code_sha256 or test_pass_output_root")
    _require(Path(manifest["test_pass_output_root"]).is_absolute(),
             "Freeze-2 manifest test_pass_output_root must be an absolute path")
    return manifest


def freeze2_required_code_files() -> list[str]:
    """Repository paths a Freeze-2 manifest must pin: ``FREEZE2_REQUIRED_CODE_PATHS`` expanded on the live tree."""
    files: set[str] = set()
    for pattern in FREEZE2_REQUIRED_CODE_PATHS:
        matches = sorted(REPO_ROOT.glob(pattern)) if "*" in pattern else [REPO_ROOT / pattern]
        _require(bool(matches) and all(path.is_file() for path in matches),
                 f"Required code path {pattern} is missing from this tree")
        files.update(path.relative_to(REPO_ROOT).as_posix() for path in matches)
    return sorted(files)


def _verify_freeze2_code(code_sha256: Mapping[str, Any]) -> dict[str, str]:
    """Every pinned file must hash to its manifest entry, and the required files must all be pinned."""
    pinned: dict[str, str] = {}
    for key, value in code_sha256.items():
        parts = Path(str(key)).parts
        _require(not Path(str(key)).is_absolute() and parts[:1] == ("src",) and ".." not in parts
                 and str(key).endswith(".py"), f"Freeze-2 manifest code entry {key!r} is not a src/...py path")
        pinned[str(key)] = str(value).strip().lower()
    missing = [path for path in freeze2_required_code_files() if path not in pinned]
    _require(not missing, f"Freeze-2 manifest does not pin {len(missing)} required code file(s), e.g. {missing[:3]}")
    absent = [key for key in sorted(pinned) if not (REPO_ROOT / key).is_file()]
    _require(not absent, f"Freeze-2 manifest pins code files missing from this tree: {absent[:5]}")
    differing = [key for key in sorted(pinned) if _sha256_file(REPO_ROOT / key) != pinned[key]]
    _require(not differing, f"Live code differs from the Freeze-2 manifest: {differing[:5]}")
    return pinned


def _ledger_pass_ids(ledger: Path) -> set[str]:
    if not ledger.is_file():
        return set()
    pass_ids: set[str] = set()
    for number, line in enumerate(ledger.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            pass_ids.add(str(json.loads(line)["pass_id"]))
        except (ValueError, KeyError, TypeError):
            raise EcPredictionExportError(f"Test pass ledger {ledger} line {number} is unreadable") from None
    return pass_ids


def _append_ledger(ledger: Path, entry: Mapping[str, Any]) -> None:
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(entry), sort_keys=True) + "\n")


def read_test_site_table(path: Path) -> dict[tuple[str, str, str], str]:
    """Accession-keyed selected sites as loader site keys (accession, placeholder EC, chain_resi) -> metal.

    The table has exactly the columns accession, chain_resi and metaltype, so it cannot carry a label.
    """
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = [column.strip() for column in (reader.fieldnames or [])]
        _require(sorted(columns) == sorted(TEST_SITE_TABLE_COLUMNS),
                 f"Test site table {Path(path).name} must have exactly the columns {list(TEST_SITE_TABLE_COLUMNS)}")
        sites: dict[tuple[str, str, str], str] = {}
        for row in reader:
            _require(None not in row, f"Test site table {Path(path).name} has a row with extra fields")
            row = {key.strip(): (value or "").strip() for key, value in row.items()}
            chain, resseq = parse_summary_chain_resi(row["chain_resi"])
            metal = row["metaltype"].upper()
            _require(bool(row["accession"]) and bool(metal), f"Empty accession or metal type in {Path(path).name}")
            key = (row["accession"].lower(), TEST_PLACEHOLDER_EC, f"{chain}_{resseq}")
            _require(key not in sites, f"Repeated test site {row['accession']} {row['chain_resi']}")
            sites[key] = metal
    _require(bool(sites), f"Test site table {Path(path).name} holds no sites")
    return sites


def export_ec_test_predictions(
    freeze2_manifest_path: Path,
    expected_sha256: str,
    pass_id: str,
    *,
    checkpoint_path: Path,
    structure_dir: Path,
    site_table_csv: Path,
    output_root: Path,
    esm_embeddings_dir: Path,
    external_features_root_dir: Path,
    ring_features_dir: Path | None = None,
    global_esm_cache_dir: Path | None = None,
    global_esm_manifest_sha256: str | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """The single label-free test pass of one frozen checkpoint; see the module docstring (D7, D12).

    Writes ``<output_root>/<pass_id>/ec_test_protein_predictions.csv`` (accession, pocket count,
    predicted class index, logits) and ``ec_test_prediction_receipt.json``, never a label or a
    metric. Structures that yield no pocket are listed in the receipt, not predicted.
    """
    from training.structure_loading import StructureLoadError, find_structure_files, load_structure_pockets
    from training.feature_paths import resolve_runtime_feature_paths

    _require(bool(_PASS_ID_RE.match(str(pass_id))), f"Invalid pass ID {pass_id!r}")
    manifest = _read_freeze2_manifest(Path(freeze2_manifest_path), expected_sha256)
    pinned_code = _verify_freeze2_code(manifest["code_sha256"])
    module_sha256 = pinned_code[EXPORT_SOURCE_KEY]
    checkpoint_sha256 = _sha256_file(Path(checkpoint_path))
    _require(checkpoint_sha256 in manifest["checkpoint_sha256s"], "Freeze-2 manifest does not list this checkpoint")
    output_root = Path(output_root).resolve()
    _require(output_root == Path(manifest["test_pass_output_root"]).resolve(),
             f"Test pass output root {output_root} is not the Freeze-2 manifest's test_pass_output_root")
    ledger = output_root / TEST_PASS_LEDGER_NAME
    _require(str(pass_id) not in _ledger_pass_ids(ledger), f"Test pass ID {pass_id!r} was already used")
    pass_dir = output_root / str(pass_id)
    _require(not pass_dir.exists(), f"Test pass output already exists: {pass_dir}")

    structure_files = find_structure_files(Path(structure_dir))
    _require(bool(structure_files), f"No structures under {structure_dir}")
    labelled = [path.stem for path in structure_files if not is_placeholder_structure_id(path.stem)]
    _require(not labelled, f"{len(labelled)} test structure name(s) do not carry the {TEST_PLACEHOLDER_SUFFIX} "
                           f"placeholder, e.g. {[_label_free_name(stem) for stem in labelled[:3]]}")
    accessions = [accession_from_structure_id(path.stem) for path in structure_files]
    _require(len(set(accessions)) == len(accessions), "Test structures repeat an accession")
    sites = read_test_site_table(Path(site_table_csv))

    model, checkpoint = rebuild_model_from_checkpoint(Path(checkpoint_path), device=device)
    _require_fixed_seven_checkpoint(checkpoint)
    config = checkpoint["config"]
    _require(config.get("site_extraction", "component") == "selected_sites",
             "The label-free test pass needs a checkpoint trained with --site-extraction selected_sites")
    _require(not config.get("use_ring_edges") or ring_features_dir is not None,
             "This checkpoint uses RING edges; pass the test ring_features_dir")
    if config.get("use_global_esm_branch"):
        _require(global_esm_cache_dir is not None and bool(global_esm_manifest_sha256),
                 "This checkpoint has the whole-protein branch; pass the test chain-mean cache and its manifest SHA-256")

    output_root.mkdir(parents=True, exist_ok=True)
    try:
        pass_dir.mkdir(exist_ok=False)  # atomic: of two passes with one ID, only one gets here
    except FileExistsError:
        raise EcPredictionExportError(f"Test pass output already exists: {pass_dir}") from None
    claim = {"pass_id": str(pass_id), "status": "claimed", "freeze2_manifest_sha256": str(expected_sha256).lower(),
             "checkpoint_sha256": checkpoint_sha256, "export_source_sha256": module_sha256, "utc": _utc_now()}
    _append_ledger(ledger, claim)

    embeddings_dir, feature_root = resolve_runtime_feature_paths(
        structure_dir=Path(structure_dir), esm_embeddings_dir=Path(esm_embeddings_dir),
        external_features_root_dir=Path(external_features_root_dir),
        external_feature_source=config["external_feature_source"],
    )
    pockets: list[PocketRecord] = []
    failures: list[dict[str, str]] = []
    inputs: list[dict[str, str]] = []
    for path, accession in zip(structure_files, accessions):
        inputs.append({"accession": accession, "sha256": _sha256_file(path)})
        try:
            loaded, _fallbacks, skipped = load_structure_pockets(
                structure_path=path, structure_root=Path(structure_dir), allowed_site_metal_labels=sites,
                esm_dim=int(config["esm_dim"]), embeddings_dir=embeddings_dir,
                require_esm_embeddings=bool(config["require_esm_embeddings"]),
                ring_features_dir=Path(ring_features_dir) if ring_features_dir is not None else None,
                feature_root_dir=feature_root, external_feature_source=config["external_feature_source"],
                require_external_features=bool(config["require_external_features"]),
                unsupported_metal_policy=config["unsupported_metal_policy"], ec_label_depth=1,
                metal_example_unit="pocket", site_extraction="selected_sites",
                selected_site_pocket_grouping=_pocket_grouping(config),
            )
        except (StructureLoadError, ValueError) as exc:
            failures.append({"accession": accession, "reason": "load_error", "detail": str(exc)})
            continue
        if not loaded:
            reasons = sorted({row.get("reason", "") for row in skipped}) or ["no_pocket"]
            failures.append({"accession": accession, "reason": ";".join(reasons), "detail": ""})
            continue
        for pocket in loaded:
            # The placeholder is not a label: no token, no target.
            pocket.metadata.pop("ec_label_token", None)
            pocket.y_ec = None
            pocket.metadata["test_accession"] = accession
        pockets.extend(loaded)
    _require(bool(pockets), "No test structure yielded a pocket")

    if config.get("use_global_esm_branch"):
        _attach_recorded_global_vectors(pockets, {**config, "global_esm_cache_dir": str(global_esm_cache_dir),
                                                  "global_esm_manifest_sha256": global_esm_manifest_sha256},
                                        (checkpoint.get("dataset_summary") or {}).get("global_esm") or {})
    from training.run import normalization_stats_from_payload

    loader = _prediction_loader(pockets, config, normalization_stats_from_payload(checkpoint["normalization_stats"]))
    logits = _predict(model, loader, pockets, device)["ec_logits"]
    members: dict[str, list[int]] = {}
    for index, pocket in enumerate(pockets):
        members.setdefault(pocket.metadata["test_accession"], []).append(index)
    rows = []
    for accession in sorted(members):
        mean = torch.stack([logits[index] for index in members[accession]], dim=0).mean(dim=0)
        rows.append({"pass_id": str(pass_id), "accession": accession, "n_pockets": str(len(members[accession])),
                     "pred": str(int(mean.argmax())),
                     **{column: _format_logit(value) for column, value in zip(LOGIT_COLUMNS, mean.tolist())}})
    predictions_file = _write_csv_once(pass_dir / TEST_PROTEIN_PREDICTIONS_NAME, TEST_PROTEIN_PREDICTION_COLUMNS, rows)
    receipt = {
        "schema_version": EC_PREDICTION_SCHEMA_VERSION,
        "pass_id": str(pass_id),
        "label_free": True,
        "labels_read": False,
        "metrics_computed": False,
        "placeholder_suffix": TEST_PLACEHOLDER_SUFFIX,
        "freeze2_manifest": {"path": str(freeze2_manifest_path), "sha256": str(expected_sha256).lower(),
                             "test_pass_output_root": str(output_root),
                             "n_code_files_verified": len(pinned_code),
                             "code_sha256_identity": _canonical_sha256(pinned_code)},
        "checkpoint_sha256": checkpoint_sha256,
        "export_source_sha256": module_sha256,
        "checkpoint_site_extraction": config["site_extraction"],
        "model": _model_identity(config),
        "seed": int(config["seed"]),
        "ec_class_tokens": list(EC_CLASS_TOKENS),
        "vocabulary_sha256": ec_class_vocabulary_sha256(),
        "aggregation": AGGREGATION,
        "site_extraction": "selected_sites",
        "site_table_sha256": _sha256_file(Path(site_table_csv)),
        "global_esm_manifest_sha256": global_esm_manifest_sha256 if config.get("use_global_esm_branch") else None,
        "input_structures": inputs,
        "n_structures": len(structure_files),
        "n_proteins_predicted": len(rows),
        "failures": failures,
        "protein_predictions": predictions_file,
        "created_utc": _utc_now(),
        **_pocket_grouping_record(config),
    }
    receipt_sha256 = _write_json_once(pass_dir / TEST_RECEIPT_NAME, receipt)
    _append_ledger(ledger, {**claim, "status": "completed", "receipt_sha256": receipt_sha256,
                            "predictions_sha256": predictions_file["sha256"], "utc": _utc_now()})
    return receipt


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EC prediction export tools (run the replay in a fresh process).")
    actions = parser.add_subparsers(dest="action", required=True)
    replay = actions.add_parser("replay", help="Rebuild a run's exported predictions and compare them with its CSVs.")
    replay.add_argument("--run-dir", type=Path, required=True)
    replay.add_argument("--output-dir", type=Path, default=None,
                        help=f"New directory for the replay outputs (default RUN/{DEFAULT_REPLAY_DIRNAME}).")
    replay.add_argument("--device", default="cpu")
    replay.add_argument("--logit-atol", type=float, default=REPLAY_LOGIT_ATOL,
                        help=f"Largest allowed logit difference (default {REPLAY_LOGIT_ATOL}, "
                             f"at most {MAX_REPLAY_LOGIT_ATOL}).")
    replay.add_argument("--forbidden-read-root", type=Path, action="append", default=[],
                        help="Held-out root this process must never read "
                             "(repeatable; adds to DEEPMZYME_FORBIDDEN_READ_ROOTS).")
    args = parser.parse_args(argv)
    receipt = replay_ec_predictions(args.run_dir, output_dir=args.output_dir, device=args.device,
                                    logit_atol=args.logit_atol, forbidden_read_roots=args.forbidden_read_root)
    print(json.dumps({key: receipt[key] for key in (
        "run_id", "mode", "replay_status", "source_reconciliation_status", "max_pocket_logit_abs_difference",
        "max_protein_logit_abs_difference")},
        sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
