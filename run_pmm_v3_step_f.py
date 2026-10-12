#!/usr/bin/env python3
"""VM-side and CPU tooling of v3 step F (rules: docs/campaigns/pmm_ion_metal_v3/step_f_spec.json; draft v3).

Additive: the hash-frozen runner files and ``src/`` are imported, never changed. The workstation launcher
(pmm_v3_step_f.py) copies this file, pmm_v3_test_prep.py and the specification next to the frozen runner.

Actions:
  plan-refit       CPU, read-only: the Stage 6B refit command, its identity and its differences from the selected
                   cell's fold command (only fold, validation, weight, selection-metric and identity flags may differ).
  refit            F2: admit, run, verify, reload-check and persist the validation-free refit of the selected cell
                   (training data only; the allocation fields of run_pmm_v3_campaign.py).
  reload-check     F2: rerun the reload check of a completed refit directory.
  prepare-test     F3b, only under the recorded Freeze 1: label-blind preparation of the test rows
                   (pmm_v3_test_prep.prepare).
  test-embeddings  F3b, only under the recorded Freeze 1, on the workstation CPU: ESMC embeddings of the planned test
                   context chains in the training embeddings' model dtype.
  predict          F4, only under the recorded Freeze 1 and Freeze 2: the single prediction pass with the bound
                   checkpoint after a campaign-wide write-once claim; immutable predictions for every source row
                   (unscorable rows carry no prediction).

The test-side actions read the freeze records only at their canonical paths under --campaign-dir (step_f_evidence/)
and run only with the exact code and specification Freeze 1 binds; after Freeze 2 or the prediction claim the
preparation actions are refused. The refit never reads held-out data (the training read guard is installed). The test
actions never use a label: the preparation reads allowlisted identifier columns (label-bearing cells pass through the
CSV reader and are discarded), and the site manifest, the summary tables, the crosswalk and the source CSV are refused
by a read guard during prediction.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pmm_v3_campaign as v3
import pmm_v3_test_prep as prep

SPEC, SPEC_PATH = prep.SPEC, prep.SPEC_PATH
SELECTION, REFIT = SPEC["selection"], SPEC["refit"]
REFIT_NAME = REFIT["run_name"]
COMMON4 = ("Mn", "Cu", "Zn", "Class VIII")
LOGGED_BA_TOLERANCE = 0.001
ALLOWED_REFIT_CONFIG_DIFFERENCES = frozenset({
    "n_folds", "fold_index", "fold_membership_csv", "fold_membership_sha256", "fold_split_source",
    "export_validation_predictions", "selection_metric", "mn_loss_multiplier", "cu_loss_multiplier",
    "zn_loss_multiplier", "class_viii_loss_multiplier", "run_name", "campaign_run_identity"})
RUNTIME_FIELDS = ("session_id", "execution_deadline", "allocation_started", "execution_max_seconds",
                  "estimated_fit_seconds", "durable_root", "persistence_mode")
require = v3.require


def file_sha(path: Path) -> str:
    return prep.sha256_file(path)


def _set_argument(argv: list[str], flag: str, value: str | None = None, *, remove: bool = False) -> None:
    """Remove every occurrence of ``flag`` (and its value) and, unless ``remove``, append it with ``value``."""
    while flag in argv:
        index = argv.index(flag)
        del argv[index]
        if index < len(argv) and not argv[index].startswith("--"):
            del argv[index]
    if not remove:
        argv.append(flag)
        if value is not None:
            argv.append(value)


# ---------------------------------------------------------------------------
# F2: the validation-free refit of the selected cell
# ---------------------------------------------------------------------------

def selected_unit(fold: int = 1) -> v3.Unit:
    """The selected cell at one fold; only its fold-specific flags are later replaced."""
    return v3.Unit(SELECTION["family"], SELECTION["target"], SELECTION["recipe"], fold, SELECTION["seed"])


def full_cohort_weights(paths: v3.V3Paths) -> dict[str, Any]:
    """w_c = N / (4 n_c) on the common four over the full non-test cohort (the campaign's weight rule)."""
    from training.source_cohort import read_cohort_csv

    manifest = json.loads(paths.manifest.read_text())
    bindings = read_cohort_csv(paths.cohort, expected_sha256=manifest["cohort"]["sha256"])
    counts = Counter(v3.v2.COMMON_FOUR[binding.native_element] for binding in bindings)
    require(set(counts) == set(COMMON4), f"The full cohort lacks a common-four class: {dict(counts)}")
    weights = {label: len(bindings) / (4.0 * counts[label]) for label in COMMON4}
    return {"rule": "w_c = N / (4 * n_c) on the common-four endpoint, full non-test cohort", "n": len(bindings),
            "common_four_counts": {label: counts[label] for label in COMMON4}, "common_four_weights": weights,
            "four_class_multipliers": {"mn": weights["Mn"], "cu": weights["Cu"], "zn": weights["Zn"],
                                       "class_viii": weights["Class VIII"]}}


def build_refit(paths: v3.V3Paths, *, epochs: int, selection_sha256: str, python_bin: str, train_dir: Path,
                esm_dir: Path | None, device: str, lane: int, load_workers: int | None = None,
                amp: bool | None = None) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    """The selected cell's command with the fold and validation flags removed, full-cohort weights and a refit
    identity (draft v3 section F2)."""
    from training.config import config_to_payload, parse_args

    require(SELECTION["target"] == "four_class", "This refit builder covers the four-class selection only")
    amp = v3.resolve_amp(paths, probe=None, amp=None) if amp is None else amp
    command, env, _ = v3.build_command(paths, selected_unit(), python_bin=python_bin, train_dir=train_dir,
                                       esm_dir=esm_dir, device=device, lane=lane, load_workers=load_workers,
                                       epochs=epochs, amp=amp, run_name=REFIT_NAME)
    for flag in REFIT["removed_flags"]:
        _set_argument(command, flag, remove=True)
    for flag, value in REFIT["added_flags"].items():
        _set_argument(command, flag, value)
    _set_argument(command, "--run-name", REFIT_NAME)
    weights = full_cohort_weights(paths)
    for key, value in weights["four_class_multipliers"].items():
        _set_argument(command, f"--{key.replace('_', '-')}-loss-multiplier", repr(float(value)))
    resolved = config_to_payload(parse_args(command[3:]))
    require(resolved["val_fraction"] == 0.0 and resolved["n_folds"] is None and resolved["fold_index"] is None
            and resolved["epochs"] == epochs and resolved["checkpoint_rule"] == REFIT["checkpoint_rule"]
            and resolved["lr_schedule"] == REFIT["lr_schedule"] and resolved["seed"] == REFIT["seed"]
            and not resolved["export_validation_predictions"] and not resolved["run_test_eval"]
            and not resolved["allow_final_refit_test_eval"] and resolved["selection_metric"] == "train_loss",
            "The refit command does not resolve to the specified validation-free terminal refit")
    manifest = json.loads(paths.manifest.read_text())
    identity = {
        "campaign_id": v3.CAMPAIGN_ID, "phase": "stage6b_full_non_test_refit",
        "family": SELECTION["family"], "target_scheme": SELECTION["target"], "recipe": SELECTION["recipe"],
        "selection_cell": SELECTION["cell"], "selection_record_sha256": selection_sha256,
        "step_f_spec_sha256": file_sha(SPEC_PATH), "step_f_runner_sha256": file_sha(Path(__file__)),
        "profile_sha256": manifest["profile_sha256"], "cohort_sha256": manifest["cohort"]["sha256"],
        "cohort_scope": "full non-test cohort", "fold_set_id": None,
        "feature_inventory_sha256": manifest["feature_inventory_identity_sha256"],
        "source_tree_sha256": manifest["frozen_source_tree_sha256"], "runner_sha256": v3.runner_sha256(),
        "common_four_weights": weights["common_four_weights"],
        "weights_sha256": v3.stable_hash(weights["common_four_weights"]),
        "model_seed": REFIT["seed"], "epochs": epochs, "amp": bool(amp), "checkpoint_rule": REFIT["checkpoint_rule"],
        "validation": "none", "calibration": None, "ensemble": False,
        "resolved_config_sha256": v3.stable_hash({k: v for k, v in resolved.items()
                                                  if k not in v3.v2.NON_IDENTITY_CONFIG_KEYS}),
    }
    if paths.extension.exists():
        identity["campaign_extension"] = {"id": v3.EXTENSION_ID, "sha256": file_sha(paths.extension)}
    identity = json.loads(json.dumps(identity, sort_keys=True))
    command += ["--campaign-run-identity", json.dumps(identity, sort_keys=True)]
    return command, env, identity


def refit_differences(paths: v3.V3Paths, **kwargs) -> dict[str, Any]:
    """Resolved trainer configuration of the refit against the selected cell's fold-1 command."""
    from training.config import config_to_payload, parse_args

    mine, _, identity = build_refit(paths, **kwargs)
    common = {key: kwargs[key] for key in ("python_bin", "train_dir", "esm_dir", "device", "lane")}
    theirs, _, _ = v3.build_command(paths, selected_unit(), epochs=kwargs["epochs"],
                                    amp=v3.resolve_amp(paths, probe=None, amp=None) if kwargs.get("amp") is None
                                    else kwargs["amp"], **common)
    a, b = config_to_payload(parse_args(mine[3:])), config_to_payload(parse_args(theirs[3:]))
    differing = sorted(key for key in set(a) | set(b) if a.get(key) != b.get(key))
    return {"differing": differing, "unexpected": sorted(set(differing) - ALLOWED_REFIT_CONFIG_DIFFERENCES),
            "multipliers": {k: a[k] for k in ("mn_loss_multiplier", "cu_loss_multiplier", "zn_loss_multiplier",
                                               "class_viii_loss_multiplier")},
            "fold_multipliers": {k: b[k] for k in ("mn_loss_multiplier", "cu_loss_multiplier", "zn_loss_multiplier",
                                                    "class_viii_loss_multiplier")},
            "identity": identity, "command": mine}


def terminal_checkpoint(run_dir: Path) -> Path:
    for name in ("terminal_model_checkpoint.pt", "last_model_checkpoint.pt"):
        if (run_dir / name).is_file():
            return run_dir / name
    raise ValueError(f"{run_dir}: no terminal checkpoint")


def verify_refit(run_dir: Path, identity: dict[str, Any], *, epochs: int) -> dict[str, Any]:
    """The refit completed every epoch with no validation partition, no test report and the bound identity."""
    import torch

    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    payload = json.loads((run_dir / "run_config.json").read_text())
    require(metadata.get("fit_status") == "completed" and metadata.get("test_report") is None,
            "The refit did not complete or wrote a held-out report")
    require(metadata.get("campaign_run_identity") == identity, "The refit's recorded identity differs")
    history = payload["history"]
    require([int(row["epoch"]) for row in history] == list(range(1, epochs + 1)),
            f"The refit history is not epochs 1..{epochs}")
    config = payload["config"]
    require(config["val_fraction"] == 0.0 and config.get("n_folds") is None and config.get("fold_index") is None
            and int(config["epochs"]) == epochs and config["checkpoint_rule"] == "terminal",
            "The refit configuration has a validation partition or another schedule")
    require(not (run_dir / "val_predictions.csv").exists(), "A validation prediction file exists")
    checkpoint_path = terminal_checkpoint(run_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(len(checkpoint["history"]) == epochs and checkpoint["config"]["val_fraction"] == 0.0
            and checkpoint["config"].get("n_folds") is None,
            "The terminal checkpoint is not the completed validation-free epoch")
    require(json.loads(checkpoint["config"]["campaign_run_identity"]) == identity, "The checkpoint identity differs")
    return {"checkpoint": checkpoint_path.name, "checkpoint_sha256": file_sha(checkpoint_path),
            "selected_epoch": epochs, "completed_epochs": epochs,
            "run_config_sha256": file_sha(run_dir / "run_config.json"),
            "run_metadata_sha256": file_sha(run_dir / "run_metadata.json"),
            "logged_train_common4_ba": history[-1].get("train_metal_collapsed4_balanced_acc")}


# ---------------------------------------------------------------------------
# Shared inference path (the trainer's cohort-bound loader; every row kept)
# ---------------------------------------------------------------------------

def load_bound_pockets(bindings: list[Any], structure_root: Path, config: dict[str, Any], *, esm_dir: Path | None,
                       external_root: Path | None) -> list[Any]:
    """Pockets of the given bindings exactly as the trainer builds them, in binding order."""
    from training.data import _cohort_structure_files
    from training.source_cohort import cohort_load_payload
    from training.structure_loading import load_structure_pockets

    payload = cohort_load_payload(bindings)
    pockets = []
    for path in _cohort_structure_files(Path(structure_root), bindings):
        loaded, _, _ = load_structure_pockets(
            structure_path=path, structure_root=Path(structure_root), allowed_site_metal_labels=None,
            esm_dim=config["esm_dim"], embeddings_dir=esm_dir, require_esm_embeddings=config["require_esm_embeddings"],
            ring_features_dir=config.get("ring_features_dir"), feature_root_dir=external_root,
            external_feature_source=config["external_feature_source"],
            require_external_features=config["require_external_features"],
            unsupported_metal_policy=config["unsupported_metal_policy"], ec_label_depth=config["ec_label_depth"],
            metal_example_unit="ion", cohort_bindings=payload)
        pockets.extend(loaded)
    by_uid = {pocket.metadata["source_uid"]: pocket for pocket in pockets}
    require(len(by_uid) == len(pockets) == len(bindings), "The loader did not build one pocket per binding")
    return [by_uid[binding.source_uid] for binding in bindings]


def infer_logits(model, pockets: list[Any], normalization, options: dict[str, Any], *, batch_size: int,
                 device: str):
    """Metal logits for every pocket in order. Unlike the trainer's evaluation helper, which keeps only labelled
    examples, every row is kept: the test pockets carry no label."""
    import torch
    from torch_geometric.loader import DataLoader

    from training.graph_dataset import PocketGraphDataset, build_graph_data_list

    graphs = build_graph_data_list(pockets, **options)
    loader = DataLoader(PocketGraphDataset(pockets, precomputed_data=graphs, normalization_stats=normalization,
                                           **options), batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    chunks = []
    with torch.no_grad():
        for batch in loader:
            chunks.append(model(batch.to(device))["logits_metal"].float().cpu())
    logits = torch.cat(chunks) if chunks else torch.zeros((0, 0))
    require(logits.size(0) == len(pockets), f"{logits.size(0)} logits for {len(pockets)} pockets")
    require(bool(torch.isfinite(logits).all()), "Inference produced nonfinite logits")
    return logits


def common_four(probabilities):
    """Native probabilities summed into the common four (identity for the four-class scheme)."""
    from training.final_test_reporting import collapse_metal_probabilities

    return collapse_metal_probabilities(probabilities)


def reload_check(run_dir: Path, paths: v3.V3Paths, identity: dict[str, Any], *, epochs: int, train_dir: Path,
                 esm_dir: Path | None, external_root: Path | None, device: str) -> dict[str, Any]:
    """Reload the terminal checkpoint, check its configuration and parameters, and re-evaluate the training-set
    common-four BA at the final epoch against the logged value."""
    import numpy as np
    import torch

    from training.campaign_runtime import load_campaign_prediction_components
    from training.source_cohort import read_cohort_csv

    receipt = verify_refit(run_dir, identity, epochs=epochs)
    # The trainer's training-side caches (the refit just filled them), so the full cohort is not rebuilt cold.
    os.environ.update(DEEPMZYME_PARSE_CACHE_DIR=str(paths.parse_cache), DEEPMZYME_GRAPH_CACHE_DIR=str(paths.graph_cache))
    checkpoint = torch.load(run_dir / receipt["checkpoint"], map_location="cpu", weights_only=False)
    require(all(bool(torch.isfinite(t).all()) for t in checkpoint["model_state_dict"].values()
                if torch.is_tensor(t) and t.is_floating_point()), "The checkpoint has nonfinite parameters")
    config = checkpoint["config"]
    model, normalization, options = load_campaign_prediction_components(checkpoint, device=device)
    manifest = json.loads(paths.manifest.read_text())
    bindings = read_cohort_csv(paths.cohort, expected_sha256=manifest["cohort"]["sha256"])
    pockets = load_bound_pockets(bindings, train_dir, config, esm_dir=esm_dir, external_root=external_root)
    logits = infer_logits(model, pockets, normalization, options, batch_size=int(config["batch_size"]), device=device)
    probabilities = common_four(logits.softmax(-1))
    from training.final_test_reporting import collapse_metal_targets

    targets = collapse_metal_targets(torch.tensor([int(p.y_metal) for p in pockets])).numpy()
    predictions = probabilities.argmax(-1).numpy()
    recalls = [float(np.mean(predictions[targets == c] == c)) for c in range(4)]
    ba = float(np.mean(recalls))
    logged = receipt["logged_train_common4_ba"]
    require(logged is not None, "The final epoch did not log the training-set common-four BA")
    check = {"checkpoint_sha256": receipt["checkpoint_sha256"], "parameters_finite": True,
             "reevaluated_train_common4_ba": ba, "logged_train_common4_ba": float(logged),
             "abs_difference": abs(ba - float(logged)), "tolerance": LOGGED_BA_TOLERANCE, "n_ions": len(pockets),
             "recalls": dict(zip(COMMON4, recalls)), "device": device}
    check["passed"] = check["abs_difference"] <= LOGGED_BA_TOLERANCE
    (run_dir / "refit_reload_check.json").write_text(json.dumps(check, indent=2, sort_keys=True) + "\n")
    require(check["passed"], f"The reload check failed: {check['abs_difference']:.6f} > {LOGGED_BA_TOLERANCE}")
    return {**receipt, "reload_check": check}


def run_refit(paths: v3.V3Paths, *, epochs: int, selection_sha256: str, lane: int, train_dir: Path,
              esm_dir: Path | None, python_bin: str, device: str, execution_policy: dict[str, Any],
              load_workers: int | None = None) -> dict[str, Any]:
    """Admit, run, verify, reload-check and persist the refit (the frozen run_unit sequence for one refit)."""
    from benchmarking.pmm_execution import CampaignExecution, ExecutionPolicy
    from training.access_guard import install_forbidden_read_guard

    preflight_started = time.time()
    install_forbidden_read_guard(v3.v2.forbidden_read_roots(train_dir))
    require(Path(train_dir).name == "train", "Only the training-side directory named train is allowed")
    name = REFIT_NAME
    existing = v3.unit_artifacts(paths, name)
    if existing:
        raise ValueError(f"Unit artifacts exist ({existing[0]}); runs are never retried automatically")
    manifest = v3.verify_campaign(paths, esm_dir=esm_dir)
    v3.require_frozen_spec(paths)
    command, env, identity = build_refit(paths, epochs=epochs, selection_sha256=selection_sha256,
                                         python_bin=python_bin, train_dir=train_dir, esm_dir=esm_dir, device=device,
                                         lane=lane, load_workers=load_workers)
    v3.require_retry_identity(paths, name, identity)
    lane_root = paths.lane(lane)
    policy = ExecutionPolicy(deadline_unix=execution_policy["deadline_unix"],
                             max_total_seconds=execution_policy["max_total_seconds"],
                             allocation_started_unix=execution_policy.get("allocation_started_unix"))
    execution = CampaignExecution(lane_root, policy=policy, session_id=execution_policy["session_id"],
                                  durable_root=Path(execution_policy["durable_root"]) / lane_root.name,
                                  persistence_mode=execution_policy["persistence_mode"])
    claim = v3.claim_run(paths, name, lane=lane, identity=identity, session_id=execution_policy["session_id"])
    admitted = False
    try:
        with execution:
            execution.admit(name, float(execution_policy["estimated_fit_seconds"]))
            admitted = True
            preflight_seconds = time.time() - preflight_started
            result = execute_refit(paths, lane, command, env, identity, execution=execution, epochs=epochs,
                                   train_dir=train_dir, esm_dir=esm_dir,
                                   external_root=paths.empty_external_features, device=device)
            status_path = lane_root / f"run_status_{name}.json"
            v3.v2.write_json(status_path, {**result, "identity": identity, "lane": lane, "unit": name, "probe": None,
                                           "step": "F2", "preflight_seconds": preflight_seconds,
                                           "manifest_sha256": file_sha(paths.manifest),
                                           "campaign_manifest_cohort_sha256": manifest["cohort"]["sha256"]})
            execution.record_result(name, result["status"], result["elapsed_seconds"])
            execution.persist([path for path in (lane_root / "runs" / name, lane_root / "runs" / f"{name}.log",
                                                 lane_root / "commands" / f"{name}.json", status_path)
                               if path.exists()])
    finally:
        if not admitted:
            claim.unlink(missing_ok=True)
    return result


def execute_refit(paths: v3.V3Paths, lane: int, command: list[str], env_extra: dict[str, str],
                  identity: dict[str, Any], *, execution, epochs: int, train_dir: Path, esm_dir: Path | None,
                  external_root: Path | None, device: str) -> dict[str, Any]:
    from training.config import config_to_payload, parse_args

    name = REFIT_NAME
    lane_root = paths.lane(lane)
    runs_dir, commands_dir = lane_root / "runs", lane_root / "commands"
    runs_dir.mkdir(parents=True, exist_ok=True)
    commands_dir.mkdir(parents=True, exist_ok=True)
    v3.v2.write_json(commands_dir / f"{name}.json", {
        "run_name": name, "argv": command, "env": env_extra, "identity": identity,
        "resolved_config": config_to_payload(parse_args(command[3:])),
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    log_path = runs_dir / f"{name}.log"
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        process = execution.run_subprocess(command, stdout=log, stderr=subprocess.STDOUT,
                                           env={**os.environ, **env_extra}, cwd=v3.ROOT)
    if process.returncode != 0:
        return {"run_name": name, "status": "failed", "return_code": process.returncode,
                "elapsed_seconds": time.time() - started}
    try:
        receipt = reload_check(runs_dir / name, paths, identity, epochs=epochs, train_dir=train_dir, esm_dir=esm_dir,
                               external_root=external_root, device=device)
    except Exception as exc:  # any verification or reload error is recorded, and the artifacts are still persisted
        return {"run_name": name, "status": "failed_verification", "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": time.time() - started}
    return {"run_name": name, "status": "completed", "elapsed_seconds": time.time() - started,
            "selected_epoch": epochs, "selected_checkpoint": receipt["checkpoint"],
            "selected_checkpoint_sha256": receipt["checkpoint_sha256"], "refit_receipt": receipt}


# ---------------------------------------------------------------------------
# F4: the single prediction pass
# ---------------------------------------------------------------------------

def read_prepared(prepared_dir: Path) -> list[dict[str, str]]:
    with (Path(prepared_dir) / "prepared_rows.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_side_tools() -> dict[str, Path]:
    """The code a test-side action runs, which must be the code Freeze 1 binds."""
    return {"run_pmm_v3_step_f.py": Path(__file__), "pmm_v3_test_prep.py": Path(prep.__file__)}


def require_preparation_window(campaign_dir: Path) -> tuple[dict[str, Any], str]:
    """F3b: the recorded Freeze 1 binds this code, and neither Freeze 2 nor the prediction claim exists yet (after
    Freeze 2 the memberships are final)."""
    freeze1, digest = prep.check_freeze1(campaign_dir, tools=test_side_tools())
    paths = prep.evidence_paths(campaign_dir)
    require(not paths["freeze2"].exists() and not paths["prediction"].exists(),
            "Freeze 2 or the prediction claim exists: the label-blind preparation is closed")
    return freeze1, digest


def test_bindings(rows: list[dict[str, str]], structure_root: Path) -> list[Any]:
    """In-memory bindings of the primary scored rows, re-resolved label-free and checked against the prepared ion keys.
    The loader's consistency check needs the residue's metal symbol; it stays in memory and is never written."""
    from graph.structure_parsing import parse_structure_file
    from training.source_cohort import CohortBinding

    files = prep.structure_files(Path(structure_root))
    primaries = [row for row in rows if row["disposition"] != prep.UNSCORABLE
                 and row["alias_primary_source_uid"] == row["source_uid"]]
    by_structure: dict[str, list[dict[str, str]]] = {}
    for row in primaries:
        by_structure.setdefault(row["structure_name"], []).append(row)
    bindings = []
    for name, group in sorted(by_structure.items()):
        path, digest = files[name]
        require(file_sha(path) == digest == group[0]["structure_sha256"], f"{name}: structure differs from preparation")
        structure = parse_structure_file(str(path), structure_id=path.stem)
        for row in group:
            candidate, reason = prep.resolve_target(structure, row["chain"], int(row["resseq"]))
            require(candidate is not None, f"{row['source_uid']}: no longer resolves ({reason})")
            key = prep.ion_key(digest, 0, row["chain"], int(row["resseq"]), candidate.icode,
                               candidate.altloc_coords[0][0])
            require(key == row["ion_key"], f"{row['source_uid']}: resolves to another ion than prepared")
            bindings.append(CohortBinding(
                source_uid=row["source_uid"], source_row=int(row["source_row"]), pdbid=row["pdbid"],
                group_id=row["pdbid"], structure_name=name, structure_sha256=digest, model_index=0,
                chain=row["chain"], resseq=int(row["resseq"]), icode=row["icode"], resname=candidate.resname,
                atom_name=candidate.atom_name, altloc=row["altloc"],
                coord=(float(row["coord_x"]), float(row["coord_y"]), float(row["coord_z"])),
                native_element=candidate.symbol, physical_ion_id=row["ion_key"], alias_source_uids=()))
    return bindings


PREDICTION_COLUMNS = ("source_uid", "source_row", "pdbid", "disposition", "reason", "link_flagged",
                      "alias_primary_source_uid", "clean_subset", "checkpoint_sha256", "epochs", "pred_common4",
                      "p_common4_Mn", "p_common4_Cu", "p_common4_Zn", "p_common4_Class_VIII")


def predict_rows(rows: list[dict[str, str]], *, structure_root: Path, checkpoint_path: Path, esm_dir: Path | None,
                 external_root: Path | None, device: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One prediction row per prepared source row: aliases share their primary ion's prediction, unscorable rows carry
    none. Labels are dropped from the in-memory pockets before graphs are built."""
    import torch

    from training.campaign_runtime import load_campaign_prediction_components

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    epochs = len(checkpoint["history"])
    model, normalization, options = load_campaign_prediction_components(checkpoint, device=device)
    bindings = test_bindings(rows, structure_root)
    pockets = load_bound_pockets(bindings, structure_root, config, esm_dir=esm_dir, external_root=external_root)
    for pocket in pockets:
        pocket.y_metal = None  # the model never sees a label; graphs carry the missing-label marker
    logits = infer_logits(model, pockets, normalization, options, batch_size=int(config["batch_size"]), device=device)
    probabilities = common_four(logits.softmax(-1))
    by_primary = {binding.source_uid: probabilities[index] for index, binding in enumerate(bindings)}
    digest = file_sha(checkpoint_path)
    out = []
    for row in rows:
        record = {key: row[key] for key in ("source_uid", "source_row", "pdbid", "disposition", "reason",
                                            "link_flagged", "alias_primary_source_uid", "clean_subset")}
        record.update(checkpoint_sha256=digest, epochs=epochs, pred_common4="",
                      **{f"p_common4_{label.replace(' ', '_')}": "" for label in COMMON4})
        if row["disposition"] != prep.UNSCORABLE:
            vector = by_primary[row["alias_primary_source_uid"]]
            record["pred_common4"] = int(vector.argmax())
            for index, label in enumerate(COMMON4):
                record[f"p_common4_{label.replace(' ', '_')}"] = f"{float(vector[index]):.8f}"
        out.append(record)
    scored = sum(1 for row in rows if row["disposition"] != prep.UNSCORABLE)
    require(sum(1 for r in out if r["pred_common4"] != "") == scored, "A scored row has no prediction")
    return out, {"physical_ions_predicted": len(bindings), "scored_rows": scored, "rows": len(out),
                 "checkpoint_sha256": digest, "epochs": epochs}


def prediction_denied_paths(test_dir: Path, source_csv: Path | None) -> list[Path]:
    """Everything F4 must not open: every file of the test directory except the structure manifest (the site manifest
    and summary tables carry the labels), every file of the dataset root (crosswalk, coverage, split metadata) and the
    source CSV. The structures directory stays readable."""
    test_dir = Path(test_dir)
    named = [test_dir / "site_manifest.csv", test_dir / "final_data_summarazing_table.csv",
             test_dir / "final_data_summarazing_table_transition_metals_only_catalytic.csv",
             test_dir / "structure_chain_manifest.csv", test_dir.parent / "site_crosswalk.csv"]
    listed = [path for directory in (test_dir, test_dir.parent) for path in sorted(directory.iterdir())
              if not path.is_dir() and path.name != "structure_manifest.csv"]
    return sorted(set(named + listed + ([Path(source_csv)] if source_csv is not None else [])))


def predict(*, campaign_dir: Path, prepared_dir: Path, checkpoint_path: Path, test_dir: Path, esm_dir: Path,
            external_root: Path | None, out_dir: Path, device: str, source_csv: Path | None,
            log_entry: str) -> dict[str, Any]:
    """F4, once per campaign, after the user's logged go-ahead (``log_entry``, recorded in the claim): under the
    recorded Freeze 1 (this code) and Freeze 2 (these memberships and embeddings, verified by SHA-256) with the bound
    checkpoint; the write-once claim is made before any test structure is opened or any prediction is made, so a
    second pass is refused whatever its output directory."""
    from training.access_guard import install_forbidden_read_guard

    require(prep.LOG_ENTRY.fullmatch(str(log_entry)) is not None,
            "Name the log entry of the user's go-ahead for the single test pass (--log-entry v3-NNN)")
    out_dir = Path(out_dir)
    freeze1, freeze1_sha = prep.check_freeze1(campaign_dir, tools=test_side_tools())
    freeze2, freeze2_sha = prep.check_freeze2(campaign_dir, freeze1_sha256=freeze1_sha, prepared_dir=prepared_dir,
                                              esm_dir=esm_dir)
    checkpoint_sha = file_sha(checkpoint_path)
    require(checkpoint_sha == freeze2["checkpoint_sha256"] == freeze1["checkpoint_sha256"],
            "The checkpoint is not the bound refit")
    claim_path = prep.evidence_paths(campaign_dir)["prediction"]
    require(not claim_path.exists(), f"The single prediction pass was already claimed ({claim_path}); a second pass "
                                     "is refused campaign-wide")
    require(not out_dir.exists(), f"{out_dir} exists; the claim is made only for a new directory")
    _, claim_sha = prep.claim_prediction(campaign_dir, {
        "user_go_ahead_log_entry": log_entry, "freeze1_sha256": freeze1_sha, "freeze2_sha256": freeze2_sha,
        "checkpoint_sha256": checkpoint_sha,
        "out_dir": str(out_dir.resolve()), "device": device, "tool_sha256": {k: file_sha(v) for k, v in
                                                                             test_side_tools().items()},
        "claimed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    install_forbidden_read_guard([str(path) for path in prediction_denied_paths(test_dir, source_csv)])
    for name in ("DEEPMZYME_GRAPH_CACHE_DIR", "DEEPMZYME_PARSE_CACHE_DIR"):  # test graphs never enter a cache
        os.environ.pop(name, None)
    rows = read_prepared(prepared_dir)
    with tempfile.TemporaryDirectory(prefix="deepmzyme_v3_no_external_features_") as empty:
        external_root = Path(empty) if external_root is None else Path(external_root)
        require(external_root.is_dir() and not any(external_root.iterdir()),
                "External features stay absent, as in training (an empty directory)")
        predictions, summary = predict_rows(rows, structure_root=test_dir, checkpoint_path=checkpoint_path,
                                            esm_dir=esm_dir, external_root=external_root, device=device)
    out_dir.mkdir(parents=True)
    path = out_dir / "test_predictions.csv"
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PREDICTION_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(predictions)
    receipt = {**summary, "predictions_sha256": file_sha(path), "freeze1_sha256": freeze1_sha,
               "freeze2_sha256": freeze2_sha, "prediction_claim_sha256": claim_sha, "labels_read": False,
               "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out_dir / "prediction_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--action", required=True, choices=("plan-refit", "refit", "reload-check", "prepare-test",
                                                            "test-embeddings", "predict"))
    parser.add_argument("--campaign-dir", type=Path, help="the campaign root; the test-side actions read the freeze "
                                                         "records and the prediction claim only under its "
                                                         f"{prep.EVIDENCE_DIR}/")
    parser.add_argument("--unit", help="refit: the refit run name (must equal the specified one)")
    parser.add_argument("--epochs", type=int, help="refit: must equal the selection record's epochs (50)")
    parser.add_argument("--selection-sha256", help="refit: SHA-256 of the step F selection record")
    parser.add_argument("--train-dir", type=Path)
    parser.add_argument("--esm-dir", type=Path)
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load-workers", type=int)
    parser.add_argument("--run-dir", type=Path, help="reload-check: the completed refit directory")
    parser.add_argument("--log-entry", help="predict: the log entry of the user's go-ahead for the single test pass")
    parser.add_argument("--test-dir", type=Path)
    parser.add_argument("--source-csv", type=Path)
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--previously-opened-dir", type=Path, help="prepare-test: test side of the opened set")
    parser.add_argument("--v2-root", type=Path, help="prepare-test: frozen v2 context root (training cohort and "
                                                     "the A1 reference chains)")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--session-id")
    for name in ("execution-deadline", "allocation-started", "execution-max-seconds", "estimated-fit-seconds"):
        parser.add_argument("--" + name, type=float)
    parser.add_argument("--durable-root", type=Path)
    parser.add_argument("--persistence-mode", choices=("mounted", "host_pull"))
    return parser.parse_args(argv)


STRUCTURE_SUFFIXES = (".pdb", ".cif", ".ent", ".mmcif", ".pdb.gz", ".cif.gz", ".ent.gz")


def previously_opened_ids(directory: Path, *, expected_files: int) -> set[str]:
    """Lowercase PDB IDs (first four characters of each structure file name) of the opened test set; file names
    only, never contents. The file count must equal the DATASETS ledger's count recorded in Freeze 1."""
    import re

    names = sorted(path.name for path in Path(directory).rglob("*")
                   if path.is_file() and path.name.lower().endswith(STRUCTURE_SUFFIXES))
    require(len(names) == expected_files, f"{directory}: {len(names)} structure files, not {expected_files}")
    ids = {name[:4].lower() for name in names}
    require(all(re.fullmatch(r"[0-9][a-z0-9]{3}", pdbid) for pdbid in ids), "A file name does not start with a PDB ID")
    return ids


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.action in ("plan-refit", "refit", "reload-check"):
        require(args.campaign_dir is not None and args.train_dir is not None, f"{args.action} needs --campaign-dir "
                                                                             "and --train-dir")
        paths = v3.V3Paths(args.campaign_dir)
        epochs = int(SELECTION["epochs"]) if args.epochs is None else args.epochs
        require(epochs == int(SELECTION["epochs"]), f"The bound schedule is {SELECTION['epochs']} epochs")
        common = dict(epochs=epochs, selection_sha256=args.selection_sha256 or "",
                      python_bin=args.python_bin, train_dir=args.train_dir.resolve(),
                      esm_dir=args.esm_dir.resolve() if args.esm_dir else None, device=args.device)
        if args.action == "plan-refit":
            result = refit_differences(paths, lane=args.lane, load_workers=args.load_workers, **common)
        elif args.action == "reload-check":
            require(args.run_dir is not None, "reload-check needs --run-dir")
            identity = json.loads((args.run_dir / "run_metadata.json").read_text())["campaign_run_identity"]
            result = reload_check(args.run_dir, paths, identity, epochs=epochs, train_dir=common["train_dir"],
                                  esm_dir=common["esm_dir"], external_root=paths.empty_external_features,
                                  device=args.device)
        else:
            missing = ["--" + key.replace("_", "-") for key in RUNTIME_FIELDS if getattr(args, key) is None]
            require(not missing and args.unit == REFIT_NAME and args.selection_sha256,
                    "refit needs --unit (the refit name), --selection-sha256 and the allocation fields: "
                    + ", ".join(missing))
            policy = {"session_id": args.session_id, "deadline_unix": args.execution_deadline,
                      "allocation_started_unix": args.allocation_started,
                      "max_total_seconds": args.execution_max_seconds,
                      "estimated_fit_seconds": args.estimated_fit_seconds,
                      "durable_root": str(args.durable_root.resolve()), "persistence_mode": args.persistence_mode}
            result = run_refit(paths, lane=args.lane, execution_policy=policy, load_workers=args.load_workers,
                               **{k: v for k, v in common.items()})
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0 if result.get("status", "completed") == "completed" else 1
    require(args.campaign_dir is not None, f"{args.action} needs --campaign-dir (the freeze records' campaign root)")
    if args.action == "prepare-test":
        record, freeze1_sha = require_preparation_window(args.campaign_dir)
        from training.source_cohort import read_cohort_csv

        require(None not in (args.test_dir, args.source_csv, args.previously_opened_dir, args.v2_root, args.train_dir,
                             args.out_dir), "prepare-test needs --test-dir, --source-csv, --previously-opened-dir, "
                                            "--v2-root, --train-dir and --out-dir")
        training = read_cohort_csv(args.v2_root / "train_cohort.csv", expected_sha256=record["cohort_sha256"])
        opened = previously_opened_ids(args.previously_opened_dir, expected_files=record["previously_opened_files"])

        def training_sequences() -> set[str]:  # read after the test rows: it installs the held-out read guard
            from benchmarking.v3_folds import load_inputs

            inputs = load_inputs(args.v2_root, args.train_dir, workers=args.workers)
            require(inputs["inputs"]["cohort_sha256"] == record["cohort_sha256"],
                    "The v2 context root's cohort is not the refit's training cohort")
            return {seq for group in inputs["pdb_sequences"].values() for seq in group}

        result = prep.prepare(args.test_dir, args.source_csv, args.out_dir,
                              expected_source_sha256=record["test_source_csv_sha256"],
                              training_pdbids={b.pdbid for b in training}, training_sequences=training_sequences,
                              previously_opened=opened, workers=args.workers, freeze1_sha256=freeze1_sha)
    elif args.action == "test-embeddings":
        _, freeze1_sha = require_preparation_window(args.campaign_dir)
        require(args.device == prep.TEST_ESM["device"], f"test-embeddings runs on the workstation CPU only "
                                                        f"(--device {prep.TEST_ESM['device']}; spec test_esm)")
        summary = json.loads((args.prepared_dir / "preparation_summary.json").read_text())
        require(summary.get("freeze1_sha256") == freeze1_sha
                and prep.sha256_file(args.prepared_dir / "test_esm_plan.csv") == summary["files_sha256"]["test_esm_plan.csv"],
                "The ESMC plan is not the one prepared under this Freeze 1")
        with (args.prepared_dir / "test_esm_plan.csv").open(encoding="utf-8", newline="") as handle:
            plan = list(csv.DictReader(handle))
        result = prep.generate_embeddings(plan, args.test_dir, args.out_dir, device=args.device,
                                          freeze1_sha256=freeze1_sha)
    else:
        require(args.source_csv is not None, "predict needs --source-csv (it is denied to the pass by the read guard)")
        result = predict(campaign_dir=args.campaign_dir, prepared_dir=args.prepared_dir,
                         checkpoint_path=args.checkpoint, test_dir=args.test_dir, esm_dir=args.esm_dir,
                         external_root=None, out_dir=args.out_dir, device=args.device, source_csv=args.source_csv,
                         log_entry=args.log_entry)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    from benchmarking.pmm_execution import ExecutionBlocked, PersistenceError

    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"v3 step F refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except ExecutionBlocked as exc:
        print(f"v3 step F blocked: {exc}", file=sys.stderr)
        raise SystemExit(3)
    except PersistenceError as exc:
        print(f"v3 step F persistence failed: {exc}", file=sys.stderr)
        raise SystemExit(4)
