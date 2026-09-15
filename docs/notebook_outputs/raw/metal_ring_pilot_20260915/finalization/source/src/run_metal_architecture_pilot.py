"""Bounded metal architecture pilot using the notebook's canonical CLI commands.

Planning is CPU-only. Execution is serial, requires persistent storage, and
counts the allocated runtime (including setup and transfers) against ten hours.
Each execute call runs at most one attempt so an external orchestrator can
archive and verify its outputs before advancing on an ephemeral Colab VM.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
from dataclasses import asdict
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time

PROFILE = "metal_architecture_pilot_10h_v1"
DATASET = "train_and_test_sets_structures_non_overlapped_pinmymetal"
BUNDLE = "DeepMzyme_Data_v12_manifest_exact_common70_nonoverlap_clean30_care30_complete_esm_ring_external.tar.gz"
BUNDLE_SHA = "90c0899829e0ac5ca94a5ef34484b74ca1014d3e9b6b1fba90bd338ee3440dee"
METRIC = "val_metal_balanced_acc"
CORE = ("Only-GVP", "Only-ESM", "GVP + late fusion")
EARLY = "GVP + early fusion"
HYBRID = "GVP + hybrid fusion"
FAMILIES = (*CORE, EARLY, HYBRID)
SCHEMES = ("four_class", "five_class", "six_class")
LRS = (3e-5, 1e-4)
LIMITS = dict(total_minutes=600, training_deadline_minutes=570, profile_minutes=60,
              main_minutes=450, retry_minutes=60, closeout_minutes=30, admission_factor=1.25)
BLOCKS = ("S", "A1", "A2", "T1", "T2", "H", "R", "RE", "RH")


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else default


def require(condition, message):
    if not condition:
        raise ValueError(message)


def csv_save(path, rows, fields):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def replace_option(command, flag, value):
    command = list(command)
    if flag in command:
        command[command.index(flag) + 1] = str(value)
    else:
        command.extend((flag, str(value)))
    return command


def parse_config(command):
    from training.config import parse_args
    return parse_args(list(map(str, command[2:])))


def validate(config, epochs=50):
    from training.run import validate_training_configuration
    validate_training_configuration(config)
    checks = {
        "task": config.task == "metal", "epochs": config.epochs == epochs,
        "batch": config.batch_size == 8, "seed": config.seed in (42, 43),
        "split": config.split_seed == 42 and config.train_val_split_by == "pdbid"
                 and config.val_fraction == .15 and config.split_stratify_by == "metal_site",
        "native selection": config.selection_metric == METRIC,
        "eligible cohort": config.metal_eligibility_scheme in ("six_class", "split_all_metals"),
        "learning rate": config.learning_rate in LRS and config.weight_decay == 1e-4,
        "features": config.node_feature_set == "conservative" and not config.omit_node_features,
        "pooling": config.edge_radius == 6 and config.classifier_pool_distance_cutoff == 0
                   and config.metal_node_mode == "none" and config.structural_readout_scope == "residue_only",
        "coverage": config.require_all_task_classes and config.require_external_features,
        "loss": config.metal_class_weight_mode == "inverse_frequency"
                and config.metal_loss_function == "cross_entropy" and config.metal_collapsed_loss_weight == 0
                and not config.balance_metal_site_symbols and config.metal_label_smoothing == 0,
        "no augmentation": config.position_noise_std == config.second_shell_dropout == config.outer_residue_dropout == 0,
        "no RING": not config.use_ring_edges and not config.prepare_missing_ring_edges,
        "no preparation": not config.prepare_missing_esm_embeddings,
        "no held-out": not config.run_test_eval and config.test_structure_dir is None and config.test_summary_csv is None,
        "fixed schedule": config.lr_schedule == "fixed" and config.deterministic,
    }
    require(all(checks.values()), "Pilot configuration mismatch: " + ", ".join(k for k, v in checks.items() if not v))


def validate_feature_overlay(data, external_root, manifest_path):
    """Check a completed overlay by structure identity, never host-specific paths."""
    from structure_store import read_structure_manifest
    report = read(manifest_path)
    require(report and report.get("status") == "complete" and not report.get("failures"),
            "External feature overlay is not complete and failure-free.")
    require(report.get("dataset") == DATASET and report.get("split") == "train"
            and report.get("test_evaluation") is False, "External overlay has the wrong scientific cohort.")
    references = read_structure_manifest(Path(data) / DATASET / "train")
    entries = report.get("files", [])
    by_name = {row["structure"]: row for row in entries}
    require(len(entries) == len(by_name) == len(references)
            and report.get("total_structures") == report.get("selected_structures") == len(references)
            and set(by_name) == {ref.structure_name for ref in references},
            "External overlay manifest does not cover the complete training cohort exactly once.")
    for ref in references:
        entry = by_name[ref.structure_name]
        # The repair tool writes one structure-stem directory per input. Its
        # manifest retains host paths as provenance; those are not portable.
        path = Path(external_root) / Path(ref.structure_name).stem / "residue_features.json"
        require(entry.get("structure_sha256") == ref.sha256, f"Overlay structure checksum mismatch: {ref.structure_name}")
        require(path.is_file() and digest(path) == entry.get("sha256"), f"Overlay feature checksum mismatch: {ref.structure_name}")
    return report


def _templates(root, data, output, external_features_root_dir=None):
    notebook = read(root / "notebooks/DeepMzyme_training_colab.ipynb")
    cells = {c["id"]: "".join(c["source"]) for c in notebook["cells"]}
    playbook = (root / "docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md").read_text()
    section = playbook.split("### Exact standalone notebook block", 1)[1]
    recipe = re.search(r"```python\n(.*?)```", section, re.S).group(1)
    train = data / DATASET / "train"
    templates = {}
    for family in FAMILIES:
        ns = {"IN_COLAB": False, "NOTEBOOK_START_CWD": root, "Path": Path}
        # Notebook code prints large configuration tables. Save only our resolved
        # commands/configuration in the manifest; errors still propagate normally.
        with contextlib.redirect_stdout(io.StringIO()):
            exec(cells["eb4db512"], ns)
            exec(recipe, ns)
            ns.update(MODEL_PRESET=family, TASK="metal", RUN_MODE="manual_configurations",
                      RECOMMENDED_RUN_SET="custom", DATASET_NAME=DATASET,
                      SEEDS_CSV="42", LEARNING_RATES_CSV="3e-5,1e-4", MAX_CONFIGURATION_RUNS=2,
                      EPOCHS=50, SINGLE_AND_MANUAL_CONFIG_EPOCHS=50, METAL_LABEL_SCHEME="four_class",
                      METAL_ELIGIBILITY_SCHEME="six_class", SELECTION_METRIC=METRIC,
                      OPTUNA_SELECTION_METRIC=METRIC, RUN_BATCH_ID=PROFILE, RUN_NAME_PREFIX=PROFILE,
                      REPO_ROOT=str(root), RUNS_DIR=str(output / "runs"),
                      ESM_EMBEDDINGS_DIR=str(data / "esm_embeddings"),
                      EXTERNAL_FEATURES_ROOT_DIR=str(external_features_root_dir or data / "updated_feature_extraction"),
                      RING_FEATURES_DIR=str(data / "RING_features"),
                      BUNDLE_FILENAME=BUNDLE, BUNDLE_SHA256=BUNDLE_SHA,
                      MOUNT_DRIVE=False, COPY_OUTPUTS_TO_DRIVE=False, DEVICE="cuda")
            exec(cells["ba89d9f5"], ns)
            ns.update(REPO_DIR=root, SRC_DIR=root / "src", TRAIN_DIR=train,
                      TRAIN_SITE_SUMMARY_CSV=train / "final_data_summarazing_table_transition_metals_only_catalytic.csv",
                      TEST_DIR=output / "unused_test", TEST_SITE_SUMMARY_CSV=output / "unused_test.csv",
                      TRAIN_STRUCTURES=[], TEST_STRUCTURES=[], DATA_ROOT=data, DATASET_ROOT=data / DATASET,
                      DRIVE_ROOT_PATH=output / "unused_drive_root",
                      DRIVE_DATA_DIR=output / "unused_drive_data")
            exec(cells["75e1f97ec96a047f"], ns)
        require(len(ns["planned_runs"]) == 2, f"Unexpected notebook expansion for {family}")
        templates[family] = ns["planned_runs"][0]
    return templates


def plan(root, data, output, source_commit, external_features_root_dir=None, feature_overlay_manifest=None):
    """Write an immutable CPU-only preview; no allocation or persistence receipt needed."""
    root, data, output = map(lambda p: Path(p).resolve(), (root, data, output))
    external_root = Path(external_features_root_dir).resolve() if external_features_root_dir else data / "updated_feature_extraction"
    if feature_overlay_manifest:
        validate_feature_overlay(data, external_root, feature_overlay_manifest)
    templates = _templates(root, data, output, external_root)
    runs = []

    def add(block, family, scheme="four_class", lr=3e-5, seed=42, epochs=50):
        ident = f"{block}_{FAMILIES.index(family)}_{scheme}_lr{lr:g}_s{seed}"
        command = list(map(str, templates[family]["command"]))
        command[0] = sys.executable
        directory = output / "runs" / ident
        for flag, value in (("--metal-label-scheme", scheme), ("--selection-metric", METRIC),
                            ("--epochs", epochs), ("--learning-rate", lr), ("--seed", seed),
                            # Only-ESM's notebook command omits irrelevant GVP
                            # options; keep graph construction identical anyway.
                            ("--edge-radius", 6), ("--classifier-pool-distance-cutoff", 0),
                            ("--run-name", ident), ("--runs-dir", output / "runs")):
            command = replace_option(command, flag, value)
        config = parse_config(command)
        validate(config, epochs)
        payload = json.loads(json.dumps(asdict(config), default=str))
        runs.append(dict(id=ident, block=block, family=family, scheme=scheme, lr=lr, seed=seed,
                         epochs=epochs, command=command, run_dir=str(directory),
                         env={**templates[family].get("env", {}), "DEEPGM_METAL_LABEL_SCHEME": scheme},
                         config=payload, config_sha256=fingerprint(payload)))

    for scheme in SCHEMES:
        add("S", CORE[0], scheme, epochs=1)
    for family in (CORE[1], EARLY, CORE[2], HYBRID):
        add("S", family, epochs=1)
    for i, lr in enumerate(LRS, 1):
        for family in (CORE[0], CORE[1], EARLY, CORE[2]):
            add(f"A{i}", family, lr=lr)
    for i, lr in enumerate(LRS, 1):
        for scheme in SCHEMES[1:]:
            for family in CORE:
                add(f"T{i}", family, scheme, lr)
    for lr in LRS:
        add("H", HYBRID, lr=lr)
    # Both LR templates are immutable; only one per family/target is activated
    # for seed 43 after the seed-42 results are frozen.
    for scheme in SCHEMES:
        for family in CORE:
            for lr in LRS:
                add("R", family, scheme, lr, seed=43)
    for block, family in (("RE", EARLY), ("RH", HYBRID)):
        for lr in LRS:
            add(block, family, lr=lr, seed=43)
    source_paths = sorted((root / "src").rglob("*.py")) + [
        root / "notebooks/DeepMzyme_training_colab.ipynb", root / "docs/METAL_TRAINING_PIPELINE_PLAYBOOK.md"]
    train = data / DATASET / "train"
    dataset_paths = [train / "structure_manifest.csv", train / "final_data_summarazing_table_transition_metals_only_catalytic.csv"]
    manifest = dict(profile=PROFILE, source_commit=source_commit, data_root=str(data), output_dir=str(output),
                    dataset=DATASET, bundle=BUNDLE, bundle_sha256=BUNDLE_SHA, limits=LIMITS,
                    metric=METRIC, held_out_evaluation=False, block_order=list(BLOCKS),
                    source_files={str(p.relative_to(root)): digest(p) for p in source_paths},
                    dataset_files={str(p.relative_to(data)): digest(p) for p in dataset_paths if p.is_file()},
                    runs=runs, maximum_smokes=7, maximum_main_runs=33,
                    external_features_root_dir=str(external_root),
                    feature_overlay_manifest=({"path": str(Path(feature_overlay_manifest).resolve()),
                                               "sha256": digest(feature_overlay_manifest)}
                                              if feature_overlay_manifest else None),
                    optional_hybrid_gate=dict(early_minus_gvp=.01, minimum_recall_degradation=.03,
                                              interpretation="exploratory_prioritization_not_promotion"))
    existing = read(output / "campaign_manifest.json")
    require(existing is None or existing == manifest, "Existing pilot manifest differs; use a new output directory.")
    save(output / "campaign_manifest.json", manifest)
    (output / "commands.txt").write_text("\n\n".join(shlex.join(r["command"]) for r in runs) + "\n")
    csv_save(output / "run_matrix.csv", runs, ["id", "block", "family", "scheme", "lr", "seed", "epochs", "run_dir"])
    return manifest


def storage_check(output, receipt_path=None):
    output = Path(output).resolve()
    mount = Path("/content/drive/MyDrive")
    if mount.is_dir() and output.is_relative_to(mount.resolve()):
        receipt = {"method": "mounted_drive", "root": str(mount)}
    else:
        require(receipt_path is not None, "Ephemeral output requires an explicit verified persistence receipt.")
        receipt = read(receipt_path)
        require(receipt and receipt.get("method") == "verified_archive_transfer"
                and receipt.get("drive_probe_verified") and receipt.get("local_write_verified")
                and receipt.get("drive_folder_id"), "Persistence receipt is incomplete or unverified.")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w+", dir=output) as stream:
        stream.write(PROFILE)
        stream.flush()
        stream.seek(0)
        require(stream.read() == PROFILE, "Output write/read check failed.")
    old = read(output / "persistent_storage.json")
    require(old is None or old == receipt, "Persistent storage identity changed during the campaign.")
    save(output / "persistent_storage.json", receipt)
    return receipt


def verify_manifest(root, output):
    manifest = read(output / "campaign_manifest.json")
    require(manifest and manifest["profile"] == PROFILE, "Missing pilot manifest.")
    for name, expected in manifest["source_files"].items():
        require(digest(root / name) == expected, f"Source changed after plan: {name}")
    for name, expected in manifest["dataset_files"].items():
        require(digest(Path(manifest["data_root"]) / name) == expected, f"Dataset changed after plan: {name}")
    overlay = manifest.get("feature_overlay_manifest")
    if overlay:
        require(digest(overlay["path"]) == overlay["sha256"], "External feature overlay manifest changed.")
        validate_feature_overlay(manifest["data_root"], manifest["external_features_root_dir"], overlay["path"])
    return manifest


def allocation_elapsed(output, started, now=None):
    now = time.time() if now is None else now
    require(math.isfinite(started) and 0 < started <= now, "Invalid allocation start time.")
    path = output / "allocation_ledger.json"
    ledger = read(path, {"intervals": []})
    intervals = ledger["intervals"]
    if not intervals or intervals[-1]["started_epoch"] != started:
        require(not intervals or "ended_epoch" in intervals[-1],
                "Previous allocation has no recorded close; close it before starting another allocation.")
        intervals.append({"started_epoch": started})
    require("ended_epoch" not in intervals[-1], "This allocation was already closed.")
    intervals[-1]["observed_epoch"] = now
    elapsed = sum(interval.get("ended_epoch", now) - interval["started_epoch"] for interval in intervals)
    save(path, ledger)
    return elapsed


def close_allocation(output, ended_epoch=None):
    ledger = read(output / "allocation_ledger.json")
    require(ledger and ledger["intervals"], "No allocation to close.")
    interval = ledger["intervals"][-1]
    end = time.time() if ended_epoch is None else ended_epoch
    require(end >= interval["observed_epoch"], "Allocation cannot end before its latest observation.")
    interval["ended_epoch"] = end
    save(output / "allocation_ledger.json", ledger)


def cohort_identity(summary):
    # Native targets intentionally differ. Membership/order/group identity must not.
    identity = {part: [{key: row[key] for key in ("structure_id", "pocket_id", "group")}
                       for row in summary["retained_split_identity"][part]["examples"]]
                for part in ("train", "validation")}
    require(identity["train"] and identity["validation"], "Empty train/validation cohort.")
    require(not ({x["group"] for x in identity["train"]} & {x["group"] for x in identity["validation"]}),
            "Train/validation PDB overlap.")
    return identity


def completed_result(run, directory, reference):
    directory = Path(directory)
    for name in ("run_metadata.json", "run_config.json", "dataset_summary.json", "epoch_metrics.csv",
                 "best_model_checkpoint.pt", "last_model_checkpoint.pt"):
        require((directory / name).is_file() and (directory / name).stat().st_size > 0,
                f"Incomplete run {run['id']}: {name}")
    meta = read(directory / "run_metadata.json")
    config = read(directory / "run_config.json")
    if "config" in run:
        for key, expected in run["config"].items():
            if key in ("run_name", "runs_dir"):
                continue  # A linked retry has a separate output directory.
            require(config.get("config", {}).get(key) == expected, f"Saved configuration mismatch: {key}")
        require(meta.get("selection_metric") == METRIC, "Saved checkpoint selection rule is not native validation.")
    require(meta.get("test_report") is None and not (directory / "test_report.json").exists(), "Held-out result present.")
    history = config["history"]
    expected_epochs = list(range(1, run["epochs"] + 1))
    with (directory / "epoch_metrics.csv").open() as stream:
        epochs = [int(row["epoch"]) for row in csv.DictReader(stream)]
    require(epochs == expected_epochs and [int(row["epoch"]) for row in history] == expected_epochs,
            f"Incomplete or duplicated epochs: {run['id']}")
    require(cohort_identity(read(directory / "dataset_summary.json")) == cohort_identity(reference),
            f"Retained cohort mismatch: {run['id']}")
    selected = next((row for row in history if row["epoch"] == meta["selected_checkpoint_epoch"]), None)
    require(selected is not None, "Selected checkpoint epoch is absent from history.")
    scores = [float(row[METRIC]) for row in history]
    require(all(math.isfinite(x) for x in scores), "Nonfinite native validation score.")
    require(math.isclose(float(selected[METRIC]), max(scores), abs_tol=1e-10),
            "Checkpoint was not selected by the maximum native validation metric.")
    recalls = selected["val_metal_per_class_recall"]
    collapsed_recalls = selected.get("val_metal_collapsed4_per_class_recall", recalls)
    collapsed_score = selected.get("val_metal_collapsed4_balanced_acc", selected[METRIC])
    require(run["scheme"] == "four_class" or "val_metal_collapsed4_balanced_acc" in selected,
            "Missing collapsed-four reporting for five/six-class training.")
    return {**{key: run[key] for key in ("id", "block", "family", "scheme", "lr", "seed")},
            "selected_epoch": selected["epoch"], "balanced_accuracy": float(selected[METRIC]),
            "collapsed4_balanced_accuracy": float(collapsed_score),
            "minimum_recall": min(float(x) for x in recalls.values() if x is not None),
            "collapsed4_minimum_recall": min(float(x) for x in collapsed_recalls.values() if x is not None),
            "per_class_recall": recalls, "collapsed4_per_class_recall": collapsed_recalls,
            "run_dir": str(directory)}


def _attempts(output):
    return read(output / "campaign_attempt_ledger.json", [])


def _save_attempts(output, attempts):
    save(output / "campaign_attempt_ledger.json", attempts)
    csv_save(output / "campaign_attempt_ledger.csv", attempts,
             ["attempt_id", "run_id", "block", "status", "started_epoch", "ended_epoch", "elapsed_seconds",
              "setup_seconds", "epoch_seconds", "gpu_peak_used_mib", "retry_of", "returncode", "run_dir", "log", "error"])


def completed_rows(output, manifest=None):
    manifest = manifest or read(output / "campaign_manifest.json")
    reference = read(output / "expected_split.json")
    by_id = {row["id"]: row for row in manifest["runs"]}
    results = []
    for attempt in _attempts(output):
        if attempt["status"] == "completed" and attempt["run_id"] in by_id:
            row = completed_result(by_id[attempt["run_id"]], attempt["run_dir"], reference)
            row["elapsed_seconds"] = attempt["elapsed_seconds"]
            results.append(row)
    return results


def hybrid_gate(rows):
    def best(family):
        candidates = [r for r in rows if r["family"] == family and r["scheme"] == "four_class"
                      and r["seed"] == 42 and r["block"] in ("A1", "A2")]
        return max(candidates, key=lambda r: (r["balanced_accuracy"], r["minimum_recall"], -r["lr"])) if len(candidates) == 2 else None
    gvp, early = best(CORE[0]), best(EARLY)
    if gvp is None or early is None:
        return {"status": "pending", "interpretation": "exploratory_prioritization_not_promotion"}
    delta = early["balanced_accuracy"] - gvp["balanced_accuracy"]
    recall_delta = early["minimum_recall"] - gvp["minimum_recall"]
    return dict(status="prioritized" if delta >= .01 - 1e-12 and recall_delta >= -.03 - 1e-12 else "deferred",
                balanced_accuracy_delta=delta, minimum_recall_delta=recall_delta,
                interpretation="exploratory_prioritization_not_promotion")


def active_block_runs(manifest, block, rows):
    runs = [r for r in manifest["runs"] if r["block"] == block]
    if block in ("H", "RH") and hybrid_gate(rows)["status"] != "prioritized":
        return []
    if block in ("R", "RE", "RH"):
        selected = []
        for family, scheme in dict.fromkeys((r["family"], r["scheme"]) for r in runs):
            previous = [r for r in rows if r["family"] == family and r["scheme"] == scheme
                        and r["seed"] == 42 and r["block"] != "S"]
            require(len(previous) == 2, "Seed repeat requires both completed learning-rate runs.")
            best = max(previous, key=lambda r: (r["balanced_accuracy"], r["minimum_recall"], -r["lr"]))
            selected.append(next(r for r in runs if r["family"] == family and r["scheme"] == scheme and r["lr"] == best["lr"]))
        return selected
    return runs


def next_block(manifest, rows):
    complete = {r["id"] for r in rows}
    for block in manifest["block_order"]:
        runs = active_block_runs(manifest, block, rows)
        pending = [r for r in runs if r["id"] not in complete]
        if pending:
            return block, runs, pending
    return None, [], []


def forecast_seconds(run, attempts, manifest):
    lookup = {r["id"]: r for r in manifest["runs"]}
    measurements = []
    for attempt in attempts:
        source = lookup.get(attempt["run_id"])
        if attempt["status"] != "completed" or source is None or source["family"] != run["family"]:
            continue
        if "epoch_seconds" in attempt:
            measurements.append(attempt["setup_seconds"] + attempt["epoch_seconds"] * run["epochs"])
    require(measurements, f"No measured profile for {run['family']}")
    # Use the slowest same-family measurement, including scheme-specific GVP
    # smokes. The margin applies again at whole-block admission.
    return max(measurements)


def budget_available(elapsed, attempts, block):
    category = "profile" if block in ("S", "P") else "main"
    used = sum(a.get("elapsed_seconds", 0) for a in attempts
               if not a.get("retry_of") and ((a["block"] in ("S", "P")) == (category == "profile")))
    if category == "profile":
        # Provisioning, downloads, and between-smoke transfers are part of the
        # first-hour readiness allocation, even when performed by the caller.
        used = max(used, elapsed)
    return max(0, min(LIMITS["training_deadline_minutes"] * 60 - elapsed,
                      LIMITS[category + "_minutes"] * 60 - used))


def _require_transfer(output, attempts, storage):
    if storage["method"] != "verified_archive_transfer" or not attempts:
        return
    previous = attempts[-1]
    receipt = read(output / "transfer_receipts" / f"{previous['attempt_id']}.json")
    require(receipt and receipt.get("drive_verified") and receipt.get("local_sha256_verified")
            and receipt.get("attempt_id") == previous["attempt_id"] and receipt.get("archive_sha256")
            and receipt.get("manifest_sha256") == digest(output / "campaign_manifest.json"),
            f"Verify and archive attempt {previous['attempt_id']} before advancing.")


def verify_training_cache(output):
    audit = read(output / "training_cache_audit.json")
    require(audit and not audit.get("failures"), "Missing successful training cache audit.")
    for entry in audit["files"]:
        path = Path(entry["path"])
        require(path.is_file(), f"Training feature disappeared: {path}")
        info = path.stat()
        require(info.st_size == entry["bytes"] and info.st_mtime_ns == entry["mtime_ns"],
                f"Training feature changed after its hash/coverage audit: {path}")


def _run_process(command, root, output, attempt, env, deadline):
    """Run a fresh process group; the deadline covers SIGTERM and SIGKILL grace."""
    log_path = Path(attempt["log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process = None
    old_term_handler = signal.getsignal(signal.SIGTERM)

    def on_term(signum, frame):
        raise InterruptedError("Orchestrator was terminated; stopping its training process group.")

    signal.signal(signal.SIGTERM, on_term)
    next_memory_sample = 0.0
    try:
        with log_path.open("w") as log:
            process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, env={**os.environ, **env, "PYTHONUNBUFFERED": "1"})
            attempt["pid"] = process.pid
            save(output / "active_process.json", attempt)
            while process.poll() is None:
                remaining = deadline - time.time()
                if remaining <= 15:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=max(.01, remaining - 1))
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                    attempt["status"] = "deadline_stopped"
                    break
                if time.time() >= next_memory_sample:
                    try:
                        memory = subprocess.run(
                            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=2, check=True)
                        measured = float(memory.stdout.strip().splitlines()[0])
                        attempt["gpu_peak_used_mib"] = max(attempt.get("gpu_peak_used_mib", 0), measured)
                        attempt["gpu_memory_measurement"] = "whole_device_peak_sampled_every_10_seconds"
                    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                        attempt["gpu_memory_measurement"] = "nvidia_smi_unavailable"
                    next_memory_sample = time.time() + 10
                time.sleep(min(2, remaining - 15))
            attempt["returncode"] = process.returncode
    except BaseException:
        if process and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        attempt["status"] = "interrupted"
        raise
    finally:
        attempt["ended_epoch"] = time.time()
        attempt["elapsed_seconds"] = attempt["ended_epoch"] - attempt["started_epoch"]
        save(output / "active_process.json", {"status": "idle"})
        signal.signal(signal.SIGTERM, old_term_handler)


def _record_attempt(root, output, run, command, env, available, attempts):
    prior = [a for a in attempts if a["run_id"] == run["id"]]
    require(len(prior) < 2, f"Retry allowance exhausted for {run['id']}")
    ident = f"attempt_{len(attempts) + 1:03d}"
    directory = Path(run["run_dir"])
    retry = bool(prior)
    if retry:
        directory = directory.with_name(directory.name + "_retry1")
        command = replace_option(command, "--preflight-run-name" if run["block"] == "P" else "--run-name", directory.name)
        retry_used = sum(a.get("elapsed_seconds", 0) for a in attempts if a.get("retry_of"))
        available = min(available, LIMITS["retry_minutes"] * 60 - retry_used)
    require(available > 20, "No attempt time remains.")
    require(not directory.exists() or run["block"] == "P", f"Untracked output directory exists: {directory}")
    attempt = dict(attempt_id=ident, run_id=run["id"], block=run["block"], status="running",
                   started_epoch=time.time(), retry_of=prior[-1]["attempt_id"] if retry else None,
                   run_dir=str(directory), log=str(output / "logs" / f"{ident}.log"), command=command)
    attempts.append(attempt)
    _save_attempts(output, attempts)
    try:
        _run_process(command, root, output, attempt, {k: str(v) for k, v in env.items()}, attempt["started_epoch"] + available)
        if attempt["status"] == "running":
            require(attempt["returncode"] == 0, f"Training process failed ({attempt['returncode']}); see {attempt['log']}")
            if run["block"] == "P":
                require(read(output / "readiness.json", {}).get("status") == "passed", "Readiness worker did not pass.")
            else:
                completed_result(run, directory, read(output / "expected_split.json"))
                ready_path = directory / "prepare_status.json"
                require(read(ready_path, {}).get("status") == "ready", "Missing successful preparation timing.")
                ready_time = ready_path.stat().st_mtime
                attempt["setup_seconds"] = max(0, ready_time - attempt["started_epoch"])
                attempt["epoch_seconds"] = max(.01, (attempt["ended_epoch"] - ready_time) / run["epochs"])
            attempt["status"] = "completed"
    except BaseException as exc:
        if attempt["status"] == "running":
            attempt["status"] = "failed"
        attempt["error"] = str(exc)
        raise
    finally:
        _save_attempts(output, attempts)
    return attempt


def _recover_interrupted(output, attempts):
    for attempt in attempts:
        if attempt["status"] == "running":
            active = read(output / "active_process.json", {})
            pid = active.get("pid")
            if pid:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    raise ValueError(f"Attempt still has live process {pid}; do not launch another worker.")
            attempt.update(status="interrupted", ended_epoch=time.time(),
                           elapsed_seconds=time.time() - attempt["started_epoch"],
                           error="Recovered after orchestrator interruption; restart requires a linked retry.")
    _save_attempts(output, attempts)


def preflight(root, output, allocation_started_epoch, persistence_receipt=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    manifest = verify_manifest(root, output)
    storage = storage_check(output, persistence_receipt)
    elapsed = allocation_elapsed(output, allocation_started_epoch)
    attempts = _attempts(output)
    _recover_interrupted(output, attempts)
    ready = read(output / "readiness.json")
    if ready and ready.get("status") == "passed":
        require(ready["manifest_sha256"] == digest(output / "campaign_manifest.json"), "Readiness manifest mismatch.")
        return ready
    _require_transfer(output, attempts, storage)
    command = [sys.executable, str(root / "src/run_metal_architecture_pilot.py"), "_preflight",
               "--output-dir", str(output)]
    run = dict(id="readiness", block="P", run_dir=str(output / "preflight_run"))
    _record_attempt(root, output, run, command, {"DEEPGM_METAL_LABEL_SCHEME": "six_class"},
                    budget_available(elapsed, attempts, "P"), attempts)
    return read(output / "readiness.json")


def _preflight_worker(root, output, run_name="preflight_run"):
    import torch
    from model import metal_distance_pool_mask, residue_node_mask
    from torch_geometric.data import Batch
    from training.run import prepare_run, set_seed
    from training.runtime_preparation import updated_external_feature_path_candidates
    from training.esm_feature_loading import embedding_path_candidates
    from structure_store import resolve_structure_files
    manifest = verify_manifest(root, output)
    data = Path(manifest["data_root"])
    bundle_receipt = read(Path("/content/v12_ready.json"))
    require(bundle_receipt and bundle_receipt.get("sha256") == BUNDLE_SHA, "Verified v12 bundle receipt is required.")
    require(len(manifest["dataset_files"]) == 2, "Training structure manifest and site summary must both be hashed.")
    require(torch.cuda.is_available(), "Pilot execution requires CUDA.")
    probe = torch.ones(32, device="cuda", requires_grad=True)
    probe.square().sum().backward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    train = data / DATASET / "train"
    external_root = Path(manifest["external_features_root_dir"])
    audit = []
    for path in resolve_structure_files(train, recursive_legacy_scan=False):
        candidates = updated_external_feature_path_candidates(path, structure_root=train,
                                                               external_features_root_dir=external_root)
        external = next((p for p in candidates if p.is_file()), None)
        require(external is not None, f"Missing external features for {path.name}")
        payload = read(external)
        require(payload.get("tooling", {}).get("pka") == "propka" and payload.get("residues"),
                f"Incomplete external-feature tooling for {path.name}; require computed PROPKA.")
        require(all(math.isfinite(float(v)) for r in payload["residues"] for v in r["features"].values()),
                f"Nonfinite external features: {external}")
        embeddings = [p for p in embedding_path_candidates(data / "esm_embeddings", path) if p.is_file()]
        require(embeddings, f"Missing ESMC embeddings for {path.name}")
        checked = [external]
        for embedding in embeddings:
            tensor = torch.load(embedding, map_location="cpu", weights_only=True)["embeddings"]
            metadata_path = Path(str(embedding) + ".json")
            metadata = read(metadata_path, {})
            require(tensor.ndim == 2 and tensor.shape[1] == 960 and torch.isfinite(tensor).all()
                    and metadata.get("esm_model_name") == "esmc_300m" and metadata.get("embedding_dim") == 960,
                    f"Invalid ESMC-300m cache: {embedding}")
            checked.extend((embedding, metadata_path))
        audit.extend({"path": str(p), "sha256": digest(p), "bytes": p.stat().st_size,
                      "mtime_ns": p.stat().st_mtime_ns} for p in checked)
    save(output / "training_cache_audit.json", dict(files=audit, failures=[], external_features_root_dir=str(external_root)))
    row = next(r for r in manifest["runs"] if r["block"] == "S" and r["family"] == CORE[2])
    command = replace_option(row["command"], "--runs-dir", output)
    command = replace_option(command, "--run-name", run_name)
    # Native-six readiness protects all six classes before any smoke epoch;
    # coarser schemes retain the same strict-six-eligible examples.
    command = replace_option(command, "--metal-label-scheme", "six_class")
    config = parse_config(command)
    validate(config, 1)
    set_seed(config.seed, deterministic=config.deterministic)
    prepared = prepare_run(config)
    cohort = cohort_identity(prepared.dataset_summary)
    rows = []
    for part, loader in (("train", prepared.train_loader), ("validation", prepared.val_loader)):
        require(loader is not None, "Missing validation loader.")
        for i in range(len(loader.dataset)):
            graph = loader.dataset[i]
            batch = Batch.from_data_list([graph])
            base = residue_node_mask(batch)
            mask = metal_distance_pool_mask(batch, 0., base_mask=base)
            extracted = int(base.sum())
            pooled = int(mask.sum())
            require(extracted > 0 and extracted == pooled, "Cutoff zero did not pool all extracted residues.")
            rows.append(dict(part=part, pocket_id=cohort[part][i]["pocket_id"],
                             extracted_residues=extracted, pooled_residues=pooled))
    save(output / "expected_split.json", prepared.dataset_summary)
    save(output / "pooling_diagnostics.json", dict(pocket_extraction_angstrom=10, edge_radius_angstrom=6,
         classifier_pool_distance_cutoff=0, structural_readout_scope="residue_only", pockets=rows))
    save(output / "readiness.json", dict(status="passed", manifest_sha256=digest(output / "campaign_manifest.json"),
         gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda, python=sys.version,
         compiled_architectures=torch.cuda.get_arch_list(), cuda_forward_backward=True,
         peak_allocated_bytes=torch.cuda.max_memory_allocated(), held_out_evaluation=False,
         data_root=str(data), cohort_sha256=fingerprint(cohort)))


def execute(root, output, allocation_started_epoch, persistence_receipt=None, max_runs=1):
    """Advance one bounded attempt; caller archives it before calling again."""
    require(max_runs == 1, "Pilot execution is one attempt per invocation; archive before advancing.")
    root, output = Path(root).resolve(), Path(output).resolve()
    manifest = verify_manifest(root, output)
    storage = storage_check(output, persistence_receipt)
    ready = read(output / "readiness.json", {})
    require(ready.get("status") == "passed" and ready["manifest_sha256"] == digest(output / "campaign_manifest.json"),
            "Run preflight successfully before execution.")
    verify_training_cache(output)
    elapsed = allocation_elapsed(output, allocation_started_epoch)
    attempts = _attempts(output)
    _recover_interrupted(output, attempts)
    _require_transfer(output, attempts, storage)
    rows = completed_rows(output, manifest)
    block, all_runs, pending = next_block(manifest, rows)
    if block is None:
        save(output / "campaign_state.json", dict(status="completed", allocated_seconds=elapsed))
        return summarize(output)
    available = budget_available(elapsed, attempts, block)
    admission_path = output / "block_admissions.json"
    admissions = read(admission_path, {})
    if block != "S":
        forecast = sum(forecast_seconds(run, attempts, manifest) for run in pending) * LIMITS["admission_factor"]
        if forecast > available:
            save(output / "campaign_state.json", dict(status="budget_stopped", block=block,
                 reason="Whole remaining block does not fit measured forecast plus 25% margin.",
                 forecast_seconds=forecast, available_seconds=available, allocated_seconds=elapsed))
            return summarize(output)
        admissions.setdefault(block, dict(run_ids=[r["id"] for r in all_runs], forecast_seconds=forecast,
                                         admitted_epoch=time.time(), allocated_seconds=elapsed))
        require(admissions[block]["run_ids"] == [r["id"] for r in all_runs], "Admitted block selection changed.")
        save(admission_path, admissions)
    elif available <= 20:
        save(output / "campaign_state.json", dict(status="budget_stopped", block=block, reason="Profiling allowance exhausted."))
        return summarize(output)
    run = pending[0]
    validate(parse_config(run["command"]), run["epochs"])
    save(output / "campaign_state.json", dict(status="running", block=block, run_id=run["id"], allocated_seconds=elapsed))
    try:
        attempt = _record_attempt(root, output, run, run["command"], run["env"], available, attempts)
    except BaseException:
        save(output / "campaign_state.json", dict(status="failed", block=block, run_id=run["id"],
             reason="Inspect the attempt log; at most one linked retry is allowed."))
        summarize(output)
        raise
    allocation_elapsed(output, allocation_started_epoch)
    save(output / "campaign_state.json", dict(status="awaiting_archive_transfer" if storage["method"] == "verified_archive_transfer"
         else "step_completed", block=block, run_id=run["id"], attempt_id=attempt["attempt_id"],
         attempt_status=attempt["status"]))
    return summarize(output)


def summarize(output):
    output = Path(output)
    manifest = read(output / "campaign_manifest.json")
    rows = completed_rows(output, manifest)
    main_rows = [r for r in rows if r["block"] != "S"]
    fields = ["id", "block", "family", "scheme", "lr", "seed", "selected_epoch", "balanced_accuracy",
              "collapsed4_balanced_accuracy", "minimum_recall", "collapsed4_minimum_recall",
              "per_class_recall", "collapsed4_per_class_recall", "elapsed_seconds", "run_dir", "block_complete"]
    complete_ids = {r["id"] for r in rows}
    for row in main_rows:
        row["block_complete"] = all(r["id"] in complete_ids for r in active_block_runs(manifest, row["block"], rows))
    csv_save(output / "architecture_screen.csv", [r for r in main_rows if r["scheme"] == "four_class"], fields)
    csv_save(output / "target_formulation_screen.csv", [r for r in main_rows if r["family"] in CORE], fields)
    coverage = [{**{k: r[k] for k in ("id", "block", "family", "scheme", "lr", "seed")},
                 "completed": r["id"] in complete_ids} for r in manifest["runs"]]
    save(output / "campaign_coverage.json", coverage)
    decision = dict(completed_smokes=len(rows) - len(main_rows), completed_main_runs=len(main_rows),
                    hybrid_gate=hybrid_gate(rows), promoted=False, held_out_evaluation=False,
                    evidence="fixed_split_screen; incomplete blocks excluded from paired comparisons",
                    state=read(output / "campaign_state.json", {"status": "planned"}))
    save(output / "validation_results.json", dict(**decision, runs=main_rows))
    text = (f"# {PROFILE}\n\nCompleted smokes: {decision['completed_smokes']}/7. "
            f"Completed full runs: {len(main_rows)} (maximum 33).\n\n"
            f"State: {decision['state']['status']}. Hybrid: {decision['hybrid_gate']['status']} "
            "(exploratory prioritization only).\n\n"
            "Checkpoints use native validation balanced accuracy. Common-four metrics come from the same checkpoint. "
            "Partial blocks are visible in CSVs and excluded from paired comparisons. No candidate is promoted. "
            "Smoke scores do not rank architectures. No held-out evaluation is included.\n")
    (output / "decision_record.md").write_text(text)
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "preflight", "execute", "summarize", "close-allocation", "_preflight"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--external-features-root-dir", type=Path)
    parser.add_argument("--feature-overlay-manifest", type=Path)
    parser.add_argument("--preflight-run-name", default="preflight_run", help=argparse.SUPPRESS)
    parser.add_argument("--allocation-started-epoch", type=float)
    parser.add_argument("--allocation-ended-epoch", type=float)
    parser.add_argument("--persistence-receipt", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.phase == "plan":
        require(args.data_root is not None and args.source_commit, "Plan requires --data-root and --source-commit.")
        result = plan(root, args.data_root, args.output_dir, args.source_commit,
                      args.external_features_root_dir, args.feature_overlay_manifest)
        print(f"Saved {len(result['runs'])} immutable templates; at most 7 smokes and 33 full runs.")
    elif args.phase in ("preflight", "execute"):
        require(args.allocation_started_epoch is not None, "Supply the actual GPU allocation start epoch, including setup.")
        operation = preflight if args.phase == "preflight" else execute
        print(json.dumps(operation(root, args.output_dir, args.allocation_started_epoch, args.persistence_receipt), indent=2))
    elif args.phase == "_preflight":
        _preflight_worker(root, args.output_dir, args.preflight_run_name)
    elif args.phase == "close-allocation":
        close_allocation(args.output_dir, args.allocation_ended_epoch)
    else:
        print(json.dumps(summarize(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
