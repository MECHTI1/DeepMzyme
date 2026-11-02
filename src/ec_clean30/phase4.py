"""Bounded Fold-0 S1 and CLEAN-inner execution; no later-stage or cloud lifecycle actions."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping

from benchmarking.pmm_execution import CampaignExecution, ExecutionPolicy
from ec_clean30 import campaign, commands, phase3, provenance, selector, views
from scripts import run_clean_split30_comparator as clean
from training.ec_membership import read_ec_fold_membership

REPO_ROOT = phase3.REPO_ROOT
SCOPE = "fold0_s1_clean_inner"
RESERVE_BYTES = 5 * 1024**3
MAX_DISTANCE_BYTES = 4_500_000_000
CLEAN_TRAIN_MEMORY_BYTES = 3 * 1024**3
CLEAN_MAP_MEMORY_BYTES = 4_500_000_000
_require, _sha, _json, _write = phase3._require, phase3._sha, phase3._json, phase3._write
Phase4Error = phase3.Phase3Error


def clean_cap(memory_bytes):
    return ["systemd-run", "--user", "--scope", "--quiet", "-p", "CPUQuota=200%", "-p",
            f"MemoryMax={memory_bytes}", "-p", "MemorySwapMax=0", "env", "OMP_NUM_THREADS=2",
            "OPENBLAS_NUM_THREADS=2", "MKL_NUM_THREADS=2", "MKL_THREADING_LAYER=GNU"]


def _clean_caps(request):
    _require(request.get("cap_prefix") == clean_cap(CLEAN_TRAIN_MEMORY_BYTES)
             and request.get("distance_cap_prefix") == clean_cap(CLEAN_MAP_MEMORY_BYTES)
             and type(request.get("threads")) is int and request["threads"] == 2
             and request.get("memory_limit_bytes") == CLEAN_TRAIN_MEMORY_BYTES,
             "CLEAN requires separate canonical 2-CPU, zero-swap caps: 4.5-GB map and 3-GiB train/export.")


def s1_units(plan: Mapping[str, Any] | None = None) -> tuple[campaign.PlannedUnit, ...]:
    expected = campaign.build_fold_plan(0)
    _require(plan is None or dict(plan) == expected, "Phase 4 requires the unchanged complete Fold-0 plan.")
    units = tuple(unit for unit in campaign.plan_units(expected) if unit.step == campaign.STEP_S1)
    _require(len(units) == 96 and all(unit.fold == 0 and unit.epochs == 30 for unit in units),
             "Frozen S1 plan no longer has the declared 96 terminal 30-epoch fits.")
    return units


def clean_units() -> tuple[dict[str, Any], ...]:
    return tuple({"unit_id": f"ecc30_clean_{regime}_inner{inner}_seed42", "regime": regime,
                  "inner_fold": inner, "seed": 42, "epochs": 2500}
                 for inner in range(4) for regime in ("r1_inner", "r2_inner"))


def _clean_roots(view, request):
    # CLEAN must import its pinned private code. Its clone's data/test roots stay blocked by its own resolver.
    private_roots = {(view.campaign_root / "clean_harness").absolute(), (view.campaign_root / "clean_harness").resolve()}
    roots = [Path(root) for root in view.forbidden_read_roots if Path(root).absolute() not in private_roots
             and Path(root).resolve() not in private_roots]
    return clean.resolve_forbidden_roots(roots, clean_clone=Path(request["clean_clone"]))


def _source_view(request, *, allow_draft=False):
    draft = request.get("draft", False)
    _require(type(draft) is bool and (allow_draft or not draft), "Draft source is allowed only for CPU input preparation.")
    source = phase3.verify_execution_source(Path(request["execution_manifest"]), require_published=not draft)
    _require("src/ec_clean30/phase4.py" in source["source_sha256"], "Execution manifest does not authenticate Phase 4.")
    view = views.View.load(request["view"])
    _require(view.fold == 0 and not view.smoke, "Only the real frozen Fold-0 view is accepted.")
    phase3._bind_view(view, source)
    return source, view


def _positive(value, name):
    _require(type(value) in (int, float) and math.isfinite(value) and value > 0, f"{name} must be positive and finite.")
    return float(value)


def verify_authority(request, source, *, resource=None):
    freeze_path, authority_path = Path(request["freeze1c"]), Path(request["g3_authority"])
    for path in (freeze_path, authority_path):
        phase3._no_restricted_path(path)
    authority, measured = _json(authority_path), _json(freeze_path)
    _require(_sha(authority_path) == request["g3_sha256"], "G3 authority checksum differs.")
    _require(authority.get("campaign_id") == campaign.CAMPAIGN_ID and authority.get("status") == "approved"
             and authority.get("scope") == SCOPE, "Numerical G3 approval for exactly Fold-0 S1 plus CLEAN inner is required.")
    _require(authority.get("execution_manifest_sha256") == source["execution_manifest_sha256"]
             and authority.get("freeze1c_sha256") == _sha(freeze_path), "G3 source or measured Freeze1c binding differs.")
    _require(measured.get("campaign_id") == campaign.CAMPAIGN_ID
             and measured.get("freeze1_sha256") == source["freeze1_sha256"]
             and measured.get("execution_manifest_sha256") == source["execution_manifest_sha256"]
             and measured.get("lanes") in (1, 3) and len(measured.get("evidence", [])) == 10,
             "Complete measured Freeze1c evidence is required before Phase 4.")
    for record in measured["evidence"]:
        path = Path(request.get("measurement_files", {}).get(record["sha256"], record["path"]))
        phase3._no_restricted_path(path)
        _require(_sha(path) == record["sha256"], "Freeze1c measurement evidence changed.")
    ceilings = authority["ceilings"]
    for key in ("gpu_seconds", "cpu_seconds", "cost_usd"):
        _positive(ceilings[key], key)
    allocations = authority.get("allocations", {})
    _require(bool(allocations), "G3 needs actual bounded allocation entries.")
    totals = {"gpu": 0.0, "cpu": 0.0, "cost": 0.0}
    for row in allocations.values():
        _require(row.get("resource") in ("gpu", "cpu"), "Unknown G3 allocation resource.")
        seconds = _positive(row["max_total_seconds"], "allocation seconds")
        _require(math.isfinite(row["allocation_started_unix"]) and math.isfinite(row["deadline_unix"])
                 and 0 < row["deadline_unix"]-row["allocation_started_unix"] <= seconds,
                 "G3 allocation deadline lies outside its numerical ceiling.")
        totals[row["resource"]] += seconds
        cost = row["cost_ceiling_usd"]
        _require(type(cost) in (float, int) and math.isfinite(cost) and cost >= 0, "Invalid allocation cost ceiling.")
        totals["cost"] += cost
    _require(totals["gpu"] <= ceilings["gpu_seconds"] and totals["cpu"] <= ceilings["cpu_seconds"]
             and totals["cost"] <= ceilings["cost_usd"], "G3 allocations exceed approved numerical ceilings.")
    if resource:
        allocation = allocations.get(request["session_id"], {})
        _require(allocation.get("resource") == resource and all(allocation.get(key) == request.get(key)
                 for key in ("allocation_started_unix", "deadline_unix", "max_total_seconds")),
                 "Runtime does not match its G3 allocation entry.")
    return authority, measured


def _runtime(request, *, resource):
    source, view = _source_view(request)
    authority, measured = verify_authority(request, source, resource=resource)
    _require(isinstance(request.get("cap_prefix"), list) and bool(request["cap_prefix"]), "Resource cap prefix is required.")
    _require(0 < _positive(request["unit_timeout_seconds"], "unit timeout") <= _positive(request["forecast_seconds"], "forecast"),
             "Timeout must fit the complete-unit forecast.")
    _positive(request["memory_limit_bytes"], "memory limit")
    _require(request["device"] == ("cuda" if resource == "gpu" else "cpu"), "Runtime device differs from its authority.")
    policy = ExecutionPolicy(deadline_unix=request["deadline_unix"], max_total_seconds=request["max_total_seconds"],
                             allocation_started_unix=request["allocation_started_unix"], reserve_seconds=900,
                             forecast_margin=1.25)
    root = Path(request["execution_root"]).resolve()
    state_path = root / "execution_state.json"
    if state_path.exists():
        for session_id, prior in _json(state_path).get("allocations", {}).items():
            recorded = authority["allocations"].get(session_id, {})
            _require(recorded.get("allocation_started_unix") == prior.get("started_unix")
                     and recorded.get("max_total_seconds", 0) >= prior.get("max_total_seconds", 0),
                     "G3 allocation history cannot reset or omit previously admitted sessions.")
    if resource == "gpu":
        _require(root == (view.default_runs_dir / "phase4").resolve(), "S1 execution must use the dedicated Fold-0 root.")
        _require(request.get("persistence_mode") == "host_pull", "Remote S1 requires independent host acknowledgment.")
        probe = _json(Path(request["vm_probe"]))
        _require(probe.get("status") == "passed" and probe.get("view_manifest_sha256") == view.manifest_sha256
                 and probe.get("source", {}).get("execution_manifest_sha256") == source["execution_manifest_sha256"],
                 "S1 needs the matching relocated VM probe.")
    else:
        _require(request.get("persistence_mode") == "mounted", "Serial workstation CLEAN uses verified independent mounted storage.")
        _clean_caps(request)
        _guard(root, view)
    return source, view, policy, authority, measured


def _execution(request, policy):
    return CampaignExecution(Path(request["execution_root"]), policy=policy, session_id=request["session_id"],
                             durable_root=Path(request["durable_root"]), persistence_mode=request["persistence_mode"])


def _guard(path: Path, view):
    provenance.refuse_test_paths([path], what="Phase-4 input/output")
    clean.refuse_forbidden_root(path, [Path(root) for root in view.forbidden_read_roots], "Phase-4 input/output")


def _unit(request, kind):
    if kind == "embedding":
        units = {f"ecc30_clean_embed_{name}": {"kind": name} for name in ("base", "masked")}
    else:
        units = ({unit.unit_id: unit for unit in s1_units(_json(Path(request["plan"])))} if kind == "s1"
                 else {unit["unit_id"]: unit for unit in clean_units()})
    _require(request["unit_id"] in units, "Unit is outside the exact 96 S1 / eight CLEAN-inner scope.")
    return units[request["unit_id"]]


def _attempt(request):
    attempt = request.get("attempt", 1)
    _require(type(attempt) is int and attempt >= 1, "Attempt must be a positive integer; failures restart their original seed.")
    return Path(request["execution_root"]) / "runs" / request["unit_id"] / f"attempt{attempt}"


def build_s1_command(request, source, view):
    unit = _unit(request, "s1")
    return commands.build_unit_command(view, campaign.recipe_for_unit(unit), unit_id=unit.unit_id,
        kind=unit.kind, seed=unit.seed, runs_dir=_attempt(request) / "fit", step=unit.step,
        inner_fold=unit.inner_fold, device=request["device"], freeze1_sha256=source["freeze1_sha256"],
        freeze1b_sha256=source["freeze1b_sha256"], freeze1_amendment_chain=source["amendment_chain"],
        git_head=source["git_head"], tree_sha256=source["source_tree_sha256"])


def s1_worker(request):
    source, view, _, _, _ = _runtime(request, resource="gpu")
    command = build_s1_command(request, source, view)
    started = time.monotonic()
    runner = commands.run_unit(command, view=view, cap_prefix=request["cap_prefix"], python=request["python"])
    _require(runner.get("status") == "completed", "S1 fit failed; preserve the attempt before original-seed retry.")
    verified = commands.verify_units(Path(command.runs_dir), unit_ids=[command.unit_id], provenance=source,
                                    memory_limit_bytes=request["memory_limit_bytes"])
    _require(verified.get("status") == "passed", "S1 terminal verification failed.")
    run = Path(command.runs_dir) / command.unit_id
    argv = [request["python"], "-m", "training.ec_prediction_export", "replay", "--run-dir", str(run),
            "--device", request["device"], "--logit-atol", "1e-6"]
    for root in view.forbidden_read_roots:
        argv += ["--forbidden-read-root", root]
    env = command.environment()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    with (run / "phase4_replay.log").open("x") as log:
        subprocess.run([*request["cap_prefix"], *argv], cwd=command.runs_dir, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    replay = run / "ec_prediction_replay/ec_prediction_replay_receipt.json"
    _require(replay.is_file(), "Independent S1 replay receipt is missing.")
    result = {"status": "passed", "unit_id": command.unit_id, "kind": "s1", "source": source,
              "g3_sha256": request["g3_sha256"], "runs_dir": str(Path(command.runs_dir).resolve()),
              "runner_sha256": _sha(commands.runner_receipt_path(Path(command.runs_dir), command.unit_id)),
              "replay_sha256": _sha(replay), "wall_seconds": time.monotonic()-started}
    _write(_attempt(request) / "unit_result.json", result)
    return result


def _verify_result(request, result, kind, source, view, *, relocated=False):
    _require(result.get("status") == "passed" and result.get("unit_id") == request["unit_id"]
             and result.get("source", {}).get("execution_manifest_sha256") == source["execution_manifest_sha256"],
             "Saved unit has another identity, source or status.")
    if kind == "s1":
        runs = Path(result["runs_dir"])
        _require(_sha(commands.runner_receipt_path(runs, request["unit_id"])) == result["runner_sha256"], "Runner receipt changed.")
        if relocated:
            fit = selector.read_runner_records([runs], fold=0)[request["unit_id"]]
            selector.verify_fit_evidence(fit)
        else:
            _require(commands.verify_unit(runs, request["unit_id"], view=view,
                     memory_limit_bytes=request["memory_limit_bytes"]).get("passed"), "Saved S1 fit no longer verifies.")
        _require(_sha(runs / request["unit_id"] / "ec_prediction_replay/ec_prediction_replay_receipt.json") == result["replay_sha256"],
                 "Independent replay evidence changed.")
    elif kind == "clean":
        export_path = Path(result["export"])
        attempt = export_path.parent
        for relative, key in ((f"work/{clean.FIT_MANIFEST}", "fit_manifest_sha256"),
                              ("capped_clean_python", "clean_python_launcher_sha256"),
                              ("distance_map_execution.json", "distance_map_execution_sha256")):
            path = attempt / relative
            _require(path.is_file() and not path.is_symlink() and _sha(path) == result.get(key),
                     "Saved CLEAN fit, launcher or map evidence changed.")
        export = clean.read_distance_export(export_path)
        spec = _unit(request, "clean")
        _require(export.receipt.get("seed") == 42 and export.receipt.get("regime") == spec["regime"]
                 and export.receipt.get("inner_fold") == spec["inner_fold"]
                 and _sha(export_path) == result["export_sha256"], "Saved CLEAN C4 export changed.")
    else:
        farm = Path(result["farm"])
        _require(_sha(farm / "farm_receipt.json") == result["producer_receipt_sha256"]
                 and _sha(farm / clean.FARM_MANIFEST) == result["farm_manifest_sha256"], "Embedding farm receipts changed.")
        for entry in clean.read_farm_manifest(farm / clean.FARM_MANIFEST).values():
            _require(_sha(farm / f"{entry.label}.pt") == entry.file_sha256, "Completed embedding farm changed.")


def run_unit(request, *, kind):
    resource = "cpu" if kind == "clean" else "gpu"
    source, view, policy, _, _ = _runtime(request, resource=resource)
    _unit(request, kind)
    root, attempt = Path(request["execution_root"]), _attempt(request)
    completed = root / "runs" / request["unit_id"] / "complete.json"
    with _execution(request, policy) as execution:
        if completed.exists():
            record = _json(completed)
            _verify_result(request, record, kind, source, view)
            return record
        result_path = attempt / "unit_result.json"
        if result_path.exists():
            record = _json(result_path)
            _verify_result(request, record, kind, source, view)
            pending = execution.state.get("pending_unit", {})
            _require(pending.get("run_name") == request["unit_id"] and pending.get("persisted"),
                     "Terminal unit must reach verified independent persistence before completion.")
            record.update(persistence_receipt=pending["persistence_receipt"], attempt=request.get("attempt", 1))
            _write(completed, record)
            execution.persist([completed])
            return record
        _require(not attempt.exists(), "An incomplete attempt already exists; preserve it and increment attempt explicitly.")
        request_file = root / "commands" / f"{request['unit_id']}_attempt{request.get('attempt', 1)}.json"
        _write(request_file, dict(request))
        execution.admit(request["unit_id"], request["forecast_seconds"])
        started = time.monotonic()
        logfile = request_file.with_suffix(".log")
        with logfile.open("x") as log:
            argv = [request["python"], str(REPO_ROOT / "scripts/run_ec_clean30_phase4.py"), f"_{kind}-worker",
                    "--request", str(request_file)]
            if kind == "clean":
                # The worker's envelope must accommodate the map; training/export have their own stricter child cap.
                argv = [*request["distance_cap_prefix"], *argv]
            status = execution.run_subprocess(argv, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                              timeout=request["unit_timeout_seconds"])
        passed = status.returncode == 0 and result_path.is_file() and _json(result_path).get("status") == "passed"
        execution.record_result(request["unit_id"], "completed" if passed else "failed_fit", time.monotonic()-started)
        artifacts = [request_file, logfile]
        if attempt.exists():
            artifacts += [path for path in attempt.rglob("*") if path.is_file() and not path.is_symlink()]
        persistence = execution.persist(artifacts)
        _require(passed, "Unit failed; its attempt is preserved. Diagnose before an original-seed retry.")
        return {"status": "awaiting_host_ack" if resource == "gpu" else "persisted", "unit_id": request["unit_id"],
                "persistence": persistence}


def prepare_clean_inputs(request):
    """Copy only the frozen training-side Fold-0 closure; never inspect campaign identity outputs."""
    source, view = _source_view(request, allow_draft=True)
    out = Path(request["out"])
    _guard(out, view)
    _require(not out.exists(), "CLEAN input closure is written once.")
    official = Path(request["official_train"])
    phase3._no_restricted_path(official)
    _require(_sha(official) == source["official_fold0_training_sha256"], "Official Fold-0 training bytes differ.")
    overlay = Path(request["overlay"])
    _require(overlay.resolve() != (view.campaign_root / "identity").resolve()
             and overlay.name == view.overlay["id"], "Only the frozen data overlay supplies CLEAN inner lists.")
    manifest_path = overlay / view.overlay["manifest_file"]
    _require(_sha(manifest_path) == view.overlay["manifest_sha256"], "Frozen overlay manifest changed.")
    manifest = _json(manifest_path)
    pins = {row["path"]: row["sha256"] for row in manifest["files"]}
    membership = read_ec_fold_membership(view.membership_file(), expected_sha256=view.membership["sha256"])
    roots = [Path(root) for root in view.forbidden_read_roots]
    out.mkdir(parents=True)
    shutil.copyfile(official, out / "split30_train_split_0.csv")
    rows, _ = clean.read_official_training_rows(out / "split30_train_split_0.csv", 0, roots)
    by_id = {row.accession: row for row in rows}
    specs, closure, mask_names = [], set(), []
    for inner in range(4):
        names = (f"identity/fold0/r1_inner_exclusions_fold0_inner{inner}.tsv",
                 f"r2_lists/r2_inner_allowlist_fold0_inner{inner}.txt")
        for name in names:
            path = overlay / name
            _require(path.is_file() and not path.is_symlink() and _sha(path) == pins.get(name), "Frozen inner list changed.")
            shutil.copyfile(path, out / path.name)
        exclusions = out / Path(names[0]).name
        excluded, _ = clean.read_inner_exclusions(exclusions, 0, inner, roots)
        query = sorted(row["accession"] for row in membership.rows if int(row["fold"]) == inner)
        _require(query and set(query) == {key for key, reason in excluded.items() if reason == "inner_validation"},
                 "CLEAN query set differs from the certified inner membership.")
        query_path = out / f"queries_inner{inner}.txt"
        query_path.write_text("".join(f"{key}\n" for key in query))
        for regime in ("r1_inner", "r2_inner"):
            allow = out / Path(names[1]).name if regime == "r2_inner" else None
            chosen = [row for row in rows if row.accession not in excluded] if allow is None else clean.select_rows(
                rows, clean.read_accession_allowlist(allow, roots)[0])
            _require(not ({row.accession for row in chosen} & set(excluded)), "R2-inner intersects homology exclusions.")
            unit = next(row for row in clean_units() if row["regime"] == regime and row["inner_fold"] == inner)
            stem = unit["unit_id"]
            _, masks = clean.write_masked_copies(chosen, 42, out / f"{stem}_masked.fasta", out / f"{stem}_masked.tsv")
            mask_names.append(masks["manifest"])
            closure.update(row.accession for row in chosen)
            closure.update(query)
            specs.append({**unit, "exclusions": exclusions.name, "allowlist": None if allow is None else allow.name,
                          "queries": query_path.name, "masked_manifest": masks["manifest"],
                          "training_accessions_sha256": clean.sha256_lines(sorted(row.accession for row in chosen)),
                          "n_training": len(chosen), "n_queries": len(query)})
    with (out / "sequence_map.tsv").open("x", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(clean.SEQUENCE_MAP_COLUMNS)
        for key in sorted(closure):
            row = by_id[key]
            writer.writerow((key, clean.sha256_bytes(row.sequence.encode()), len(row.sequence), 1, row.sequence))
    mask_labels = set()
    for name in mask_names:
        mask_labels.update(row["label"] for row in clean.read_masked_manifest(out / name, roots)[0])
    record = {"format": "deepmzyme_ec_clean30_clean_inputs_v1", "fold": 0, "source": source,
              "overlay_manifest_sha256": view.overlay["manifest_sha256"], "membership_sha256": view.membership["sha256"],
              "units": specs, "n_sequences": len(closure), "n_masked_sequences": len(mask_labels), "masked_manifests": mask_names,
              "files": {path.name: _sha(path) for path in out.iterdir() if path.is_file()}}
    _write(out / "clean_inputs.json", record)
    return record


def _clean_inputs(request, source, view, authority):
    root = Path(request["clean_inputs"])
    _guard(root, view)
    path = root / "clean_inputs.json"
    _require(_sha(path) == authority["clean_inputs_sha256"], "CLEAN input closure differs from G3 authority.")
    record = _json(path)
    _require(record.get("source", {}).get("execution_manifest_sha256") == source["execution_manifest_sha256"]
             and record.get("membership_sha256") == view.membership["sha256"], "CLEAN input provenance differs.")
    _require({row["unit_id"] for row in record["units"]} == {row["unit_id"] for row in clean_units()}, "CLEAN closure must cover exactly eight units.")
    for name, digest in record["files"].items():
        path = phase3._safe_file(root, name)
        _require(_sha(path) == digest, "Prepared CLEAN input changed.")
    return root, record


def check_hot_storage(root: Path, working_bytes: int):
    _require(type(working_bytes) is int and working_bytes > 0, "Declare the active CLEAN working-byte forecast.")
    parent = root.resolve()
    while not parent.exists():
        parent = parent.parent
    filesystem = subprocess.check_output(["findmnt", "-n", "-o", "FSTYPE", "--target", str(parent)], text=True).strip()
    _require(filesystem == "ext4", "Active CLEAN files must live on internal ext4.")
    _require(shutil.disk_usage(parent).free >= RESERVE_BYTES + math.ceil(1.25*working_bytes),
             "CLEAN staging would breach the 5-GiB free-space reserve including its 25% working margin.")


def _copy_farm(source, target, labels, roots):
    clean.refuse_forbidden_root(source, roots, "embedding farm")
    entries = clean.read_farm_manifest(source / clean.FARM_MANIFEST)
    _require(set(labels) <= set(entries), "Embedding farm lacks exact required Fold-0 labels.")
    target.mkdir(parents=True)
    for label in sorted(labels):
        path = source / f"{label}.pt"
        _require(_sha(path) == entries[label].file_sha256, "Embedding file changed.")
        shutil.copyfile(path, target / path.name)
        _require(_sha(target / path.name) == entries[label].file_sha256, "Hot embedding copy failed verification.")
    clean.write_farm_manifest(target, [entries[label] for label in sorted(labels)])


def _clean_fit(request, work, attempt, roots):
    """Run only the map under its exception, then use the frozen fit/export hooks under the ordinary cap."""
    _clean_caps(request)
    options = dict(work_dir=work, clean_python=Path(request["clean_python"]), clean_clone=Path(request["clean_clone"]),
        training_script=Path(request["training_script"]), distance_helper=Path(request["distance_helper"]), epochs=2500,
        threads=2, extra_forbidden_roots=roots)
    plan = clean.run_fit(**options, dry_run=True)
    map_steps = []
    for step in plan["steps"]:
        if step["step"] == "distance_map":
            env = clean.build_clean_env(clean_clone=Path(request["clean_clone"]), hash_seed=step["hash_seed"],
                                        threads=2, forbidden_roots=roots)
            code = clean._run([*request["distance_cap_prefix"], *step["command"]], cwd=work, env=env,
                              log_path=attempt / "distance_map.log")
            _require(code == 0, "CLEAN distance map failed under its separate memory cap.")
            map_steps.append({**step, "cap_prefix": request["distance_cap_prefix"],
                              "log_sha256": _sha(attempt / "distance_map.log")})
    prepared = _json(work / clean.PREPARE_MANIFEST)
    verified_map = clean.verify_distance_map(work / "data/distance_map", prepared["training_name"],
        prepared["training_table"]["sha256"], base_farm_sha256=prepared["embedding_farm"]["base"]["farm_sha256"],
        input_dim=prepared["input_dim"], helper_sha256=clean.DISTANCE_MAP_HELPER_SHA256)
    _write(attempt / "distance_map_execution.json", {"status": "passed", "steps": map_steps,
           "verified_distance_map": verified_map, "memory_limit_bytes": CLEAN_MAP_MEMORY_BYTES})
    # A tiny executable adapter preserves the existing runner APIs and the actual private interpreter.
    launcher = attempt / "capped_clean_python"
    launcher.write_text("#!/bin/sh\nexec " + shlex.join([*request["cap_prefix"], request["clean_python"]]) + ' "$@"\n')
    launcher.chmod(0o755)
    fit = clean.run_fit(**{**options, "clean_python": launcher})
    return fit, launcher


def clean_worker(request):
    source, view, _, authority, _ = _runtime(request, resource="cpu")
    inputs, closure = _clean_inputs(request, source, view, authority)
    spec = next(row for row in closure["units"] if row["unit_id"] == request["unit_id"])
    unit = _unit(request, "clean")
    attempt = _attempt(request)
    check_hot_storage(attempt, request["working_bytes"])
    _require(0 < request["distance_map_bytes"] <= MAX_DISTANCE_BYTES, "Distance-map forecast exceeds 4.5 GB.")
    attempt.mkdir(parents=True)
    roots = _clean_roots(view, request)
    rows, _ = clean.read_official_training_rows(inputs / "split30_train_split_0.csv", 0, roots)
    excluded, _ = clean.read_inner_exclusions(inputs / spec["exclusions"], 0, unit["inner_fold"], roots)
    chosen = [row for row in rows if row.accession not in excluded] if spec["allowlist"] is None else clean.select_rows(
        rows, clean.read_accession_allowlist(inputs / spec["allowlist"], roots)[0])
    query = clean.read_accession_allowlist(inputs / spec["queries"], roots)[0]
    masks, _ = clean.read_masked_manifest(inputs / spec["masked_manifest"], roots)
    _copy_farm(Path(request["base_farm"]), attempt / "farms/base", {row.accession for row in chosen} | set(query), roots)
    _copy_farm(Path(request["masked_farm"]), attempt / "farms/masked", {row["label"] for row in masks}, roots)
    work = attempt / "work"
    clean.prepare_fit(official_train_csv=inputs / "split30_train_split_0.csv", fold=0, regime=unit["regime"], seed=42,
        work_dir=work, accession_allowlist=None if spec["allowlist"] is None else inputs / spec["allowlist"],
        accession_exclusions=inputs / spec["exclusions"], inner_fold=unit["inner_fold"],
        base_esm_dir=attempt / "farms/base", masked_esm_dir=attempt / "farms/masked", forbidden_roots=roots)
    # Keep the active farm on ext4 and make crash persistence independent of symlink traversal.
    for path in (work / "data/esm_data").iterdir():
        if path.is_symlink():
            original = path.resolve()
            path.unlink()
            os.link(original, path)
    started = time.monotonic()
    fit, capped_python = _clean_fit(request, work, attempt, roots)
    _require(all(path.stat().st_size <= MAX_DISTANCE_BYTES for path in (work / "data/distance_map").glob("*.pkl")),
             "A completed CLEAN distance map exceeds 4.5 GB.")
    export_path = attempt / "validation.npz"
    clean.run_export(work_dir=work, query_ids_file=inputs / spec["queries"], query_esm_dir=attempt / "farms/base",
        out=export_path, mode="validation", clean_python=capped_python, clean_clone=Path(request["clean_clone"]),
        export_script=Path(request["export_script"]), threads=request["threads"], extra_forbidden_roots=roots)
    export = clean.read_distance_export(export_path)
    _require(set(export.query_ids) == set(query) and len(export.query_ids) == len(query)
             and export.receipt.get("n_in_sample_queries") == 0, "CLEAN validation export query closure differs.")
    result = {"status": "passed", "kind": "clean", "unit_id": unit["unit_id"], "source": source,
              "g3_sha256": request["g3_sha256"], "export": str(export_path.resolve()), "export_sha256": _sha(export_path),
              "fit_manifest_sha256": _sha(work / clean.FIT_MANIFEST), "recipe": fit["recipe"],
              "clean_python_launcher_sha256": _sha(capped_python),
              "distance_map_execution_sha256": _sha(attempt / "distance_map_execution.json"),
              "wall_seconds": time.monotonic()-started}
    _write(attempt / "unit_result.json", result)
    return result


def run_clean_queue(request):
    completed = []
    for unit in clean_units():
        current = {**request, "unit_id": unit["unit_id"]}
        result = run_unit(current, kind="clean")
        if result["status"] == "persisted":
            result = run_unit(current, kind="clean")
        _require(result["status"] == "passed", "CLEAN queue stops before an unverified result.")
        completed.append(result["unit_id"])
        # The independently verified mounted archive retains these inactive working files.
        attempt = Path(result["export"]).parent
        for relative in ("farms", "work/data/esm_data", "work/data/distance_map"):
            path = attempt / relative
            if path.exists():
                shutil.rmtree(path)
    return {"status": "passed", "completed": completed, "next_gate": "S1_selection_and_G4"}


def embedding_worker(request):
    source, view, _, authority, _ = _runtime(request, resource="gpu")
    inputs, record = _clean_inputs(request, source, view, authority)
    kind = _unit(request, "embedding")["kind"]
    directory = _attempt(request)
    directory.mkdir(parents=True)
    roots = _clean_roots(view, request)
    helpers = {"harness": "73c25c5396f89a603a2ca97cf2adf82bb6b220bccf4d9f45801d1bdcf60bede4",
               "extract_script": "902a69cffa0b35ab16798ba655d7b0e357d0e8efd5a268558cc4b064d2e7719d"}
    for key, digest in helpers.items():
        path = clean.refuse_forbidden_root(request[key], roots, key)
        _require(_sha(path) == digest, "Private embedding producer changed.")
    weights = clean.refuse_forbidden_root(request["weights"], roots, "ESM-1b weights")
    contact = Path(str(weights.with_suffix("")) + "-contact-regression.pt")
    for path, key in ((weights, "weights_sha256"), (contact, "contact_weights_sha256")):
        _require(path.is_file() and _sha(path) == request[key], "Local ESM-1b weight pin differs.")
    out = directory / "farm"
    argv = [request["embedding_python"], request["harness"], "--runner", str(REPO_ROOT / "scripts/run_clean_split30_comparator.py"),
            "--extract-script", request["extract_script"], "--extract-sha256", helpers["extract_script"],
            "--weights", str(weights), "--weights-sha256", request["weights_sha256"], "--sequence-map", str(inputs / "sequence_map.tsv"),
            "--sequence-map-sha256", record["files"]["sequence_map.tsv"], "--kind", kind,
            "--out-dir", str(out), "--device", "cuda", "--threads", str(request["threads"])]
    if kind == "masked":
        argv += ["--seed", "42"]
        for name in record["masked_manifests"]:
            argv += ["--masked-manifest", str(inputs / name)]
    for root in roots:
        argv += ["--forbidden-root", str(root)]
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    started = time.monotonic()
    proof = phase3._esm1b_cuda_preflight({**request, "clean_python": request["embedding_python"]}, env, directory)
    with (directory / "embedding.log").open("x") as log:
        subprocess.run([*request["cap_prefix"], *argv], cwd=REPO_ROOT, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    proof.update(phase3._esm1b_gpu_evidence(out))
    result = {"status": "passed", "kind": "embedding", "unit_id": request["unit_id"], "source": source,
              "g3_sha256": request["g3_sha256"], "farm": str(out.resolve()), "device_evidence": proof,
              "producer_receipt_sha256": _sha(out / "farm_receipt.json"),
              "farm_manifest_sha256": _sha(out / clean.FARM_MANIFEST), "wall_seconds": time.monotonic()-started}
    _write(directory / "unit_result.json", result)
    return result


def _attach_session(request, transport):
    from ec_clean30 import workstation

    workstation._identity(request, workstation._controller_config(transport.controller_root))
    _require(workstation._status(transport.controller("status"), request["expected_identity"]) == "RUNNING",
             "Phase 4 attaches only to the already running, provider-protected allocation.")
    current = _json(transport.controller_root / "state/current_session.json")
    _require(current.get("active") is True and current.get("session_id") == request["session_id"]
             and all(str(current.get(key)) == str(value) for key, value in request["expected_identity"].items()),
             "Controller allocation identity differs from G3; no workload submitted.")
    started, deadline = workstation._epoch(current["session_start"]), workstation._epoch(current["termination_ts"])
    _require(started == request["allocation_started_unix"] and deadline == request["deadline_unix"]
             and 0 < current["max_run_duration_s"] <= request["max_total_seconds"]
             and 0 < deadline-started <= request["max_total_seconds"], "Actual provider deadline differs from G3 admission.")
    return current


def _close_session(request, transport):
    from ec_clean30 import workstation

    workstation._identity(request, workstation._controller_config(transport.controller_root))
    current = _json(transport.controller_root / "state/current_session.json")
    _require(not current.get("active") or current.get("session_id") == request["session_id"],
             "Another allocation owns this endpoint; refusing a blind stop.")
    transport.controller("stop", timeout=240)
    status = transport.controller("status")
    _require(workstation._status(status, request["expected_identity"]) == "TERMINATED", "Provider shutdown is not verified.")
    return {"session_id": request["session_id"], "provider_stop_verified": True, "verified_unix": time.time()}


def _run_remote_queue(request, units, *, action, next_gate):
    """One owned allocation, all remaining exact units; admission refuses work that needs another approved session."""
    from ec_clean30.workstation import ControllerTransport, HostPull, complete_remote_unit

    source = phase3.verify_execution_source(Path(request["execution_manifest"]))
    verify_authority(request, source, resource="gpu")
    transport = ControllerTransport(Path(request["controller_root"]), request["ssh_alias"],
                                    bootstrap_python=request.get("bootstrap_python", "/usr/bin/python3"))
    transport.deadline_unix = request["deadline_unix"]
    transport.reserve_seconds = 900
    host = HostPull(transport, Path(request["durable_root"]), Path(request["host_state_root"]),
                   max_download_bytes=request["max_download_bytes"], deadline_unix=request["deadline_unix"])
    _require(re.fullmatch(r"[A-Za-z0-9_.-]+", request["session_id"]), "Unsafe allocation identifier.")
    completed = []
    owned = False
    try:
        _attach_session(request, transport)
        owned = True
        for unit in units:
            _attach_session(request, transport)
            name = unit["unit_id"]
            current = {**request["remote_request_template"], **unit,
                       "attempt": request.get("attempts", {}).get(name, 1)}
            _require(all(current.get(key) == request.get(key) for key in
                     ("session_id", "allocation_started_unix", "deadline_unix", "max_total_seconds", "g3_sha256")),
                     "Remote request allocation differs from the controller-verified G3 allocation.")
            result = complete_remote_unit(transport, host, current, runner=request["remote_runner"], action=action,
                remote_request=f"{current['execution_root']}/queue/{request['session_id']}/{name}_attempt{current['attempt']}.json",
                completed_relative=f"runs/{name}/complete.json", timeout=current["unit_timeout_seconds"]+60)
            _require(result.get("status") == "passed" and result.get("unit_id") == name,
                     "Queue stops at an incomplete terminal unit.")
            completed.append(name)
    finally:
        if owned:
            try:
                host.sync_pending(request["remote_request_template"]["execution_root"])
            finally:
                receipt = _close_session(request, transport)
                _write(Path(request["host_state_root"]) / f"closeout_{request['session_id']}.json", receipt)
    return {"status": "passed", "completed": completed, "next_gate": next_gate}


def run_s1_queue(request):
    units = [{"unit_id": unit.unit_id, "forecast_seconds": request["family_forecasts"][unit.family],
              "unit_timeout_seconds": request["family_timeouts"][unit.family]}
             for unit in s1_units(_json(Path(request["local_plan"])))]
    return _run_remote_queue(request, units, action="s1-unit", next_gate="S1_selection_and_G4")


def run_embedding_queue(request):
    units = [{"unit_id": f"ecc30_clean_embed_{kind}", "forecast_seconds": request["embedding_forecasts"][kind],
              "unit_timeout_seconds": request["embedding_timeouts"][kind]} for kind in ("base", "masked")]
    return _run_remote_queue(request, units, action="clean-embeddings", next_gate="eight_serial_CLEAN_inner_fits")


def select_gap(request):
    """Use the unchanged published selector in its own sparse source checkout."""
    source, view = _source_view(request)
    verify_authority(request, source)
    plan = _json(Path(request["plan"]))
    units = s1_units(plan)
    runs = []
    for unit in units:
        unit_root = Path(request["s1_execution_root"]) / "runs" / unit.unit_id
        record = _json(unit_root / "complete.json")
        record["runs_dir"] = str(unit_root / f"attempt{int(record['attempt'])}" / "fit")
        _verify_result({**request, "unit_id": unit.unit_id}, record, "s1", source, view, relocated=True)
        runs.append(record["runs_dir"])
    exports = {"r1_inner": [], "r2_inner": []}
    for unit in clean_units():
        unit_root = Path(request["clean_execution_root"]) / "runs" / unit["unit_id"]
        record = _json(unit_root / "complete.json")
        record["export"] = str(unit_root / f"attempt{int(record['attempt'])}" / "validation.npz")
        _verify_result({**request, "unit_id": unit["unit_id"]}, record, "clean", source, view)
        exports[unit["regime"]].append(record["export"])
    out = Path(request["out"])
    _guard(out, view)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ecc30-select-source-") as temporary:
        baseline = Path(temporary) / "source"
        subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(REPO_ROOT), str(baseline)],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        phase3._git(baseline, "sparse-checkout", "set", "--no-cone", "/src/", "/scripts/", "/" + phase3.freeze.CAMPAIGN_DOCS_DIR + "/")
        phase3._git(baseline, "checkout", "--quiet", "--detach", source["amendment_commit"])
        common = ["--fold", "0", "--plan", str(Path(request["plan"]).resolve()), "--components", str(Path(request["components"]).resolve()),
                  "--freeze1-sha256", source["freeze1_sha256"], "--freeze1-amendment", str(baseline / source["amendment_path"])]
        for path in runs:
            common += ["--runs", path]
        env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(baseline / "src"), str(baseline))), PYTHONDONTWRITEBYTECODE="1")
        runner = [request["python"], str(baseline / "scripts/run_ec_clean30_campaign.py")]
        chain, gate = out / "selection.jsonl", out / "gap_gate.json"
        if not chain.exists():
            with (out / "selection.log").open("x") as log:
                subprocess.run([*request["cap_prefix"], *runner, "select", *common, "--step", "S1", "--out", str(chain)],
                               cwd=baseline, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        argv = [*request["cap_prefix"], *runner, "gap-gate", *common, "--after", "S1", "--step-chain", str(chain), "--out", str(gate)]
        for regime, paths in exports.items():
            for path in paths:
                argv += ["--clean-" + regime.replace("_", "-"), path]
        if not gate.exists():
            with (out / "gap.log").open("x") as log:
                subprocess.run(argv, cwd=baseline, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    record = {"status": "passed", "execution_manifest_sha256": source["execution_manifest_sha256"],
              "selection_source_commit": source["amendment_commit"], "selection_sha256": _sha(chain),
              "gap_sha256": _sha(gate), "next_gate": "G4_maintainer_decision_before_any_later_stage"}
    _write(out / "phase4_evidence.json", record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "clean-inputs", "s1-unit", "s1-queue", "clean-unit", "clean-queue",
                                           "clean-embeddings", "embedding-queue", "select-gap", "_s1-worker", "_clean-worker", "_embedding-worker"))
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        request = _json(args.request)
        if args.action == "plan":
            result = {"scope": SCOPE, "s1": [unit.to_dict() for unit in s1_units()], "clean": list(clean_units())}
        elif args.action == "clean-inputs":
            record = prepare_clean_inputs(request)
            result = {"status": "passed", "n_units": len(record["units"]), "n_sequences": record["n_sequences"]}
        elif args.action in ("s1-unit", "clean-unit"):
            result = run_unit(request, kind=args.action.split("-")[0])
        elif args.action == "clean-queue":
            result = run_clean_queue(request)
        elif args.action == "s1-queue":
            result = run_s1_queue(request)
        elif args.action == "clean-embeddings":
            result = run_unit(request, kind="embedding")
        elif args.action == "embedding-queue":
            result = run_embedding_queue(request)
        elif args.action == "select-gap":
            result = select_gap(request)
        else:
            result = {"_s1-worker": s1_worker, "_clean-worker": clean_worker, "_embedding-worker": embedding_worker}[args.action](request)
        print(json.dumps({key: result[key] for key in ("status", "unit_id", "persistence", "n_units", "n_sequences", "next_gate")
                          if key in result} if args.action != "plan" else result, sort_keys=True))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
