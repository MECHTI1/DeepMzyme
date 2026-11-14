"""Bounded Fold-0 S1 and CLEAN-inner execution; no later-stage or cloud lifecycle actions."""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping

from benchmarking.pmm_execution import CampaignExecution, ExecutionPolicy
from ec_clean30 import campaign, commands, hot_storage, phase3, provenance, selector, views
from scripts import run_clean_split30_comparator as clean
from training.ec_membership import read_ec_fold_membership

REPO_ROOT = phase3.REPO_ROOT
SCOPE = "fold0_s1_clean_inner"
V5_SCOPE = "amendment2_fold_execution_v5"
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
    if request.get("provider") == "runpod":
        from ec_clean30 import container_caps

        profile = request.get("clean_cap_profile", {})
        _require(isinstance(profile, dict) and set(profile) ==
                 {"cgroup_root", "cpu_cores", "memory_bytes", "distance_map_memory_bytes"},
                 "Remote CLEAN requires an explicit kernel cap profile.")
        _require(all(type(profile[key]) is int and profile[key] > 0
                     for key in ("cpu_cores", "memory_bytes", "distance_map_memory_bytes"))
                 and request.get("threads") == profile["cpu_cores"]
                 and request.get("memory_limit_bytes") == profile["memory_bytes"],
                 "Remote CLEAN thread and memory metadata differ from its cap profile.")
        for key, memory in (("cap_prefix", "memory_bytes"), ("distance_cap_prefix", "distance_map_memory_bytes")):
            expected = container_caps.cap_prefix(request["python"], REPO_ROOT, profile["cgroup_root"],
                                                cpu_cores=profile["cpu_cores"], memory_bytes=profile[memory])
            _require(request.get(key) == expected, "Remote CLEAN cap differs from its explicit kernel profile.")
        return
    _require(request.get("cap_prefix") == clean_cap(CLEAN_TRAIN_MEMORY_BYTES)
             and request.get("distance_cap_prefix") == clean_cap(CLEAN_MAP_MEMORY_BYTES)
             and type(request.get("threads")) is int and request["threads"] == 2
             and request.get("memory_limit_bytes") == CLEAN_TRAIN_MEMORY_BYTES,
             "CLEAN requires separate canonical 2-CPU, zero-swap caps: 4.5-GB map and 3-GiB train/export.")


def s1_units(plan: Mapping[str, Any] | None = None) -> tuple[campaign.PlannedUnit, ...]:
    if plan is not None and plan.get("format") == "deepmzyme_ec_clean30_fold_plan_v2":
        return fold_units(plan, "S1")
    expected = campaign.build_fold_plan(0)
    _require(plan is None or dict(plan) == expected, "Phase 4 requires the unchanged complete Fold-0 plan.")
    units = tuple(unit for unit in campaign.plan_units(expected) if unit.step == campaign.STEP_S1)
    _require(bool(units) and all(unit.fold == 0 and unit.epochs > 0 for unit in units),
             "The authenticated Fold-0 plan must declare nonempty terminal S1 fits.")
    return units


def fold_units(plan, step):
    expected = campaign.build_fold_plan_v2(plan.get("fold"))
    _require(dict(plan) == expected, "Phase 4 requires the complete canonical amendment-2 fold plan.")
    return campaign.fold_units_v2(plan, step)


def clean_final_units(fold):
    """Read the unchanged CLEAN seeds from the published Freeze-1 record, without importing analysis code."""
    _require(type(fold) is int and fold in range(5), "Invalid CLEAN outer fold.")
    path = REPO_ROOT / phase3.freeze.CAMPAIGN_DOCS_DIR / "freeze1.json"
    settings = _json(path)["constants"]["scoring"]["FROZEN_SETTINGS"]
    return tuple({"unit_id": f"ecc30_clean_fold{fold}_{arm}_seed{seed}", "fold": fold,
                  "arm": arm, "regime": "r1" if arm == "c_esmc" else arm, "inner_fold": None,
                  "seed": seed, "epochs": clean.PUBLISHED_RECIPE["epochs"]}
                 for arm in ("r1", "r2", "c_esmc") for seed in settings[f"{arm}_seeds"])


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


def _request_source(request, *, require_published=True):
    chain_options = {}
    if request.get("amendment_chain_receipt") is not None:
        chain_options["chain_receipt_path"] = Path(request["amendment_chain_receipt"])
    if request.get("expected_chain") is not None:
        chain_options["expected_chain"] = request["expected_chain"]
    if not chain_options and require_published:
        return phase3.verify_execution_source(Path(request["execution_manifest"]))
    return phase3.verify_execution_source(Path(request["execution_manifest"]), require_published=require_published,
                                          **chain_options)


def _source_view(request, *, allow_draft=False):
    draft = request.get("draft", False)
    _require(type(draft) is bool and (allow_draft or not draft), "Draft source is allowed only for CPU input preparation.")
    source = _request_source(request, require_published=not draft)
    _require("src/ec_clean30/phase4.py" in source["source_sha256"], "Execution manifest does not authenticate Phase 4.")
    view = views.View.load(request["view"])
    _require(not view.smoke and (view.fold == 0 or source.get("scientific_identity_sha256")),
             "Only real frozen views are accepted; later folds require amendment 2.")
    _require(request.get("fold", view.fold) == view.fold, "Request and authenticated view folds differ.")
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
    v5 = bool(source.get("scientific_identity_sha256"))
    _require(authority.get("campaign_id") == campaign.CAMPAIGN_ID and authority.get("status") == "approved"
             and authority.get("scope") == (V5_SCOPE if v5 else SCOPE),
             "Numerical G3 approval for the exact Phase-4 execution scope is required.")
    if v5:
        folds = authority.get("allowed_folds")
        _require(isinstance(folds, list) and folds and all(type(fold) is int and fold in range(5) for fold in folds)
                 and folds == sorted(set(folds)), "G3 must declare its exact authorized outer folds.")
        expected_plans = {str(fold): campaign.build_fold_plan_v2(fold)["plan_sha256"] for fold in folds}
        _require(authority.get("plan_sha256_by_fold") == expected_plans,
                 "G3 must bind every authorized canonical amendment-2 plan.")
        steps = authority.get("allowed_steps", {})
        _require(set(steps) == set(expected_plans) and all(isinstance(value, list) and value
                 and value == sorted(set(value)) and set(value) <= {"S1", "S4"} for value in steps.values()),
                 "G3 must declare each fold's allowed fitted steps.")
        allowed_clean = authority.get("clean_unit_ids")
        canonical_clean = {unit["unit_id"] for fold in folds for unit in clean_final_units(fold)}
        if 0 in folds:
            canonical_clean.update(unit["unit_id"] for unit in clean_units())
        _require(isinstance(allowed_clean, list) and allowed_clean == sorted(set(allowed_clean))
                 and set(allowed_clean) <= canonical_clean, "G3 CLEAN unit scope is not canonical.")
    phase3.require_same_scientific_identity(authority, source)
    phase3.require_record_precedence(authority, source)
    _require(authority.get("freeze1c_sha256") == _sha(freeze_path), "G3 measured Freeze1c binding differs.")
    phase3.require_same_scientific_identity(measured, source)
    phase3.require_record_precedence(measured, source)
    _require(measured.get("campaign_id") == campaign.CAMPAIGN_ID
             and measured.get("freeze1_sha256") == source["freeze1_sha256"]
             and measured.get("lanes") in (1, 3) and len(measured.get("evidence", [])) == 10,
             "Complete measured Freeze1c evidence is required before Phase 4.")
    if source.get("scientific_identity_sha256"):
        _require(measured.get("format") == "deepmzyme_ec_clean30_freeze1c_v2"
                 and measured.get("timing_recipes") == [name for name, _ in campaign.timing_recipes_v2()],
                 "Phase 4 requires the complete amendment-2 Freeze1c v2 timing scope.")
        pretrain = measured.get("pretrain_evidence", {})
        path = Path(request.get("measurement_files", {}).get(pretrain.get("sha256"), pretrain.get("path", "")))
        _require(path.is_file() and _sha(path) == pretrain.get("sha256"),
                 "Freeze1c v2 pretraining preparation evidence is missing or changed.")
        phase3.verify_pretrain_evidence(path, source)
    for record in measured["evidence"]:
        path = Path(request.get("measurement_files", {}).get(record["sha256"], record["path"]))
        phase3._no_restricted_path(path)
        _require(_sha(path) == record["sha256"], "Freeze1c measurement evidence changed.")
        if source.get("scientific_identity_sha256"):
            measurement = _json(path)
            phase3.require_same_scientific_identity(measurement, source)
            phase3.require_record_precedence(measurement, source)
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


def _runtime(request, *, resource, kind=None):
    source, view = _source_view(request)
    authority, measured = verify_authority(request, source, resource=resource)
    if source.get("scientific_identity_sha256"):
        _require(request.get("provider") == "runpod" and resource == "gpu",
                 "Execution v5 uses the owned Runpod GPU session; local or GCP work is refused.")
        _require(Path(request["campaign_root"]).resolve() == view.campaign_root.resolve(),
                 "Futility pointers must be read from the authenticated view's campaign root.")
        _require(view.fold in authority["allowed_folds"], "This fold is outside the numerical G3 authority.")
        if kind in ("s1", "refit"):
            step = "S1" if kind == "s1" else "S4"
            _require(step in authority["allowed_steps"][str(view.fold)], "This fitted step is outside G3.")
            _require(_json(Path(request["plan"])) == campaign.build_fold_plan_v2(view.fold),
                     "The fit request does not carry its full authorized v2 plan.")
        if kind == "clean":
            _require(request.get("clean_scope", "inner") in ("inner", "final"), "Unknown CLEAN queue scope.")
            _require(request["unit_id"] in authority["clean_unit_ids"], "This CLEAN fit is outside G3.")
        if kind in ("science", "science_futility"):
            step = request.get("scientific_step")
            _require(step in SCIENTIFIC_STEPS and request.get("unit_id") == f"ecc30_science_fold{view.fold}_{step}",
                     "Scientific jobs must use their canonical fold and action identity.")
            _require(step not in ("held_out_sanity", "verify_sanity", "futility") or view.fold == 0,
                     "The held-out and futility jobs belong to fold 0 only.")
            _require(_json(Path(request["plan"])) == campaign.build_fold_plan_v2(view.fold),
                     "The scientific request does not carry its canonical v2 plan.")
            _science_cap(request, authority)
        if kind != "science_futility":
            require_futility(request, source, view.fold, require_continue=(view.fold != 0 or kind == "refit"
                              or (kind == "clean" and request.get("clean_scope") == "final")))
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
        _require(root == (view.default_runs_dir / "phase4").resolve(), "Execution must use its dedicated outer-fold root.")
        _require(request.get("persistence_mode") == "host_pull", "Remote S1 requires independent host acknowledgment.")
        probe = _json(Path(request["vm_probe"]))
        phase3.require_same_scientific_identity(probe.get("source", {}), source)
        _require(probe.get("status") == "passed" and probe.get("view_manifest_sha256") == view.manifest_sha256,
                 "S1 needs the matching relocated VM probe.")
        if request.get("provider") == "runpod":
            # Every GPU check of a unit names the session's qualified device; none falls back to another provider's.
            profile = request.get("hardware_profile")
            _require(isinstance(profile, str) and profile.startswith("runpod_")
                     and probe.get("hardware_profile") == profile,
                     "A remote unit needs the hardware profile that its session's probe qualified.")
            phase3._approved_runtime_profile(source, profile)
        if request.get("provider") == "runpod" and request.get("clean_cap_profile") is not None:
            from ec_clean30 import container_caps

            _clean_caps(request)
            profile = request["clean_cap_profile"]
            evidence = probe["container_caps_probe"]
            _require(profile["cgroup_root"] == evidence["cgroup_root"]
                     and authority.get("clean_cap_profile") == profile,
                     "Remote CLEAN caps differ from the qualified container or approved numerical authority.")
            qualification_cap = container_caps.cap_prefix(request["python"], REPO_ROOT, evidence["cgroup_root"],
                                                          cpu_cores=2, memory_bytes=8 * 1024**3)
            phase3._verify_container_caps(source, evidence, qualification_cap, python=request["python"])
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


def embedding_units(request):
    if request.get("clean_scope") != "final":
        return tuple({"unit_id": f"ecc30_clean_embed_{kind}", "kind": kind,
                      "embedding": "esm1b", "seed": 42 if kind == "masked" else None}
                     for kind in ("base", "masked"))
    units = clean_final_units(request["fold"])
    result = []
    for embedding in ("esm1b", "esmc"):
        seeds = sorted({unit["seed"] for unit in units
                        if (unit["arm"] == "c_esmc") == (embedding == "esmc")})
        stem = f"ecc30_clean_embed_fold{request['fold']}_{embedding}"
        result.append({"unit_id": f"{stem}_base", "kind": "base", "embedding": embedding, "seed": None})
        result.extend({"unit_id": f"{stem}_masked_seed{seed}", "kind": "masked",
                       "embedding": embedding, "seed": seed} for seed in seeds)
    return tuple(result)


def final_farms_by_unit(request):
    root = PurePosixPath(request["remote_request_template"]["execution_root"])
    attempts = request.get("attempts", {})
    farms = {(unit["embedding"], unit["kind"], unit["seed"]): str(root / "runs" / unit["unit_id"] /
             f"attempt{attempts.get(unit['unit_id'], 1)}" / "farm") for unit in embedding_units(request)}
    return {unit["unit_id"]: {"base_farm": farms[("esmc" if unit["arm"] == "c_esmc" else "esm1b", "base", None)],
            "masked_farm": farms[("esmc" if unit["arm"] == "c_esmc" else "esm1b", "masked", unit["seed"])]}
            for unit in clean_final_units(request["fold"])}


def _unit(request, kind):
    if kind == "embedding":
        units = {unit["unit_id"]: unit for unit in embedding_units(request)}
    else:
        if kind in ("s1", "refit"):
            plan = _json(Path(request["plan"]))
            planned = s1_units(plan) if kind == "s1" else fold_units(plan, "S4")
            units = {unit.unit_id: unit for unit in planned}
        else:
            planned = clean_final_units(request["fold"]) if request.get("clean_scope") == "final" else clean_units()
            units = {unit["unit_id"]: unit for unit in planned}
    _require(request["unit_id"] in units, "Unit is outside the authenticated Fold-0 queue scope.")
    return units[request["unit_id"]]


def require_futility(request, source, fold, *, require_continue=False):
    """Check binding stop pointers even for a resumed fold-0 inner unit; never cache a continue."""
    if not source.get("scientific_identity_sha256"):
        return None
    from ec_clean30 import futility

    campaign_root = Path(request["campaign_root"])
    phase3._no_restricted_path(campaign_root)
    verified = phase3.authenticated_chain(source)
    pointers = futility.read_pointers(campaign_root, verified["tip_sequence"])
    prior_fut = campaign_root / "runs/fold0/phase4/runs/ecc30_science_fold0_futility/complete.json"
    if not pointers and not require_continue and not prior_fut.exists():
        return None
    decision = futility.verify_futility_decision(Path(request["fold0_step_chain"]),
        freeze1_sha256=source["freeze1_sha256"], amendment_tip=REPO_ROOT / source["amendment_path"],
        campaign_root=campaign_root, chain_receipt=Path(source["amendment_chain_receipt"]),
        expected_chain=source["amendment_chain"])
    phase3.require_same_scientific_identity(decision, source)
    _require(decision["decision"] == "continue", "Binding stop_futility refuses all further campaign fits.")
    return decision


def _attempt(request):
    attempt = request.get("attempt", 1)
    _require(type(attempt) is int and attempt >= 1, "Attempt must be a positive integer; failures restart their original seed.")
    return Path(request["execution_root"]) / "runs" / request["unit_id"] / f"attempt{attempt}"


def build_fit_command(request, source, view, *, kind):
    unit = _unit(request, kind)
    recipe = None
    if kind == "refit":
        spec = selector.verified_refit_recipe(Path(request["step_chain"]), view.fold,
            freeze1_sha256=source["freeze1_sha256"], amendment_tip=REPO_ROOT / source["amendment_path"],
            campaign_root=Path(request["campaign_root"]), plan_path=Path(request["plan"]),
            chain_receipt=Path(source["amendment_chain_receipt"]), expected_chain=source["amendment_chain"])
        _require(unit.unit_id in spec.refit_unit_ids and unit.seed in spec.seeds,
                 "The refit template differs from the verified selected recipe.")
        recipe = spec.recipe
    command = commands.build_unit_command(view, campaign.recipe_for_unit(unit, selected=recipe), unit_id=unit.unit_id,
        kind=unit.kind, seed=unit.seed, runs_dir=_attempt(request) / "fit", step=unit.step,
        inner_fold=unit.inner_fold, device=request["device"], freeze1_sha256=source["freeze1_sha256"],
        freeze1b_sha256=source["freeze1b_sha256"], freeze1_amendment_chain=source.get("fit_chain", source["amendment_chain"]),
        git_head=source["git_head"], tree_sha256=source["source_tree_sha256"])
    return replace(command, unset_env=tuple(sorted(set(command.unset_env)
                   | {key for key in os.environ if key.startswith("GIT_")})))


def build_s1_command(request, source, view):
    return build_fit_command(request, source, view, kind="s1")


def replay_cpu_logits(saved_logits, replayed_logits, *, device="cpu"):
    """Frozen cross-device acceptance rule; near ties never excuse an argmax change."""
    if device != "cpu":
        raise ValueError("Amendment-2 replay must run on CPU.")
    try:
        saved = [[float(value) for value in row] for row in saved_logits]
        replayed = [[float(value) for value in row] for row in replayed_logits]
    except (TypeError, ValueError) as exc:
        raise ValueError("Replay logits must be numeric matrices.") from exc
    if not saved or len(saved) != len(replayed):
        raise ValueError("Replay logits must have matching nonempty rows.")
    width = len(saved[0])
    if width < 2 or any(len(row) != width for row in saved + replayed):
        raise ValueError("Replay logit shapes differ or have fewer than two classes.")
    if not all(math.isfinite(value) for row in saved + replayed for value in row):
        raise ValueError("Replay logits must be finite.")
    largest = max(abs(left - right) for old, new in zip(saved, replayed) for left, right in zip(old, new))
    near_ties = 0
    changed = 0
    for old, new in zip(saved, replayed):
        old_ordered, new_ordered = sorted(old, reverse=True), sorted(new, reverse=True)
        if min(old_ordered[0] - old_ordered[1], new_ordered[0] - new_ordered[1]) <= 2e-5:
            near_ties += 1
        # max returns the first index for an exact tie, matching the frozen argmax rule.
        changed += max(range(width), key=old.__getitem__) != max(range(width), key=new.__getitem__)
    if largest > 1e-5 or changed:
        raise ValueError(f"CPU replay differs: max absolute difference {largest:.12g}; argmax changes {changed}.")
    return {"status": "passed", "device": "cpu", "absolute_tolerance": 1e-5,
            "near_tie_tolerance": 2e-5, "argmax_agreement": True,
            "max_logit_abs_difference": largest, "near_tie_rows": near_ties,
            "n_rows": len(saved), "n_classes": width}


def verify_cpu_replay(run_dir):
    """Recheck the CPU receipt and both hash-bound prediction tables under the frozen rule."""
    from training.ec_prediction_export import (EC_PREDICTION_RECEIPT_NAME, EC_REPLAY_RECEIPT_NAME,
                                                DEFAULT_REPLAY_DIRNAME, LOGIT_COLUMNS)

    run = Path(run_dir)
    replay_dir = run / DEFAULT_REPLAY_DIRNAME
    original_path = run / EC_PREDICTION_RECEIPT_NAME
    original = _json(original_path)
    receipt_path = replay_dir / EC_REPLAY_RECEIPT_NAME
    receipt = _json(receipt_path)
    _require(receipt.get("device") == "cpu" and receipt.get("logit_atol") == 1e-5
             and receipt.get("replay_status") == "match" and receipt.get("n_problems") == 0
             and receipt.get("source_receipt_sha256") == _sha(original_path)
             and receipt.get("checkpoint_sha256") == original.get("checkpoint", {}).get("sha256"),
             "Independent CPU replay receipt policy, identity or result differs.")
    compared = {}
    for key in ("pocket_predictions", "protein_predictions"):
        bindings = (original.get(key, {}), receipt.get("replay_files", {}).get(key, {}))
        tables = []
        for directory, binding in zip((run, replay_dir), bindings):
            path = phase3._safe_file(directory, binding.get("path", ""))
            _require(_sha(path) == binding.get("sha256"), "CPU replay prediction bytes changed.")
            with path.open(newline="") as handle:
                tables.append(list(csv.DictReader(handle)))
        left, right = tables
        _require(len(left) == len(right), "CPU replay prediction row count differs.")
        for saved, replayed in zip(left, right):
            _require(set(saved) == set(replayed) and set(LOGIT_COLUMNS) <= set(saved)
                     and {key: value for key, value in saved.items() if key not in LOGIT_COLUMNS}
                     == {key: value for key, value in replayed.items() if key not in LOGIT_COLUMNS},
                     "CPU replay prediction row order, identity or argmax differs.")
        try:
            compared[key] = replay_cpu_logits([[row[column] for column in LOGIT_COLUMNS] for row in left],
                                              [[row[column] for column in LOGIT_COLUMNS] for row in right],
                                              device=receipt["device"])
        except ValueError as exc:
            raise Phase4Error(str(exc)) from exc
    return {"status": "passed", "replay_receipt_sha256": _sha(receipt_path), "comparisons": compared}


def _fit_worker(request, *, kind):
    source, view, _, _, _ = _runtime(request, resource="gpu", kind=kind)
    command = (build_s1_command(request, source, view) if kind == "s1"
               else build_fit_command(request, source, view, kind=kind))
    started = time.monotonic()
    runner = commands.run_unit(command, view=view, cap_prefix=request["cap_prefix"], python=request["python"])
    _require(runner.get("status") == "completed", "S1 fit failed; preserve the attempt before original-seed retry.")
    verified = commands.verify_units(Path(command.runs_dir), unit_ids=[command.unit_id], provenance=source,
                                    memory_limit_bytes=request["memory_limit_bytes"])
    _require(verified.get("status") == "passed", "S1 terminal verification failed.")
    run = Path(command.runs_dir) / command.unit_id
    replay_atol = "1e-5" if request["device"] == "cuda" else "1e-6"
    argv = [request["python"], "-m", "training.ec_prediction_export", "replay", "--run-dir", str(run),
            "--device", "cpu", "--logit-atol", replay_atol]
    for root in view.forbidden_read_roots:
        argv += ["--forbidden-read-root", root]
    env = command.environment()
    env = {name: value for name, value in env.items() if not name.startswith("GIT_")}
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    with (run / "phase4_replay.log").open("x") as log:
        subprocess.run([*request["cap_prefix"], *argv], cwd=command.runs_dir, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    replay = run / "ec_prediction_replay/ec_prediction_replay_receipt.json"
    _require(replay.is_file(), "Independent S1 replay receipt is missing.")
    replay_record = _json(replay)
    _require(replay_record.get("device") == "cpu" and replay_record.get("logit_atol") == float(replay_atol)
             and replay_record.get("replay_status") == "match", "Independent S1 replay policy or result differs.")
    comparison = verify_cpu_replay(run) if source.get("scientific_identity_sha256") else None
    result = {"status": "passed", "unit_id": command.unit_id, "kind": kind, "source": source,
              "g3_sha256": request["g3_sha256"], "runs_dir": str(Path(command.runs_dir).resolve()),
              "runner_sha256": _sha(commands.runner_receipt_path(Path(command.runs_dir), command.unit_id)),
              "replay_sha256": _sha(replay), "wall_seconds": time.monotonic()-started}
    if source.get("scientific_identity_sha256"):
        result.update(phase3.source_identity(source), git_head=source["git_head"], created_utc=provenance.utc_now(),
                      replay_comparison=comparison)
    _write(_attempt(request) / "unit_result.json", result)
    return result


def s1_worker(request):
    return _fit_worker(request, kind="s1")


def refit_worker(request):
    return _fit_worker(request, kind="refit")


def _dispositions_for(records, dispositions):
    """Route dispositions by their record bytes; the scientific consumer validates them."""
    digests = {_sha(Path(path)) for path in records}
    return [Path(path) for path in dispositions
            if _json(Path(path)).get("record_sha256") in digests]


def _unit_dispositions(request, record):
    paths = request.get("clean_sanity_dispositions_by_unit", {}).get(request["unit_id"], [])
    return _dispositions_for([record], paths)


def _verify_result(request, result, kind, source, view, *, relocated=False):
    phase3.require_same_scientific_identity(result.get("source", {}), source)
    if source.get("scientific_identity_sha256"):
        phase3.require_same_scientific_identity(result, source)
        phase3.require_record_precedence(result, source)
    _require(result.get("status") == "passed" and result.get("unit_id") == request["unit_id"],
             "Saved unit has another identity, source or status.")
    if kind in ("s1", "refit"):
        runs = Path(result["runs_dir"])
        _require(_sha(commands.runner_receipt_path(runs, request["unit_id"])) == result["runner_sha256"], "Runner receipt changed.")
        if relocated:
            fit = selector.read_runner_records([runs], fold=view.fold)[request["unit_id"]]
            selector.verify_fit_evidence(fit)
        else:
            _require(commands.verify_unit(runs, request["unit_id"], view=view,
                     memory_limit_bytes=request["memory_limit_bytes"]).get("passed"), "Saved S1 fit no longer verifies.")
        _require(_sha(runs / request["unit_id"] / "ec_prediction_replay/ec_prediction_replay_receipt.json") == result["replay_sha256"],
                 "Independent replay evidence changed.")
        if source.get("scientific_identity_sha256"):
            _require(verify_cpu_replay(runs / request["unit_id"]) == result.get("replay_comparison"),
                     "Saved CPU replay comparison changed.")
    elif kind == "clean":
        export_path = Path(result["export"])
        attempt = export_path.parent
        for relative, key in ((f"work/{clean.FIT_MANIFEST}", "fit_manifest_sha256"),
                              ("capped_clean_python", "clean_python_launcher_sha256"),
                              ("distance_map_execution.json", "distance_map_execution_sha256")):
            path = attempt / relative
            _require(path.is_file() and not path.is_symlink() and _sha(path) == result.get(key),
                     "Saved CLEAN fit, launcher or map evidence changed.")
        if request.get("provider") == "runpod":
            _require(_clean_device_evidence(attempt, export_name=export_path.name) == result.get("device_evidence"),
                     "Saved CLEAN execution device or precision evidence changed.")
        if source.get("scientific_identity_sha256"):
            from ec_clean30 import clean_sanity

            sanity_path = attempt / "clean_sanity.json"
            _require(_sha(sanity_path) == result.get("clean_sanity_sha256"), "CLEAN sanity evidence changed.")
            summary = clean_sanity.require_passed_or_disposed(sanity_path,
                        _unit_dispositions(request, sanity_path))
            fit_manifest = _json(attempt / "work" / clean.FIT_MANIFEST)
            _require(summary["checkpoint_sha256"] == fit_manifest["training_receipt"]["final_weights_sha256"],
                     "CLEAN sanity record belongs to another checkpoint.")
        export = clean.read_distance_export(export_path)
        spec = _unit(request, "clean")
        _require(export.receipt.get("seed") == spec["seed"] and export.receipt.get("regime") == spec["regime"]
                 and export.receipt.get("inner_fold") == spec["inner_fold"]
                 and _sha(export_path) == result["export_sha256"], "Saved CLEAN C4 export changed.")
    else:
        farm = Path(result["farm"])
        _require(_sha(farm / "farm_receipt.json") == result["producer_receipt_sha256"]
                 and _sha(farm / clean.FARM_MANIFEST) == result["farm_manifest_sha256"], "Embedding farm receipts changed.")
        observed = (_esmc_farm_evidence(farm) if _unit(request, "embedding")["embedding"] == "esmc"
                    else phase3._esm1b_gpu_evidence(farm))
        _require(all(result.get("device_evidence", {}).get(key) == value for key, value in observed.items()),
                 "Saved embedding device or precision evidence changed.")
        for entry in clean.read_farm_manifest(farm / clean.FARM_MANIFEST).values():
            _require(_sha(farm / f"{entry.label}.pt") == entry.file_sha256, "Completed embedding farm changed.")


def run_unit(request, *, kind):
    resource = "cpu" if kind == "clean" and request.get("provider") != "runpod" else "gpu"
    source, view, policy, authority, _ = _runtime(request, resource=resource, kind=kind)
    _unit(request, kind)
    root, attempt = Path(request["execution_root"]), _attempt(request)
    if source.get("scientific_identity_sha256"):
        _require(request.get("attempt", 1) in (1, 2), "A lost planned unit permits only one identical retry.")
        terminal = list(attempt.parent.glob("attempt*/unit_result.json"))
        _require(len(terminal) <= 1 and (not terminal or terminal[0].parent == attempt),
                 "A preserved result must be reused, never rerun or replaced.")
    remote_clean = kind == "clean" and request.get("provider") == "runpod"
    # Recovery must not acknowledge a canonical marker while its hot files remain uncopied.
    hot_storage.require_restored_attempts(root.resolve())
    require_restored_scientific_outputs(root.resolve())
    completed = root / "runs" / request["unit_id"] / "complete.json"
    with _execution(request, policy) as execution:
        if completed.exists():
            record = _json(completed)
            _verify_result(request, record, kind, source, view)
            pending = execution.state.get("pending_unit") or {}
            if (remote_clean and pending.get("run_name") == request["unit_id"] and pending.get("persisted")
                    and not execution.state.get("pending_transfer")):
                hot_storage.cleanup_hot_attempt(attempt / hot_storage.MARKER,
                                                host_ack_verified=True, result_verified=True)
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
        if remote_clean:
            _check_clean_hot_storage(request, view, authority)
        request_file = root / "commands" / f"{request['unit_id']}_attempt{request.get('attempt', 1)}.json"
        _write(request_file, dict(request))
        execution.admit(request["unit_id"], request["forecast_seconds"])
        marker = None
        if remote_clean:
            marker = hot_storage.prepare_hot_attempt(request["hot_work_root"], attempt,
                                                     **_hot_identity(request, source))["marker_path"]
        started = time.monotonic()
        logfile = request_file.with_suffix(".log")
        failure = None
        with logfile.open("x") as log:
            argv = [request["python"], str(REPO_ROOT / "scripts/run_ec_clean30_phase4.py"), f"_{kind}-worker",
                    "--request", str(request_file)]
            if kind == "clean":
                # The worker's envelope must accommodate the map; training/export have their own stricter child cap.
                worker_cap = request["distance_cap_prefix"]
                if request.get("provider") == "runpod":
                    profile = request["clean_cap_profile"]
                    if profile["memory_bytes"] > profile["distance_map_memory_bytes"]:
                        worker_cap = request["cap_prefix"]
                argv = [*worker_cap, *argv]
            try:
                env = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
                status = execution.run_subprocess(argv, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                  timeout=request["unit_timeout_seconds"])
            except Exception as exc:
                # run_subprocess terminates and reaps its child group before returning an error.
                failure = exc
                status = None
            finally:
                if marker is not None:
                    hot_storage.restore_hot_attempt(marker, worker_terminated=True)
        passed = (status is not None and status.returncode == 0 and result_path.is_file()
                  and _json(result_path).get("status") == "passed")
        execution.record_result(request["unit_id"], "completed" if passed else "failed_fit", time.monotonic()-started)
        artifacts = [request_file, logfile]
        if attempt.exists():
            artifacts += [path for path in attempt.rglob("*") if path.is_file() and not path.is_symlink()]
        persistence = execution.persist(artifacts)
        _require(passed, "Unit failed; its attempt is preserved. Diagnose before an original-seed retry."
                 + (f" {type(failure).__name__}: {failure}" if failure is not None else ""))
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
    if source.get("scientific_identity_sha256"):
        record.update(phase3.source_identity(source), git_head=source["git_head"], created_utc=provenance.utc_now())
    _write(out / "clean_inputs.json", record)
    return record


def prepare_clean_final_inputs(request):
    """Build final-arm training inputs only; publication-bound table receipts authenticate the full files."""
    source, view = _source_view(request)
    _require(source.get("scientific_identity_sha256"), "Final CLEAN preparation requires amendment 2.")
    official, table_receipt = Path(request["official_train"]), Path(request["official_training_receipt"])
    for path in (official, table_receipt):
        _guard(path, view)
    phase3._bound_pretrain_file(source, f"inputs.f6_corpus.tables.{view.fold}", table_receipt)
    receipt = _json(table_receipt)
    binding = receipt.get("inputs", {}).get("official_training_file", {})
    _require(receipt.get("fold") == view.fold and receipt.get("smoke") is False
             and official.name == f"split30_train_split_{view.fold}.csv"
             and binding.get("file") == official.name and binding.get("sha256") == _sha(official),
             "The complete official training file differs from the published corpus table receipt.")
    overlay = Path(request["overlay"])
    manifest_path = overlay / view.overlay["manifest_file"]
    _require(_sha(manifest_path) == view.overlay["manifest_sha256"], "Frozen data overlay changed.")
    pins = {entry["path"]: entry["sha256"] for entry in _json(manifest_path)["files"]}
    relative = f"r2_lists/r2_allowlist_fold{view.fold}.txt"
    allow = phase3._safe_file(overlay, relative)
    _require(_sha(allow) == pins.get(relative), "Frozen final R2 training allowlist changed.")
    out = Path(request["out"])
    _guard(out, view)
    _require(not out.exists(), "Final CLEAN input closure is written once.")
    out.mkdir(parents=True)
    shutil.copyfile(official, out / official.name)
    shutil.copyfile(allow, out / allow.name)
    roots = [Path(root) for root in view.forbidden_read_roots]
    rows, _ = clean.read_official_training_rows(out / official.name, view.fold, roots)
    allowed = clean.read_accession_allowlist(out / allow.name, roots)[0]
    specs, masked = [], []
    for unit in clean_final_units(view.fold):
        chosen = clean.select_rows(rows, allowed) if unit["regime"] == "r2" else rows
        stem = unit["unit_id"]
        _, masks = clean.write_masked_copies(chosen, unit["seed"], out / f"{stem}_masked.fasta",
                                            out / f"{stem}_masked.tsv")
        masked.append(masks["manifest"])
        specs.append({**unit, "allowlist": allow.name if unit["regime"] == "r2" else None,
            "exclusions": None, "queries": None, "masked_manifest": masks["manifest"],
            "input_dim": clean.ESMC_DIM if unit["arm"] == "c_esmc" else clean.DEFAULT_INPUT_DIM,
            "embedding": clean.esmc_embedding_tag("final") if unit["arm"] == "c_esmc" else clean.DEFAULT_EMBEDDING,
            "training_accessions_sha256": clean.sha256_lines(sorted(row.accession for row in chosen))})
    with (out / "sequence_map.tsv").open("x", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(clean.SEQUENCE_MAP_COLUMNS)
        for row in sorted(rows, key=lambda row: row.accession):
            writer.writerow((row.accession, clean.sha256_bytes(row.sequence.encode()), len(row.sequence), 1, row.sequence))
    record = {"format": "deepmzyme_ec_clean30_clean_inputs_v2", "fold": view.fold, "clean_scope": "final",
        "source": source, **phase3.source_identity(source), "git_head": source["git_head"], "created_utc": provenance.utc_now(),
        "official_training_receipt_sha256": _sha(table_receipt), "membership_sha256": view.membership["sha256"],
        "units": specs, "masked_manifests": masked, "n_sequences": len(rows),
        "files": {path.name: _sha(path) for path in out.iterdir() if path.is_file()}}
    _write(out / "clean_inputs.json", record)
    return record


def _clean_inputs(request, source, view, authority):
    root = Path(request["clean_inputs"])
    _guard(root, view)
    path = root / "clean_inputs.json"
    scoped = authority.get("clean_inputs_sha256_by_scope")
    expected_sha = (scoped.get(f"fold{view.fold}_{request.get('clean_scope', 'inner')}") if isinstance(scoped, dict)
                    else authority.get("clean_inputs_sha256"))
    _require(_sha(path) == expected_sha, "CLEAN input closure differs from G3 authority.")
    record = _json(path)
    phase3.require_same_scientific_identity(record.get("source", {}), source)
    _require(record.get("membership_sha256") == view.membership["sha256"], "CLEAN input provenance differs.")
    expected_units = clean_final_units(view.fold) if request.get("clean_scope") == "final" else clean_units()
    _require(record.get("fold") == view.fold and {row["unit_id"] for row in record["units"]}
             == {row["unit_id"] for row in expected_units}, "CLEAN closure must cover its exact authorized unit set.")
    for unit in expected_units:
        recorded = next(row for row in record["units"] if row["unit_id"] == unit["unit_id"])
        _require(all(recorded.get(key) == value for key, value in unit.items()), "CLEAN closure changes a frozen fit.")
        dim = clean.ESMC_DIM if unit.get("arm") == "c_esmc" else clean.DEFAULT_INPUT_DIM
        embedding = clean.esmc_embedding_tag("final") if unit.get("arm") == "c_esmc" else clean.DEFAULT_EMBEDDING
        _require(recorded.get("input_dim", clean.DEFAULT_INPUT_DIM) == dim
                 and recorded.get("embedding", clean.DEFAULT_EMBEDDING) == embedding,
                 "CLEAN closure changes its arm's frozen embedding recipe.")
    for name, digest in record["files"].items():
        path = phase3._safe_file(root, name)
        _require(_sha(path) == digest, "Prepared CLEAN input changed.")
    return root, record


def check_hot_storage(root: Path, working_bytes: int, *, runpod=False):
    _require(type(working_bytes) is int and working_bytes > 0, "Declare the active CLEAN working-byte forecast.")
    parent = root.resolve()
    while not parent.exists():
        parent = parent.parent
    filesystem = subprocess.check_output(["findmnt", "-n", "-o", "FSTYPE", "--target", str(parent)], text=True).strip()
    allowed = {"ext4", "xfs", "overlay"} if runpod else {"ext4"}
    _require(filesystem in allowed, "Active CLEAN files must live on local POSIX storage (internal ext4 on the workstation).")
    _require(shutil.disk_usage(parent).free >= RESERVE_BYTES + math.ceil(1.25*working_bytes),
             "CLEAN staging would breach the 5-GiB free-space reserve including its 25% working margin.")


def _hot_identity(request, source):
    return {"session_id": request["session_id"], "unit_id": request["unit_id"],
            "attempt": request.get("attempt", 1),
            "execution_manifest_sha256": source["execution_manifest_sha256"], "g3_sha256": request["g3_sha256"]}


def _check_clean_hot_storage(request, view, authority):
    root = Path(request.get("hot_work_root", ""))
    _require(root.is_absolute() and root == root.resolve()
             and authority.get("hot_work_root") == str(root),
             "Runpod CLEAN needs its canonical hot_work_root bound by G3.")
    _guard(root, view)
    protected = [REPO_ROOT, view.root, Path(request["execution_root"]),
                 Path(request["python"]).parent.parent, Path(request["clean_python"]).parent.parent,
                 *(Path(request[key]) for key in ("clean_clone", "clean_inputs", "base_farm", "masked_farm"))]
    for mapping_name in ("farms_by_unit", "farms_by_arm"):
        mapping = request.get(mapping_name, {})
        _require(isinstance(mapping, dict), "CLEAN farm mapping must be an object.")
        for farms in mapping.values():
            _require(isinstance(farms, dict) and set(farms) == {"base_farm", "masked_farm"},
                     "CLEAN farm overrides name exactly the base and masked farms.")
            for value in farms.values():
                path = Path(value)
                _guard(path, view)
                protected.append(path)
    _require(all(root != path.resolve() and root not in path.resolve().parents and path.resolve() not in root.parents
                 for path in protected), "CLEAN hot storage overlaps source, runtime, view, inputs or canonical outputs.")
    check_hot_storage(root, request["working_bytes"], runpod=True)
    parents = []
    for path in (root, Path(request["execution_root"])):
        while not path.exists():
            path = path.parent
        parents.append(path)
    copies = 2 if parents[0].stat().st_dev == parents[1].stat().st_dev else 1
    required = RESERVE_BYTES + copies * math.ceil(1.25 * request["working_bytes"])
    _require(all(shutil.disk_usage(path).free >= required for path in parents),
             "CLEAN hot work and its canonical copy must both fit with the free-space reserve.")


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
        threads=request["threads"], extra_forbidden_roots=roots)
    if request.get("provider") == "runpod":
        options["device"] = "cuda"
    plan = clean.run_fit(**options, dry_run=True)
    map_steps = []
    for step in plan["steps"]:
        if step["step"] == "distance_map":
            env = clean.build_clean_env(clean_clone=Path(request["clean_clone"]), hash_seed=step["hash_seed"],
                                        threads=request["threads"], forbidden_roots=roots,
                                        device="cuda" if request.get("provider") == "runpod" else "cpu")
            code = clean.run_command([*request["distance_cap_prefix"], *step["command"]], cwd=work, env=env,
                              log_path=attempt / "distance_map.log")
            _require(code == 0, "CLEAN distance map failed under its separate memory cap.")
            map_steps.append({**step, "cap_prefix": request["distance_cap_prefix"],
                              "log_sha256": _sha(attempt / "distance_map.log")})
            if request.get("provider") == "runpod":
                map_steps[-1]["runtime_policy"] = phase3.runtime_precision.read_policy_log(
                    attempt / "distance_map.log", device="cuda")
    prepared = _json(work / clean.PREPARE_MANIFEST)
    verified_map = clean.verify_distance_map(work / "data/distance_map", prepared["training_name"],
        prepared["training_table"]["sha256"], base_farm_sha256=prepared["embedding_farm"]["base"]["farm_sha256"],
        input_dim=prepared["input_dim"], helper_sha256=clean.DISTANCE_MAP_HELPER_SHA256)
    _write(attempt / "distance_map_execution.json", {"status": "passed", "steps": map_steps,
           "verified_distance_map": verified_map, "memory_limit_bytes":
           request.get("clean_cap_profile", {}).get("distance_map_memory_bytes", CLEAN_MAP_MEMORY_BYTES)})
    # A tiny executable adapter preserves the existing runner APIs and the actual private interpreter.
    launcher = attempt / "capped_clean_python"
    launcher.write_text("#!/bin/sh\nexec " + shlex.join([*request["cap_prefix"], request["clean_python"]]) + ' "$@"\n')
    launcher.chmod(0o755)
    fit = clean.run_fit(**{**options, "clean_python": launcher})
    return fit, launcher


def _clean_device_evidence(attempt, *, export_name="validation.npz"):
    fit = clean.cuda_step_evidence(attempt / "work/logs/train.log", training=True)
    receipt = clean.export_side_files(attempt / export_name)["export_receipt"]
    _require(_json(receipt).get("runtime", {}).get("device") == "cpu",
             "The pinned CLEAN validation exporter must record actual CPU execution.")
    return {"fit": fit, "export_device": "cpu", "export_receipt_sha256": _sha(receipt)}


def _clean_sanity_commands(request, work, attempt, roots, *, arm, clean_python):
    """Finish the in-sample engineering check while the owned hot fit still exists."""
    runner = REPO_ROOT / "scripts/run_ec_clean30_clean_sanity.py"
    sample, export, out = (attempt / name for name in ("sanity_sample.txt", "sanity_in_sample.npz", "clean_sanity.json"))
    env = phase3.clean_repository_environment(dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"))
    for action, arguments in (
        ("sample", ["--work-dir", str(work), "--out", str(sample)]),
        ("in-sample", ["--work-dir", str(work), "--arm", arm, "--sample", str(sample), "--export", str(export),
                        "--policy-log", str(work / "logs/train.log"), "--policy-device", "cuda", "--out", str(out)]),
    ):
        if action == "in-sample":
            clean.run_export(work_dir=work, query_ids_file=sample, query_esm_dir=attempt / "farms/base", out=export,
                mode="validation", clean_python=clean_python, clean_clone=Path(request["clean_clone"]),
                export_script=Path(request["export_script"]), threads=request["threads"],
                extra_forbidden_roots=roots, allow_in_sample=True, device="cpu")
        with (attempt / f"sanity_{action}.log").open("x") as log:
            subprocess.run([*request["cap_prefix"], request["python"], str(runner), action, *arguments],
                cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    from ec_clean30 import clean_sanity

    clean_sanity.verify_record(out)
    return _sha(out)


def clean_worker(request):
    remote = request.get("provider") == "runpod"
    source, view, _, authority, _ = _runtime(request, resource="gpu" if remote else "cpu", kind="clean")
    inputs, closure = _clean_inputs(request, source, view, authority)
    spec = next(row for row in closure["units"] if row["unit_id"] == request["unit_id"])
    unit = _unit(request, "clean")
    canonical = _attempt(request)
    attempt = (hot_storage.worker_hot_attempt(canonical / hot_storage.MARKER, **_hot_identity(request, source))
               if remote else canonical)
    if remote:
        _require(authority.get("hot_work_root") == request.get("hot_work_root")
                 and attempt == Path(request["hot_work_root"]) / request["session_id"] / request["unit_id"]
                 / f"attempt{request.get('attempt', 1)}", "CLEAN worker hot placement differs from its approved request.")
    check_hot_storage(attempt, request["working_bytes"], runpod=remote)
    _require(0 < request["distance_map_bytes"] <= MAX_DISTANCE_BYTES, "Distance-map forecast exceeds 4.5 GB.")
    if not remote:
        attempt.mkdir(parents=True)
    roots = _clean_roots(view, request)
    final = request.get("clean_scope") == "final"
    official = inputs / f"split30_train_split_{view.fold}.csv"
    rows, _ = clean.read_official_training_rows(official, view.fold, roots)
    excluded = {} if final else clean.read_inner_exclusions(inputs / spec["exclusions"], view.fold, unit["inner_fold"], roots)[0]
    chosen = [row for row in rows if row.accession not in excluded] if spec["allowlist"] is None else clean.select_rows(
        rows, clean.read_accession_allowlist(inputs / spec["allowlist"], roots)[0])
    query = () if final else clean.read_accession_allowlist(inputs / spec["queries"], roots)[0]
    masks, _ = clean.read_masked_manifest(inputs / spec["masked_manifest"], roots)
    farms = request.get("farms_by_unit", {}).get(unit["unit_id"],
             request.get("farms_by_arm", {}).get(unit.get("arm"), request))
    _copy_farm(Path(farms["base_farm"]), attempt / "farms/base", {row.accession for row in chosen} | set(query), roots)
    _copy_farm(Path(farms["masked_farm"]), attempt / "farms/masked", {row["label"] for row in masks}, roots)
    work = attempt / "work"
    clean.prepare_fit(official_train_csv=official, fold=view.fold, regime=unit["regime"], seed=unit["seed"],
        work_dir=work, accession_allowlist=None if spec["allowlist"] is None else inputs / spec["allowlist"],
        accession_exclusions=None if final else inputs / spec["exclusions"], inner_fold=unit["inner_fold"],
        base_esm_dir=attempt / "farms/base", masked_esm_dir=attempt / "farms/masked", forbidden_roots=roots,
        input_dim=spec.get("input_dim", clean.DEFAULT_INPUT_DIM), embedding=spec.get("embedding", clean.DEFAULT_EMBEDDING))
    # Keep active vectors local and make ordinary completion independent of symlink traversal.
    for path in (work / "data/esm_data").iterdir():
        if path.is_symlink():
            original = path.resolve()
            path.unlink()
            os.link(original, path)
    started = time.monotonic()
    fit, capped_python = _clean_fit(request, work, attempt, roots)
    _require(all(path.stat().st_size <= MAX_DISTANCE_BYTES for path in (work / "data/distance_map").glob("*.pkl")),
             "A completed CLEAN distance map exceeds 4.5 GB.")
    export_path = attempt / ("sanity_in_sample.npz" if final else "validation.npz")
    if not final:
        clean.run_export(work_dir=work, query_ids_file=inputs / spec["queries"], query_esm_dir=attempt / "farms/base",
            out=export_path, mode="validation", clean_python=capped_python, clean_clone=Path(request["clean_clone"]),
            export_script=Path(request["export_script"]), threads=request["threads"], extra_forbidden_roots=roots,
            **({"device": "cpu"} if remote else {}))
        export = clean.read_distance_export(export_path)
        _require(set(export.query_ids) == set(query) and len(export.query_ids) == len(query)
                 and export.receipt.get("n_in_sample_queries") == 0, "CLEAN validation export query closure differs.")
    sanity_sha = None
    if source.get("scientific_identity_sha256"):
        sanity_sha = _clean_sanity_commands(request, work, attempt, roots,
                                           arm=unit.get("arm", unit["regime"]), clean_python=capped_python)
    result = {"status": "passed", "kind": "clean", "unit_id": unit["unit_id"], "source": source,
              "g3_sha256": request["g3_sha256"], "export": str((canonical / export_path.name).resolve()),
              "export_sha256": _sha(export_path),
              "fit_manifest_sha256": _sha(work / clean.FIT_MANIFEST), "recipe": fit["recipe"],
              "clean_python_launcher_sha256": _sha(capped_python),
              "distance_map_execution_sha256": _sha(attempt / "distance_map_execution.json"),
              "wall_seconds": time.monotonic()-started}
    if remote:
        result["device_evidence"] = _clean_device_evidence(attempt, export_name=export_path.name)
    if source.get("scientific_identity_sha256"):
        result.update(phase3.source_identity(source), git_head=source["git_head"], created_utc=provenance.utc_now(),
                      clean_sanity_sha256=sanity_sha)
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


def _embedding_mask_names(record, unit):
    return sorted({row["masked_manifest"] for row in record["units"]
                   if row["seed"] == unit["seed"]
                   and (row.get("arm") == "c_esmc") == (unit["embedding"] == "esmc")})


def _esmc_farm_evidence(farm):
    from ec_clean30 import runtime_precision
    from training.global_esm_features import MANIFEST_FILENAME, read_manifest

    receipt = _json(farm / clean.FARM_RECEIPT)
    cache = farm.parent / "sequence_cache"
    _require(receipt.get("action") == "esmc-farm" and receipt.get("dim") == clean.ESMC_DIM
             and receipt.get("layer") == "final"
             and receipt.get("embedding") == clean.esmc_embedding_tag("final")
             and receipt.get("farm_manifest_sha256") == _sha(farm / clean.FARM_MANIFEST)
             and receipt.get("source_cache", {}).get("dir") == str(cache), "Invalid ESMC farm identity.")
    manifest = read_manifest(cache, receipt["source_cache"]["manifest_sha256"])
    runtime = manifest.get("build", {}).get("runtime", {})
    _require(str(runtime.get("device", "")).startswith("cuda")
             and runtime.get("forward_precision") == "fp32" and runtime.get("parameter_dtype") == "float32"
             and runtime.get("autocast_dtype") is None and runtime.get("use_flash_attn") is False,
             "ESMC farm lacks CUDA fp32 runtime evidence.")
    log = farm.parent / "embedding.log"
    policy = runtime_precision.read_policy_log(log, device="cuda")
    return {"extraction_log_sha256": _sha(log), "precision_policy": policy,
            "cache_manifest_sha256": _sha(cache / MANIFEST_FILENAME)}


def _esmc_farm_worker(request, source, view, inputs, record, unit, directory, roots):
    # Use the pinned fp32 producer with its input guard and offline weight checks.
    sequences, _ = clean.read_sequence_map_file(inputs / "sequence_map.tsv", roots, record["files"]["sequence_map.tsv"])
    masks = _embedding_mask_names(record, unit) if unit["kind"] == "masked" else []
    labels, _ = clean.farm_labels(sequences, kind=unit["kind"], seed=unit["seed"],
                                 masked_manifests=[inputs / name for name in masks], forbidden_roots=roots)
    fasta = directory / "sequences.fasta"
    fasta.write_text("".join(f">{label}\n{sequence}\n" for label, (_, sequence) in sorted(labels.items())))
    cache, farm = directory / "sequence_cache", directory / "farm"
    argv, env = phase3.pretrain_embedding_command(request, source, view, fasta=fasta,
        sequence_map=inputs / "sequence_map.tsv", sequence_map_sha256=record["files"]["sequence_map.tsv"],
        output=cache, device="cuda")
    started = time.monotonic()
    with (directory / "embedding.log").open("x") as log:
        subprocess.run([*request["cap_prefix"], *argv], cwd=REPO_ROOT, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    from training.global_esm_features import MANIFEST_FILENAME
    argv = [request["python"], str(REPO_ROOT / "scripts/run_clean_split30_comparator.py"), "esmc-farm",
            "--cache-dir", str(cache), "--cache-manifest-sha256", _sha(cache / MANIFEST_FILENAME),
            "--layer", "final", "--sequence-map", str(inputs / "sequence_map.tsv"),
            "--sequence-map-sha256", record["files"]["sequence_map.tsv"], "--kind", unit["kind"], "--out-dir", str(farm)]
    if unit["seed"] is not None:
        argv += ["--seed", str(unit["seed"])]
    for name in masks:
        argv += ["--masked-manifest", str(inputs / name)]
    for root in roots:
        argv += ["--forbidden-root", str(root)]
    with (directory / "farm.log").open("x") as log:
        subprocess.run([*request["cap_prefix"], *argv], cwd=REPO_ROOT, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    result = {"status": "passed", "kind": "embedding", "unit_id": unit["unit_id"], "source": source,
              "g3_sha256": request["g3_sha256"], "farm": str(farm.resolve()),
              "device_evidence": _esmc_farm_evidence(farm),
              "producer_receipt_sha256": _sha(farm / clean.FARM_RECEIPT),
              "farm_manifest_sha256": _sha(farm / clean.FARM_MANIFEST), "wall_seconds": time.monotonic()-started,
              **phase3.source_identity(source), "git_head": source["git_head"], "created_utc": provenance.utc_now()}
    _write(directory / "unit_result.json", result)
    return result


def embedding_worker(request):
    source, view, _, authority, _ = _runtime(request, resource="gpu", kind="embedding")
    inputs, record = _clean_inputs(request, source, view, authority)
    unit = _unit(request, "embedding")
    kind = unit["kind"]
    directory = _attempt(request)
    directory.mkdir(parents=True)
    roots = _clean_roots(view, request)
    if unit["embedding"] == "esmc":
        return _esmc_farm_worker(request, source, view, inputs, record, unit, directory, roots)
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
    argv = [request["embedding_python"], str(REPO_ROOT / "src/ec_clean30/runtime_precision.py"), "esm1b-farm",
            "--harness", request["harness"], "--", "--runner", str(REPO_ROOT / "scripts/run_clean_split30_comparator.py"),
            "--extract-script", request["extract_script"], "--extract-sha256", helpers["extract_script"],
            "--weights", str(weights), "--weights-sha256", request["weights_sha256"], "--sequence-map", str(inputs / "sequence_map.tsv"),
            "--sequence-map-sha256", record["files"]["sequence_map.tsv"], "--kind", kind,
            "--out-dir", str(out), "--device", "cuda", "--threads", str(request["threads"])]
    if kind == "masked":
        argv += ["--seed", str(unit["seed"])]
        for name in _embedding_mask_names(record, unit):
            argv += ["--masked-manifest", str(inputs / name)]
    for root in roots:
        argv += ["--forbidden-root", str(root)]
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1",
               NVIDIA_TF32_OVERRIDE="0")
    env = {name: value for name, value in env.items() if not name.startswith("GIT_")}
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
    if source.get("scientific_identity_sha256"):
        result.update(phase3.source_identity(source), git_head=source["git_head"], created_utc=provenance.utc_now())
    _write(directory / "unit_result.json", result)
    return result


def _attach_session(request, transport, *, current=None):
    from ec_clean30 import workstation

    if request.get("provider") == "runpod":
        _require(isinstance(transport, workstation.RunpodTransport), "Runpod Phase 4 needs its owned Runpod transport.")
        controller = transport.stop_controller
        _require(request.get("runpod_config_sha256") == controller.config_sha,
                 "Phase 4 belongs to another immutable Runpod configuration.")
        current = controller.session(request["session_id"]) if current is None else current
        _require(all(key in request and current.get(key) == request[key] for key in
                     ("pod_id", "session_id", "allocation_started_unix", "deadline_unix", "max_total_seconds")),
                 "Runpod allocation identity or original deadline differs from Phase 4 admission.")
        return current
    _require(request.get("provider", "gcp") == "gcp", "Unknown Phase 4 transport provider.")
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

    if request.get("provider") == "runpod":
        _require(isinstance(transport, workstation.RunpodTransport), "Runpod Phase 4 needs its owned Runpod transport.")
        controller = transport.stop_controller
        current = controller._cleanup_state()
        _require(request.get("runpod_config_sha256") == controller.config_sha
                 and all(current.get(key) == request.get(key) for key in ("pod_id", "session_id")),
                 "Another allocation owns this Runpod endpoint; refusing a blind stop.")
        receipt = controller.stop()
        _require(receipt.get("runtime_status") == "stopped", "Runpod shutdown is not verified.")
        return {"session_id": request["session_id"], "pod_id": request["pod_id"],
                "provider_stop_verified": True, "verified_unix": time.time()}
    workstation._identity(request, workstation._controller_config(transport.controller_root))
    current = _json(transport.controller_root / "state/current_session.json")
    _require(not current.get("active") or current.get("session_id") == request["session_id"],
             "Another allocation owns this endpoint; refusing a blind stop.")
    transport.controller("stop", timeout=240)
    status = transport.controller("status")
    _require(workstation._status(status, request["expected_identity"]) == "TERMINATED", "Provider shutdown is not verified.")
    return {"session_id": request["session_id"], "provider_stop_verified": True, "verified_unix": time.time()}


def _run_remote_queue(request, units, *, action, next_gate, transport=None, check_session=None, close_session=True):
    """One owned allocation, all remaining exact units; admission refuses work that needs another approved session."""
    from ec_clean30.workstation import ControllerTransport, HostPull, complete_remote_unit

    source = _request_source(request)
    verify_authority(request, source, resource="gpu")
    if source.get("scientific_identity_sha256"):
        _require(request.get("provider") == "runpod" and transport is not None,
                 "Execution v5 queues require the coordinator's existing Runpod session.")
    _require(close_session or (transport is not None and callable(check_session)),
             "Borrowed Phase 4 ownership requires a live transport and its session check.")
    if transport is None:
        _require(request.get("provider", "gcp") == "gcp", "Runpod Phase 4 must reuse the coordinator's live transport.")
        transport = ControllerTransport(Path(request["controller_root"]), request["ssh_alias"],
                                        bootstrap_python=request.get("bootstrap_python", "/usr/bin/python3"))
    previous_deadline = getattr(transport, "deadline_unix", None)
    previous_reserve = getattr(transport, "reserve_seconds", 30)
    transport.deadline_unix = request["deadline_unix"]
    transport.reserve_seconds = 900
    host = HostPull(transport, Path(request["durable_root"]), Path(request["host_state_root"]),
                   max_download_bytes=request["max_download_bytes"], deadline_unix=request["deadline_unix"],
                   mirror_root=Path(request["mirror_root"]) if request.get("mirror_root") else None)
    _require(re.fullmatch(r"[A-Za-z0-9_.-]+", request["session_id"]), "Unsafe allocation identifier.")
    completed = []
    owned = False
    def attach():
        if check_session is None:
            return _attach_session(request, transport)
        return _attach_session(request, transport, current=check_session())
    try:
        attach()
        owned = True
        for unit in units:
            attach()
            name = unit["unit_id"]
            current = {**request["remote_request_template"], **unit,
                       "attempt": request.get("attempts", {}).get(name, 1)}
            _require(all(current.get(key) == request.get(key) for key in
                     ("session_id", "allocation_started_unix", "deadline_unix", "max_total_seconds", "g3_sha256")),
                     "Remote request allocation differs from the controller-verified G3 allocation.")
            if source.get("scientific_identity_sha256"):
                _require(current.get("expected_chain") == source["amendment_chain"]
                         and isinstance(current.get("amendment_chain_receipt"), str)
                         and bool(current["amendment_chain_receipt"]),
                         "Remote Phase-4 request lacks the authenticated session chain receipt.")
            if request.get("provider") == "runpod":
                _require(all(current.get(key) == request.get(key) for key in
                             ("provider", "pod_id", "runpod_config_sha256")),
                         "Remote request changes the Runpod provider or pod binding.")
                transport.stop_controller.admit_output_root(current["execution_root"])
            result = complete_remote_unit(transport, host, current, runner=request["remote_runner"], action=action,
                remote_request=f"{current['execution_root']}/queue/{request['session_id']}/{name}_attempt{current['attempt']}.json",
                completed_relative=f"runs/{name}/complete.json", timeout=current["unit_timeout_seconds"]+60)
            _require(result.get("status") == "passed" and result.get("unit_id") == name,
                     "Queue stops at an incomplete terminal unit.")
            if source.get("scientific_identity_sha256"):
                phase3.require_same_scientific_identity(result, source)
                phase3.require_record_precedence(result, source)
            completed.append(name)
    finally:
        try:
            if owned:
                try:
                    host.sync_pending(request["remote_request_template"]["execution_root"])
                finally:
                    if close_session:
                        receipt = _close_session(request, transport)
                        _write(Path(request["host_state_root"]) / f"closeout_{request['session_id']}.json", receipt)
        finally:
            if not close_session:
                transport.deadline_unix = previous_deadline
                transport.reserve_seconds = previous_reserve
    return {"status": "passed", "completed": completed, "next_gate": next_gate}


def run_fold_attachment(request, **transport_options):
    unit = {"unit_id": f"ecc30_pretrain_fold{request['fold']}_attempt1",
            "forecast_seconds": request["attachment_forecast_seconds"],
            "unit_timeout_seconds": request["attachment_timeout_seconds"]}
    return _run_remote_queue(request, [unit], action="attach-fold", next_gate="s1", **transport_options)


def attach_fold(request):
    """Attach the preserved union cache under the same persistence protocol as F6 preparation."""
    source, view, policy, _, _ = _runtime(request, resource="gpu", kind="s1")
    _require(view.fold > 0 and request["unit_id"] == phase3._pretrain_name(request), "Invalid later-fold attachment.")
    root = Path(request["execution_root"])
    out = root / "runs" / request["unit_id"]
    complete, terminal = out / "complete.json", out / "attachment_result.json"
    with _execution(request, policy) as execution:
        if complete.exists():
            result = _json(complete)
            phase3.require_record_precedence(result, source)
            phase3.attach_pretrain(request)
            return result
        if terminal.exists():
            result = _json(terminal)
            phase3.require_record_precedence(result, source)
            phase3.verify_preserved_pretrain_attachment(out / phase3.PRETRAIN_ATTACHMENT_MARKER, original_attempt=out)
            pending = execution.state.get("pending_unit", {})
            _require(pending.get("run_name") == request["unit_id"] and pending.get("persisted"),
                     "Fold attachment awaits independent persistence.")
            _write(complete, result)
            execution.persist([complete])
            return result
        _require(not out.exists(), "Partial fold attachment requires explicit recovery.")
        command = root / "commands" / f"{request['unit_id']}.json"
        _write(command, request)
        execution.admit(request["unit_id"], request["forecast_seconds"])
        marker = phase3.prepare_pretrain_attachment(out, request=request, source=source, view=view, request_path=command)
        started = time.monotonic()
        passed = False
        try:
            with command.with_suffix(".log").open("x") as log:
                result = execution.run_subprocess([request["python"], str(REPO_ROOT / "scripts/run_ec_clean30_phase4.py"),
                    "_attach-fold-worker", "--request", str(command)], cwd=REPO_ROOT,
                    env=phase3.clean_repository_environment(), stdout=log, stderr=subprocess.STDOUT,
                    timeout=request["unit_timeout_seconds"])
            passed = result.returncode == 0 and terminal.is_file()
        finally:
            phase3.restore_pretrain_attachment(marker, worker_terminated=True, termination_proof={
                "kind": "owned_process_group_reaped", "parent_pid": os.getpid(), "session_id": request["session_id"],
                "request_sha256": _sha(command)}, execution=execution)
            execution.record_result(request["unit_id"], "completed" if passed else "failed_preparation", time.monotonic()-started)
            persistence = execution.persist([command, command.with_suffix(".log"), out])
        _require(passed, "Fold attachment failed; its bytes are preserved.")
        return {"status": "awaiting_host_ack", "unit_id": request["unit_id"], "persistence": persistence}


def attachment_worker(request):
    source, view, _, _, _ = _runtime(request, resource="gpu", kind="s1")
    out = Path(request["execution_root"]) / "runs" / request["unit_id"]
    phase3._require_pretrain_worker_marker(out, request, source, view)
    result = {**phase3.attach_pretrain(request), "unit_id": request["unit_id"]}
    _write(out / "attachment_result.json", result)
    return result


def run_s1_queue(request, **transport_options):
    units = [{"unit_id": unit.unit_id, "forecast_seconds": request["family_forecasts"][unit.family],
              "unit_timeout_seconds": request["family_timeouts"][unit.family]}
             for unit in s1_units(_json(Path(request["local_plan"])))]
    return _run_remote_queue(request, units, action="s1-unit", next_gate="S1_selection_and_G4", **transport_options)


def run_refit_queue(request, **transport_options):
    units = [{"unit_id": unit.unit_id, "forecast_seconds": request["refit_forecast_seconds"],
              "unit_timeout_seconds": request["refit_timeout_seconds"]}
             for unit in fold_units(_json(Path(request["local_plan"])), "S4")]
    return _run_remote_queue(request, units, action="refit-unit", next_gate="frozen_refits_complete", **transport_options)


def run_embedding_queue(request, **transport_options):
    units = [{"unit_id": unit["unit_id"],
              "forecast_seconds": request["embedding_forecasts"][unit["unit_id"] if request.get("clean_scope") == "final" else unit["kind"]],
              "unit_timeout_seconds": request["embedding_timeouts"][unit["unit_id"] if request.get("clean_scope") == "final" else unit["kind"]]}
             for unit in embedding_units(request)]
    return _run_remote_queue(request, units, action="clean-embeddings", next_gate="eight_serial_CLEAN_inner_fits", **transport_options)


def run_clean_remote_queue(request, **transport_options):
    _require(request.get("provider") == "runpod" and request["remote_request_template"].get("device") == "cuda",
             "Remote CLEAN fits require the Runpod GPU route.")
    planned = clean_final_units(request["fold"]) if request.get("clean_scope") == "final" else clean_units()
    units = [{"unit_id": unit["unit_id"], "forecast_seconds": request["clean_forecasts"][unit["unit_id"]],
              "unit_timeout_seconds": request["clean_timeouts"][unit["unit_id"]]} for unit in planned]
    return _run_remote_queue(request, units, action="clean-unit", next_gate="S1_selection_and_G4", **transport_options)


SCIENTIFIC_STEPS = ("held_out_sanity", "verify_sanity", "select_s1", "select_s2", "futility", "select_s4")
SCIENTIFIC_PENDING = "scientific_outputs_pending.json"


def _scientific_outputs(request):
    step, inputs = request["scientific_step"], request["scientific_inputs"]
    if step == "held_out_sanity":
        return [Path(inputs["clean_sanity_record"])]
    if step == "verify_sanity":
        return [Path(inputs["clean_sanity_verify_all"])]
    result = [Path(request["step_chain"])]
    if step == "futility":
        from ec_clean30 import futility
        result += [Path(inputs["futility_out"]),
                   futility.pointer_path(Path(request["campaign_root"]), len(request["expected_chain"]))]
    return result


def _science_preservation_paths(request):
    from ec_clean30 import futility
    paths = _scientific_outputs(request)
    for sequence in range(2, len(request["expected_chain"]) + 1):
        pointer = futility.pointer_path(Path(request["campaign_root"]), sequence)
        if pointer not in paths:
            paths.append(pointer)
    return paths


def _prepare_scientific_outputs(request, source, request_path):
    attempt = _attempt(request)
    attempt.mkdir(parents=True)
    shutil.copyfile(request_path, attempt / "request.json")
    identity = {"canonical_attempt": str(attempt.resolve()), "unit_id": request["unit_id"],
                "scientific_step": request["scientific_step"], "request_sha256": _sha(request_path),
                **phase3.source_identity(source), "outputs": [str(path) for path in _science_preservation_paths(request)]}
    _write(attempt / SCIENTIFIC_PENDING, {"format": "deepmzyme_ec_clean30_scientific_preservation_v1",
            "identity": identity, "identity_sha256": campaign.canonical_sha256(identity), "status": "pending"})


def require_restored_scientific_outputs(root):
    for path in Path(root).glob(f"runs/*/attempt*/{SCIENTIFIC_PENDING}"):
        _require(not path.is_symlink() and _json(path).get("status") == "restored",
                 "Pending scientific outputs block automatic recovery and further admission.")
        verify_preserved_scientific_outputs(path, original_attempt=path.parent)


def _restore_scientific_outputs(attempt):
    """Called by the owning parent only after its child process group has terminated."""
    marker = attempt / SCIENTIFIC_PENDING
    record = _json(marker)
    _require(record.get("status") == "pending", "Scientific restoration is owned once by the terminating parent.")
    files = []
    for index, value in enumerate(record["identity"]["outputs"]):
        path = Path(value)
        _require(not path.is_symlink(), "A scientific output became a symlink.")
        entry = {"path": value, "exists": path.exists()}
        if path.exists():
            _require(path.is_file(), "A scientific output is not a regular file.")
            snapshot = attempt / f"preserved_output{index}{path.suffix}"
            shutil.copyfile(path, snapshot)
            _require(_sha(path) == _sha(snapshot), "Scientific output changed during terminated-child restoration.")
            entry.update(snapshot=snapshot.name, sha256=_sha(snapshot), size_bytes=snapshot.stat().st_size)
        files.append(entry)
    record.update(status="restored", files=files)
    temporary = marker.with_suffix(".restored")
    _write(temporary, record)
    os.replace(temporary, marker)
    verify_preserved_scientific_outputs(marker, original_attempt=attempt)
    return record


def verify_preserved_scientific_outputs(local_marker, *, original_attempt):
    """Validate copied scientific snapshots without reading any recorded remote path."""
    marker = Path(local_marker)
    _require(marker.is_file() and not marker.is_symlink(), "Scientific preservation marker is not a regular file.")
    record = _json(marker)
    identity = record.get("identity", {})
    original = PurePosixPath(str(original_attempt))
    _require(original.is_absolute() and ".." not in original.parts
             and record.get("format") == "deepmzyme_ec_clean30_scientific_preservation_v1"
             and record.get("status") == "restored" and identity.get("canonical_attempt") == str(original)
             and record.get("identity_sha256") == campaign.canonical_sha256(identity),
             "Scientific preservation identity or completion differs.")
    request_path = marker.parent / "request.json"
    _require(request_path.is_file() and not request_path.is_symlink()
             and _sha(request_path) == identity.get("request_sha256"), "Preserved scientific request changed.")
    request = _json(request_path)
    expected_attempt = PurePosixPath(request["execution_root"]) / "runs" / request["unit_id"] / f"attempt{request.get('attempt', 1)}"
    _require(original == expected_attempt and identity.get("unit_id") == request["unit_id"]
             and identity.get("scientific_step") == request["scientific_step"]
             and identity.get("outputs") == [str(path) for path in _science_preservation_paths(request)]
             and re.fullmatch(r"[0-9a-f]{64}", identity.get("scientific_identity_sha256", ""))
             and isinstance(identity.get("fit_chain"), list) and len(identity["fit_chain"]) >= 2
             and all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in identity["fit_chain"])
             and identity["fit_chain"] == request["expected_chain"][:len(identity["fit_chain"])],
             "Scientific preservation request or fit identity differs.")
    files = record.get("files")
    _require(isinstance(files, list) and [row.get("path") for row in files] == identity["outputs"],
             "Scientific preservation inventory is incomplete.")
    for entry in files:
        _require(type(entry.get("exists")) is bool, "Scientific preservation needs explicit absence evidence.")
        if entry["exists"]:
            name = entry.get("snapshot", "")
            _require(bool(name) and PurePosixPath(name).name == name, "Unsafe scientific snapshot name.")
            path = marker.parent / name
            _require(path.is_file() and not path.is_symlink() and path.stat().st_size == entry.get("size_bytes")
                     and _sha(path) == entry.get("sha256"), "Preserved scientific output bytes differ.")
        else:
            _require(set(entry) == {"path", "exists"}, "An absent scientific output carries contradictory bytes.")
    return record


def _overlay_components(value, view):
    """The one scientific input inside a restricted-name folder: the bound view's frozen data-overlay copy.

    The real selector and futility steps read only <overlay>/identity/fold<K> next to <overlay>/overlay_manifest.json,
    so this accepts exactly <campaign root>/overlay/<frozen overlay id>/identity/fold<K>: canonical (no link, no
    '..'), case-exact, this fold only, never the campaign's own identity folder and never a test-named part.
    """
    overlay = getattr(view, "overlay", None) or {}
    _require(isinstance(value, str) and not any(c in value for c in "\n\r\0")
             and isinstance(overlay.get("id"), str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", overlay["id"]),
             "Scientific components need the frozen data overlay of a real fold view.")
    path, campaign_root = Path(value), Path(view.campaign_root).resolve()
    expected = campaign_root / "overlay" / overlay["id"] / "identity" / f"fold{view.fold}"
    _require(path.is_absolute() and ".." not in path.parts and str(path) == value and path.resolve() == path
             and path == expected and path.parts[-2:] == ("identity", f"fold{view.fold}"),
             "Scientific components must be the canonical frozen overlay copy "
             "<campaign root>/overlay/<overlay id>/identity/fold<K>.")
    manifest = path.parent.parent / views.OVERLAY_MANIFEST_NAME
    _require(manifest.is_file() and not manifest.is_symlink(),
             "The overlay identity copy needs its regular overlay manifest two folders up.")
    provenance.refuse_test_paths([path], what="Scientific components", relative_to=campaign_root)
    selector.check_components_path(view.fold, path, smoke=False)


def _science_paths(request, view):
    inputs = request["scientific_inputs"]
    _require(isinstance(inputs, dict) and set(inputs) <= {"components", "clean_sanity_record",
        "clean_sanity_dispositions", "clean_sanity_verify_all", "futility_out", "f6_corpus", "r1_inner_lists"},
        "Scientific dispatch accepts declared input paths only, not policy or command overrides.")
    for name, value in inputs.items():
        paths = value if isinstance(value, list) else [value]
        _require(all(isinstance(path, str) and bool(path) for path in paths), "Scientific inputs must be paths.")
        if name == "components":
            _overlay_components(value, view)
            continue
        for path in paths:
            phase3._no_restricted_path(Path(path))
    root = Path(request["execution_root"]).resolve()
    for name in ("clean_sanity_record", "clean_sanity_verify_all", "futility_out"):
        if name in inputs:
            _require(root in Path(inputs[name]).resolve().parents, "Scientific outputs must stay in the owned execution root.")
    _require(root in Path(request["step_chain"]).resolve().parents,
             "The current fold's step chain must stay in its owned execution root.")
    return inputs


def _science_cap(request, authority):
    from ec_clean30 import container_caps

    profile = request.get("science_cap_profile")
    _require(isinstance(profile, dict) and set(profile) == {"cgroup_root", "cpu_cores", "memory_bytes"}
             and profile == authority.get("science_cap_profile")
             and all(type(profile[key]) is int and profile[key] > 0 for key in ("cpu_cores", "memory_bytes")),
             "Scientific CPU caps differ from numerical G3.")
    return container_caps.cap_prefix(request["python"], REPO_ROOT, profile["cgroup_root"],
                                     cpu_cores=profile["cpu_cores"], memory_bytes=profile["memory_bytes"])


def _completed_fit_inputs(request, source, view):
    runs = []
    for unit in s1_units(_json(Path(request["plan"]))):
        root = Path(request["execution_root"]) / "runs" / unit.unit_id
        record = _json(root / "complete.json")
        _verify_result({**request, "unit_id": unit.unit_id}, record, "s1", source, view)
        runs.append(record["runs_dir"])
    return runs


def _completed_clean_inputs(request, source, view):
    fits, sanity, exports = [], [], {"r1_inner": [], "r2_inner": []}
    root = Path(request.get("clean_execution_root", request["execution_root"]))
    closure = Path(request["clean_inputs"])
    for unit in clean_units():
        record = _json(root / "runs" / unit["unit_id"] / "complete.json")
        _verify_result({**request, "unit_id": unit["unit_id"], "clean_scope": "inner"}, record, "clean", source, view)
        attempt = Path(record["export"]).parent
        fits.append({"arm": unit["regime"], "inner_fold": unit["inner_fold"], "work_dir": str(attempt / "work"),
                     "export": record["export"], "policy_log": str(attempt / "work/logs/train.log"), "policy_device": "cuda"})
        sanity.append(str(attempt / "clean_sanity.json"))
        exports[unit["regime"]].append(record["export"])
    spec = {"official_training_file": str(closure / "split30_train_split_0.csv"), "fits": fits,
            "query_farm": request["base_farm"], "inner_exclusions": {str(inner): str(closure /
                  f"r1_inner_exclusions_fold0_inner{inner}.tsv") for inner in range(4)}}
    return spec, sanity, exports


def _science_command(request, source, view, attempt):
    step = request["scientific_step"]
    _require(step in SCIENTIFIC_STEPS, "Unknown scientific dispatch step.")
    inputs = _science_paths(request, view)
    runner = [request["python"], str(REPO_ROOT / "scripts/run_ec_clean30_campaign.py")]
    outputs = []
    if step in ("held_out_sanity", "verify_sanity"):
        _require(view.fold == 0, "CLEAN held-out sanity is a fold-0 hook only.")
        spec, records, _ = _completed_clean_inputs(request, source, view)
        runner = [request["python"], str(REPO_ROOT / "scripts/run_ec_clean30_clean_sanity.py")]
        if step == "held_out_sanity":
            spec_path = attempt / "held_out_spec.json"
            _write(spec_path, spec)
            out = inputs["clean_sanity_record"]
            argv = [*runner, "held-out", "--spec", str(spec_path), "--out", out]
        else:
            out = inputs["clean_sanity_verify_all"]
            argv = [*runner, "verify-all", "--out", out]
            for path in [*records, inputs["clean_sanity_record"]]:
                argv += ["--record", path]
            for path in _dispositions_for([*records, inputs["clean_sanity_record"]],
                                          inputs.get("clean_sanity_dispositions", [])):
                argv += ["--disposition", str(path)]
        outputs.append(Path(out))
    else:
        common = ["--fold", str(view.fold), "--plan", request["plan"], "--components", inputs["components"],
            "--freeze1-sha256", source["freeze1_sha256"], "--freeze1-amendment", str(REPO_ROOT / source["amendment_path"]),
            "--amendment-chain-receipt", source["amendment_chain_receipt"], "--campaign-root", request["campaign_root"]]
        for path in _completed_fit_inputs(request, source, view):
            common += ["--runs", path]
        if view.fold == 0 and step in ("select_s1", "futility"):
            common += ["--clean-sanity-record", inputs["clean_sanity_record"]]
            for path in _dispositions_for([inputs["clean_sanity_record"]],
                                          inputs.get("clean_sanity_dispositions", [])):
                common += ["--clean-sanity-disposition", str(path)]
        outputs.append(Path(request["step_chain"]))
        if step == "futility":
            _require(view.fold == 0, "The binding futility decision is written in fold 0 only.")
            _, _, exports = _completed_clean_inputs(request, source, view)
            argv = [*runner, "futility", *common, "--step-chain", request["step_chain"],
                    "--f6-corpus", inputs["f6_corpus"], "--out", inputs["futility_out"]]
            for regime, paths in exports.items():
                for path in paths:
                    argv += ["--clean-" + regime.replace("_", "-"), path]
            _require(len(inputs["r1_inner_lists"]) == 4, "Binding futility needs all four R1-inner exclusion lists.")
            for path in inputs["r1_inner_lists"]:
                argv += ["--r1-inner-list", path]
            from ec_clean30 import futility
            outputs += [Path(inputs["futility_out"]), futility.pointer_path(Path(request["campaign_root"]),
                                                                          len(source["amendment_chain"]))]
        else:
            selected_step = {"select_s1": "S1", "select_s2": "S2", "select_s4": "S4"}[step]
            argv = [*runner, "select", *common, "--step", selected_step, "--out", request["step_chain"]]
            if view.fold != 0:
                argv += ["--fold0-step-chain", request["fold0_step_chain"]]
    return argv, outputs


def _verify_scientific_result(request, result, source):
    phase3.require_same_scientific_identity(result, source)
    phase3.require_record_precedence(result, source)
    _require(result.get("status") == "passed" and result.get("unit_id") == request["unit_id"]
             and result.get("scientific_step") == request["scientific_step"], "Scientific result identity differs.")
    attempt = _attempt(request)
    step = request["scientific_step"]
    _require(isinstance(result.get("snapshots"), list) and len(result["snapshots"]) == (3 if step == "futility" else 1),
             "Scientific output snapshot set is incomplete.")
    _require([row.get("path") for row in result["snapshots"]] == [str(path) for path in _scientific_outputs(request)],
             "Scientific output snapshot paths differ from the declared action.")
    for binding in result["snapshots"]:
        path = phase3._safe_file(attempt, binding["snapshot"])
        _require(_sha(path) == binding["sha256"], "Scientific output snapshot changed.")
    if step in ("held_out_sanity", "verify_sanity"):
        from ec_clean30 import clean_sanity
        inputs = request["scientific_inputs"]
        clean_sanity.held_out_binding(inputs["clean_sanity_record"],
            _dispositions_for([inputs["clean_sanity_record"]], inputs.get("clean_sanity_dispositions", [])))
        if step == "verify_sanity":
            _require(not _json(Path(inputs["clean_sanity_verify_all"])).get("stopped"),
                     "CLEAN sanity has a binding stop disposition.")
    if step == "futility":
        from ec_clean30 import futility
        decision = futility.verify_futility_decision(Path(request["fold0_step_chain"]),
            freeze1_sha256=source["freeze1_sha256"], amendment_tip=REPO_ROOT / source["amendment_path"],
            campaign_root=Path(request["campaign_root"]), chain_receipt=Path(source["amendment_chain_receipt"]),
            expected_chain=source["amendment_chain"])
        _require(decision["decision"] == result["decision"], "The binding futility decision changed.")


def scientific_worker(request):
    kind = "science_futility" if request["scientific_step"] == "futility" else "science"
    source, view, _, authority, _ = _runtime(request, resource="gpu", kind=kind)
    _require(source.get("scientific_identity_sha256"), "Scientific dispatch requires execution v5.")
    attempt = _attempt(request)
    _require((attempt / SCIENTIFIC_PENDING).is_file()
             and _json(attempt / SCIENTIFIC_PENDING).get("status") == "pending",
             "Scientific worker needs its parent's pending preservation marker.")
    argv, outputs = _science_command(request, source, view, attempt)
    _require(outputs == _scientific_outputs(request), "Scientific outputs differ from their preserved path set.")
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
    cap = _science_cap(request, authority)
    env = phase3.clean_repository_environment(dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"))
    snapshots = []
    started = time.monotonic()
    try:
        with (attempt / "science.log").open("x") as log:
            subprocess.run([*cap, *argv], cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    finally:
        for index, path in enumerate(outputs):
            if path.is_file() and not path.is_symlink():
                snapshot = attempt / f"output{index}{path.suffix}"
                shutil.copyfile(path, snapshot)
                snapshots.append({"path": str(path), "snapshot": snapshot.name, "sha256": _sha(snapshot)})
    _require(len(snapshots) == len(outputs), "A scientific action did not produce its required outputs.")
    result = {"status": "passed", "unit_id": request["unit_id"], "kind": "scientific",
              "scientific_step": request["scientific_step"], "snapshots": snapshots,
              **phase3.source_identity(source), "git_head": source["git_head"], "created_utc": provenance.utc_now(),
              "wall_seconds": time.monotonic() - started}
    if request["scientific_step"] == "futility":
        from ec_clean30 import futility
        result.update(futility.verify_futility_decision(Path(request["fold0_step_chain"]),
            freeze1_sha256=source["freeze1_sha256"], amendment_tip=REPO_ROOT / source["amendment_path"],
            campaign_root=Path(request["campaign_root"]), chain_receipt=Path(source["amendment_chain_receipt"]),
            expected_chain=source["amendment_chain"]))
    _write(attempt / "unit_result.json", result)
    return result


def run_scientific_unit(request):
    kind = "science_futility" if request["scientific_step"] == "futility" else "science"
    source, _, policy, _, _ = _runtime(request, resource="gpu", kind=kind)
    _require(source.get("scientific_identity_sha256"), "Scientific dispatch requires execution v5.")
    root, attempt = Path(request["execution_root"]), _attempt(request)
    hot_storage.require_restored_attempts(root.resolve())
    require_restored_scientific_outputs(root.resolve())
    complete = root / "runs" / request["unit_id"] / "complete.json"
    with _execution(request, policy) as execution:
        if complete.exists():
            result = _json(complete)
            _verify_scientific_result(request, result, source)
            return result
        terminal = attempt / "unit_result.json"
        if terminal.exists():
            result = _json(terminal)
            _verify_scientific_result(request, result, source)
            pending = execution.state.get("pending_unit", {})
            _require(pending.get("run_name") == request["unit_id"] and pending.get("persisted"),
                     "Scientific output needs independent host acknowledgment before its next step.")
            result.update(attempt=request.get("attempt", 1), persistence_receipt=pending["persistence_receipt"])
            _write(complete, result)
            execution.persist([complete])
            return result
        _require(not attempt.exists(), "An interrupted scientific action needs explicit recovery, never blind replay.")
        request_path = root / "commands" / f"{request['unit_id']}_attempt{request.get('attempt', 1)}.json"
        _write(request_path, dict(request))
        execution.admit(request["unit_id"], request["forecast_seconds"])
        _prepare_scientific_outputs(request, source, request_path)
        started = time.monotonic()
        log_path = request_path.with_suffix(".log")
        failure = None
        with log_path.open("x") as log:
            try:
                result = execution.run_subprocess([request["python"], str(REPO_ROOT / "scripts/run_ec_clean30_phase4.py"),
                    "_scientific-worker", "--request", str(request_path)], cwd=REPO_ROOT,
                    env=phase3.clean_repository_environment(), stdout=log, stderr=subprocess.STDOUT,
                    timeout=request["unit_timeout_seconds"])
            except Exception as exc:
                failure, result = exc, None
            finally:
                _restore_scientific_outputs(attempt)
        passed = result is not None and result.returncode == 0 and terminal.is_file() and _json(terminal).get("status") == "passed"
        execution.record_result(request["unit_id"], "completed" if passed else "failed_fit", time.monotonic() - started)
        persisted = execution.persist([request_path, log_path, *([attempt] if attempt.exists() else [])])
        _require(passed, "Scientific action failed; its outputs are preserved before diagnosis."
                 + (f" {type(failure).__name__}: {failure}" if failure is not None else ""))
        return {"status": "awaiting_host_ack", "unit_id": request["unit_id"], "persistence": persisted}


def run_scientific_step(request, *, step, transport=None, check_session=None, close_session=False):
    _require(step in SCIENTIFIC_STEPS, "Unknown scientific dispatch step.")
    fold = int(request.get("fold", 0))
    unit = {"unit_id": f"ecc30_science_fold{fold}_{step}", "scientific_step": step,
            "forecast_seconds": request["scientific_forecasts"][step],
            "unit_timeout_seconds": request["scientific_timeouts"][step]}
    summary = _run_remote_queue(request, [unit], action="scientific-unit", next_gate=step,
        transport=transport, check_session=check_session, close_session=close_session)
    # The terminal record has already reached both independent host copies.
    result = _json(Path(request["durable_root"]) / "runs" / unit["unit_id"] / "complete.json")
    _require(summary["status"] == "passed", "Scientific step did not complete.")
    return result


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
        git_env = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
        subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(REPO_ROOT), str(baseline)],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=git_env)
        phase3._git(baseline, "sparse-checkout", "set", "--no-cone", "/src/", "/scripts/", "/" + phase3.freeze.CAMPAIGN_DOCS_DIR + "/")
        phase3._git(baseline, "checkout", "--quiet", "--detach", source["amendment_commit"])
        common = ["--fold", "0", "--plan", str(Path(request["plan"]).resolve()), "--components", str(Path(request["components"]).resolve()),
                  "--freeze1-sha256", source["freeze1_sha256"], "--freeze1-amendment", str(baseline / source["amendment_path"])]
        for path in runs:
            common += ["--runs", path]
        env = dict(git_env, PYTHONPATH=os.pathsep.join((str(baseline / "src"), str(baseline))), PYTHONDONTWRITEBYTECODE="1")
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
    parser.add_argument("action", choices=("plan", "plan-v2", "clean-inputs", "clean-final-inputs", "s1-unit", "s1-queue", "refit-unit", "refit-queue",
                                           "clean-unit", "clean-queue", "attach-fold", "_attach-fold-worker", "scientific-unit", "clean-embeddings", "embedding-queue",
                                           "select-gap", "_s1-worker", "_refit-worker", "_clean-worker", "_embedding-worker", "_scientific-worker"))
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        request = _json(args.request)
        if args.action == "plan":
            result = {"scope": SCOPE, "s1": [unit.to_dict() for unit in s1_units()], "clean": list(clean_units())}
        elif args.action == "plan-v2":
            plan = campaign.build_fold_plan_v2(request["fold"])
            result = {"scope": V5_SCOPE, "plan": plan, "s1": [unit.to_dict() for unit in fold_units(plan, "S1")],
                      "refit": [unit.to_dict() for unit in fold_units(plan, "S4")],
                      "clean_final": list(clean_final_units(request["fold"]))}
        elif args.action in ("clean-inputs", "clean-final-inputs"):
            record = (prepare_clean_inputs if args.action == "clean-inputs" else prepare_clean_final_inputs)(request)
            result = {"status": "passed", "n_units": len(record["units"]), "n_sequences": record["n_sequences"]}
        elif args.action in ("s1-unit", "refit-unit", "clean-unit"):
            result = run_unit(request, kind=args.action.split("-")[0])
        elif args.action == "clean-queue":
            result = run_clean_queue(request)
        elif args.action == "s1-queue":
            result = run_s1_queue(request)
        elif args.action == "refit-queue":
            result = run_refit_queue(request)
        elif args.action == "attach-fold":
            result = attach_fold(request)
        elif args.action == "scientific-unit":
            result = run_scientific_unit(request)
        elif args.action == "clean-embeddings":
            result = run_unit(request, kind="embedding")
        elif args.action == "embedding-queue":
            result = run_embedding_queue(request)
        elif args.action == "select-gap":
            result = select_gap(request)
        else:
            result = {"_attach-fold-worker": attachment_worker, "_s1-worker": s1_worker, "_refit-worker": refit_worker, "_clean-worker": clean_worker,
                      "_embedding-worker": embedding_worker, "_scientific-worker": scientific_worker}[args.action](request)
        print(json.dumps({key: result[key] for key in ("status", "unit_id", "persistence", "n_units", "n_sequences", "next_gate", "decision")
                          if key in result} if args.action not in ("plan", "plan-v2") else result, sort_keys=True))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
