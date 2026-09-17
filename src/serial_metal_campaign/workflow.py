"""Explicit, serial campaign transitions. Advancing the queue never trains."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import fcntl
import json
import math
import shutil
import time
import uuid

import run_metal_architecture_pilot as base
from serial_metal_campaign import budget, control, evidence, profile, reporting, runtime


@contextmanager
def controller_lock(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".controller.lock").open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another campaign transition or execution is active") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def current_session(output):
    rows = [r for r in base.read(Path(output) / "sessions.json", []) if r.get("stopped_epoch") is None]
    base.require(len(rows) == 1, "Exactly one registered allocation is required")
    return rows[0]


def state_files(output):
    """Mutable controller state, excluding receipts that attest to this snapshot."""
    output = Path(output)
    names = ("campaign_manifest.json", "queue.json", "fold_plan.json", "input_identity.json", "preparation.json",
             "training_cache_audit.json", "sessions.json", "attempts.json", "confirmation_manifest.json",
             "discovery_closed.json", "refinement_decisions.json", "chain_decisions.json", "comparisons.json",
             "active_process.json", "launch_intents.json", "budget_authorizations.json",
             "control_events.json", "host_accounting_reconciliation.json")
    paths = [output / name for name in names if (output / name).is_file()]
    paths.extend(sorted(output.glob("USER_REQUESTED_PAUSE*.json")))
    for directory in ("reuse", "readiness", "comparisons", "persistence"):
        paths.extend(sorted((output / directory).glob("*.json")))
    return paths


def state_inventory(output):
    output = Path(output)
    return {str(p.relative_to(output)): base.digest(p) for p in state_files(output)}


def export_state(output):
    """Create an immutable state snapshot for independent durable readback."""
    output = Path(output)
    inventory = state_inventory(output)
    ident = base.fingerprint(inventory)
    directory = output / "state_snapshots" / ident
    for source in state_files(output):
        target = directory / source.relative_to(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
        base.require(base.digest(target) == inventory[str(source.relative_to(output))], "State snapshot changed")
    return dict(state_sha256=ident, source_root=str(directory), files=inventory)


def verify_state_transfer(output, receipt):
    expected = export_state(output)
    readback = Path(receipt["readback_root"]).resolve()
    base.require(receipt.get("destination_uri") and readback != Path(expected["source_root"]).resolve(),
                 "State persistence needs a durable destination and independent readback")
    base.require(receipt.get("state_sha256") == expected["state_sha256"], "State receipt is stale")
    for name, digest in expected["files"].items():
        base.require(base.digest(readback / name) == digest, f"Persisted state differs: {name}")
    result = {**receipt, **expected, "verified": True}
    runtime.atomic_json(Path(output) / "state_persistence.json", result)
    return result


def require_persistent_state(output):
    if runtime._is_drive_mount(output):
        # Atomic ledger writes reside on the mounted durable destination.
        return
    receipt = base.read(Path(output) / "state_persistence.json", {})
    base.require(receipt.get("verified") and receipt.get("state_sha256") == base.fingerprint(state_inventory(output)),
                 "Export current campaign state, transfer it, and verify-state-transfer before execution")
    for name, digest in receipt["files"].items():
        base.require(base.digest(Path(receipt["readback_root"]) / name) == digest, "State readback changed")


def collect_results(output, *, verify=False):
    output = Path(output)
    queue = base.read(output / "queue.json")
    by_id = {run["id"]: run for run in queue["runs"]}
    rows = []
    for attempt in base.read(output / "attempts.json", []):
        if attempt["status"] != "completed" or attempt["run_id"] not in by_id:
            continue
        run = by_id[attempt["run_id"]]
        if run.get("kind") == "ring_audit":
            continue
        result = evidence.verify_result(output, run, attempt["run_dir"]) if verify else attempt["result"]
        rows.append({**result, "elapsed_seconds": attempt["elapsed_seconds"], "session_id": attempt["session_id"]})
    for path in sorted((output / "reuse").glob("*.json")):
        receipt = base.read(path)
        if verify:
            evidence.verify_reuse_receipt(output, receipt)
        base.require(receipt.get("run_id") in by_id, "Reuse receipt refers to an unplanned run")
        rows.append(receipt["result"])
    base.require(len({r["id"] for r in rows}) == len(rows), "Multiple accepted results for one run")
    return rows


def _timings(output):
    """Current hardware only; new allocations must supply a fresh timing anchor."""
    session = current_session(output)
    sessions = {s["session_id"]: s for s in base.read(Path(output) / "sessions.json", [])}
    queue = {r["id"]: r for r in base.read(Path(output) / "queue.json")["runs"]}
    rows = []
    for attempt in base.read(Path(output) / "attempts.json", []):
        if attempt["status"] != "completed" or attempt["run_id"] not in queue:
            continue
        if sessions[attempt["session_id"]]["hardware"] != session["hardware"]:
            continue
        run = queue[attempt["run_id"]]
        if run.get("kind"):
            continue
        prepared = Path(attempt["run_dir"]) / "prepare_status.json"
        measured = base.read(Path(attempt["run_dir"]) / "performance_profile.json", {})
        recorded_setup = measured.get("setup_seconds")
        def valid_time(value):
            return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
        if (isinstance(recorded_setup, (int, float)) and not isinstance(recorded_setup, bool)
                and math.isfinite(recorded_setup) and 0 <= recorded_setup <= attempt["elapsed_seconds"]):
            setup = float(recorded_setup)
            recorded_total = measured.get("total_seconds", measured.get("elapsed_seconds"))
            recorded_training = measured.get("training_seconds")
            if (valid_time(recorded_total) and valid_time(recorded_training)
                    and 0 < recorded_total <= attempt["elapsed_seconds"]
                    and math.isclose(recorded_setup + recorded_training, recorded_total, rel_tol=1e-6, abs_tol=.001)):
                # Launch/import and result-verification overhead is paid once per
                # fit. A one-epoch probe must not multiply it by fifty epochs.
                setup += max(0., attempt["elapsed_seconds"] - recorded_total)
        else:
            observed_setup = prepared.stat().st_mtime - attempt["started_epoch"]
            # A copied file's timestamp can be outside the original attempt.
            # With no usable timing record, price all elapsed time per epoch.
            setup = observed_setup if 0 <= observed_setup < attempt["elapsed_seconds"] else 0.
        rows.append({**run, "session_id": attempt["session_id"], "elapsed_seconds": attempt["elapsed_seconds"],
                     "setup_seconds": setup, "epoch_seconds": max(.001, (attempt["elapsed_seconds"]-setup)/run["epochs"])})
    fresh = [r for r in rows if r["session_id"] == session["session_id"]]
    base.require(fresh, "A fresh one-epoch timing anchor is required for this allocation")
    # Same-device later sessions remeasure a reference GVP anchor. Slower observed
    # wall time scales every historical family estimate conservatively.
    anchors = [r for r in fresh if r["arm"] == "gvp_four" and r["epochs"] == 1]
    previous = [r for r in rows if r["session_id"] != session["session_id"] and
                r["arm"] == "gvp_four" and r["epochs"] == 1 and r["parameters"] == anchors[0]["parameters"]] if anchors else []
    scale = max(1., max(r["elapsed_seconds"] for r in anchors) / min(r["elapsed_seconds"] for r in previous)) if previous else 1.
    for row in rows:
        if row["session_id"] != session["session_id"]:
            for key in ("elapsed_seconds", "setup_seconds", "epoch_seconds"):
                row[key] *= scale
    return rows


def readiness(output, operations_plan=None):
    """Verify hardware/inputs and enqueue new-session timing, without training."""
    output = Path(output)
    manifest = profile.verify_manifest(output)
    evidence.verify_preparation(output, hash_contents=True)
    session = current_session(output)
    hardware = evidence.hardware_probe()
    base.require(hardware == session["hardware"], "Observed hardware differs from registered session")
    checked_plan = validate_operations_plan(operations_plan) if operations_plan else None
    queue = base.read(output / "queue.json")
    path = output / "readiness" / f"{session['session_id']}.json"
    existing = base.read(path)
    if existing:
        base.require(existing["hardware"] == hardware, "Readiness hardware changed")
        if operations_plan is not None:
            existing["operations_plan"] = checked_plan
            runtime.atomic_json(path, existing)
        return existing
    previous = [s for s in base.read(output / "sessions.json", []) if s["session_id"] != session["session_id"]]
    if previous:
        same = any(s["hardware"] == hardware for s in previous)
        templates = [r for r in manifest["runs"] if r["stage"] == "operations" and r.get("kind") is None]
        if same:
            templates = [r for r in templates if r["arm"] == "gvp_four" and r["parameters"]["capacity"] == "reference"][:1]
        probes = []
        for run in templates:
            probes.append(profile.make_run(manifest, run["arm"], run["parameters"],
                                  block="smoke_"+base.fingerprint(session["session_id"])[:8],
                                  stage="operations", epochs=1, ring=run["ring"]))
        _append_unique(queue, probes)
        _save_queue(output, queue)
    # Prove local write/read even when independent persistence is used.
    token = uuid.uuid4().hex
    probe = output / ".storage_probe"
    probe.write_text(token)
    base.require(probe.read_text() == token, "Output write/read failed")
    probe.unlink()
    value = dict(session_id=session["session_id"], hardware=hardware, status="inputs_hardware_verified",
                 manifest_sha256=base.digest(output / "campaign_manifest.json"),
                 operations_plan=checked_plan)
    runtime.atomic_json(path, value)
    return value


def validate_operations_plan(plan):
    fields = ("setup_seconds", "smoke_seconds", "audit_seconds", "persistence_seconds", "recovery_seconds", "shutdown_seconds")
    values = {key: budget.nonnegative(plan[key], key) for key in fields}
    sessions = plan.get("sessions_remaining")
    base.require(isinstance(sessions, int) and not isinstance(sessions, bool) and sessions > 0,
                 "Operations forecast needs the remaining session count")
    base.require(values["shutdown_seconds"] >= 900*sessions, "Reserve fifteen minutes for each remaining shutdown")
    base.require(str(plan.get("basis", "")).strip(), "Document the operations timing basis")
    return {**plan, **values, "raw_seconds": sum(values.values())}


def measured_forecast(output, *, candidates=None, extra_runs=()):
    output = Path(output)
    manifest = profile.verify_manifest(output)
    session = current_session(output)
    ready = base.read(output / "readiness" / f"{session['session_id']}.json", {})
    base.require(ready.get("operations_plan"), "Supply a measured future operations plan before full-fit admission")
    frozen = base.read(output / "confirmation_manifest.json")
    if frozen:
        candidates = frozen["candidates"]
    result = budget.forecast(output, manifest, base.read(output / "queue.json"), collect_results(output),
                             _timings(output), ready["operations_plan"]["raw_seconds"],
                             candidates=candidates, extra_runs=extra_runs)
    if frozen:
        selected_cost = sum(1.25 * result["confirmation_raw_seconds"][b] for b in frozen["planned_blocks"])
        available = min(result["budget"]["remaining_seconds"]["confirmation"],
                        result["budget"]["total_remaining_seconds"] - result["operations_buffered_seconds"])
        result["full_training_admitted"] = result["operations_admitted"] and selected_cost <= available
        result["confirmation"].update(planned_blocks=frozen["planned_blocks"], deferred_blocks=frozen["deferred_blocks"],
                                       frozen_coverage_fits=selected_cost <= available)
    return budget.write_preview(output, result)


def _complete(runs, results):
    done = {r["id"] for r in results}
    return all(r["id"] in done for r in runs)


def _save_queue(output, queue):
    ids = [r["id"] for r in queue["runs"]]
    base.require(len(ids) == len(set(ids)), "Duplicate queue run identity")
    runtime.atomic_json(Path(output) / "queue.json", queue)
    return queue


def _append_unique(queue, runs):
    # Reuse an already planned recipe/seed instead of fitting the same cell twice.
    units = {_unit(r) for r in queue["runs"]}
    added = []
    for run in runs:
        key = _unit(run)
        if key not in units:
            queue["runs"].append(run)
            added.append(run)
            units.add(key)
    return added


def _unit(run):
    if run["stage"] == "operations":
        return ("operations", run["id"])  # A new allocation needs a fresh timing anchor.
    return (run["arm"], run["recipe_id"], run["seed"], run["stage"], run.get("fold_index"))


def reconcile_repeat_allowance(output, queue):
    if queue.get("top_two"):
        placeholders = [r for r in base.read(Path(output) / "comparisons.json", [])
                        if r["block_id"] == "future_top_two_repeats" and not r.get("released_reason")]
        if placeholders:
            runtime.release_comparison(output, "future_top_two_repeats", "Replaced by the frozen top-two repeat identities")


def _admit_optional(output, runs, baseline):
    proposed = measured_forecast(output, extra_runs=runs)
    base.require(proposed["confirmation"]["planned_blocks"] == baseline["confirmation"]["planned_blocks"],
                 "Optional work would reduce confirmation coverage")
    base.require(proposed["full_training_admitted"], "Optional work does not fit protected stage budgets")
    return proposed


def _profile_optional(output, manifest, queue, runs):
    """Admit bounded operations probes before pricing a never-measured capacity."""
    measurements = _timings(output)
    session_suffix = base.fingerprint(current_session(output)["session_id"])[:8]
    missing = []
    for run in runs:
        try:
            profile.forecast(run, measurements)
        except ValueError as exc:
            if "Missing compatible" not in str(exc):
                raise
            probe = profile.make_run(manifest, run["arm"], run["parameters"], epochs=1,
                                     stage="operations", block="profile_"+run["recipe_id"]+"_"+session_suffix,
                                     ring=run["ring"])
            if probe["id"] not in {r["id"] for r in missing}:
                missing.append(probe)
    if not missing:
        return []
    planned = {r["id"] for r in queue["runs"]}
    completed = {r["id"] for r in collect_results(output)}
    base.require(not any(r["id"] in completed for r in missing),
                 "Completed capacity profiling did not produce compatible timing")
    fresh = [r for r in missing if r["id"] not in planned]
    if fresh:
        decision = runtime.admission(output, "operations", 600*len(fresh), confirmation_reserve_seconds=operation_confirmation_reserve(output))
        base.require(decision["allowed"], "Cannot fund the required capacity profiling: " + decision["reason"])
        _append_unique(queue, fresh)
        _save_queue(output, queue)
    return [r["id"] for r in missing]


def admit(output):
    """Reserve complete scientific blocks cumulatively, separate from dispatch."""
    output = Path(output)
    queue = base.read(output / "queue.json")
    reconcile_repeat_allowance(output, queue)
    forecast = measured_forecast(output)
    base.require(forecast["full_training_admitted"], "Measured campaign forecast rejects full training")
    results = collect_results(output, verify=True)
    done = {r["id"] for r in results}
    measurements = _timings(output)
    reserved_ids = {ident for row in base.read(output / "comparisons.json", []) for ident in row["run_ids"]}
    stage = "confirmation" if queue["phase"] == "confirmation" else "discovery"
    reserve = max(forecast["budget"]["caps_seconds"]["confirmation"],
                  sum(forecast["confirmation"]["buffered_seconds"][b]
                      for b in forecast["confirmation"]["planned_blocks"])) if stage == "discovery" else 0.
    if not queue.get("top_two") and "future_repeat_0" not in reserved_ids:
        costs = {}
        for index, arm in enumerate(profile.ARMS):
            rows = [r for r in queue["runs"] if r["arm"] == arm and r["block"] == "screen"]
            cost = max(profile.forecast(r, measurements) for r in rows)
            for seed_slot in range(2):
                costs[f"future_repeat_{index*2+seed_slot}"] = cost
        runtime.reserve_comparison(output, "future_top_two_repeats", "discovery", list(costs), costs,
                                   confirmation_reserve_seconds=reserve)
    groups = {}
    for run in queue["runs"]:
        if run["stage"] == stage and run["id"] not in done and run["id"] not in reserved_ids:
            groups.setdefault(run["block"], []).append(run)
    for block, runs in groups.items():
        runtime.reserve_comparison(output, stage+"_"+block, stage, runs,
                                   {r["id"]: profile.forecast(r, measurements) for r in runs},
                                   confirmation_reserve_seconds=reserve)
    return dict(status="reserved", phase=queue["phase"], forecast=forecast,
                comparisons=base.read(output / "comparisons.json", []))


def advance(output):
    """Move one decision boundary. No GPU or training subprocess is started."""
    output = Path(output)
    manifest = profile.verify_manifest(output)
    evidence.verify_preparation(output)
    queue = base.read(output / "queue.json")
    reconcile_repeat_allowance(output, queue)
    results = collect_results(output, verify=True)
    phase = queue["phase"]
    if phase == "confirmation":
        wanted = [r for r in queue["runs"] if r["stage"] == "confirmation"]
        if _complete(wanted, results):
            queue["phase"] = "complete"
            _save_queue(output, queue)
        return dict(phase=queue["phase"], remaining=len(wanted)-sum(r["id"] in {x["id"] for x in results} for r in wanted))
    if phase == "complete":
        return dict(phase=phase)
    if (output / "confirmation_manifest.json").exists():
        base.require(phase == "freeze", "Discovery cannot change after confirmation freeze")
        return finish_confirmation_freeze(output, queue, base.read(output / "confirmation_manifest.json"))
    if phase == "screen":
        mandatory = [r for r in queue["runs"] if r["stage"] == "discovery"]
        if not _complete(mandatory, results):
            return dict(phase=phase, action="readiness, then execute the next admitted mandatory fit")
        top_two = reporting.choose_screen_candidates(queue["runs"], results)
        queue["top_two"] = {arm: [r["id"] for r in chosen] for arm, chosen in top_two.items()}
        repeats = [profile.make_run(manifest, arm, run["parameters"], seed=43, block="repeat")
                   for arm, chosen in top_two.items() for run in chosen]
        _append_unique(queue, repeats)
        queue["phase"] = "repeat"
        _save_queue(output, queue)
        if any(r["block_id"] == "future_top_two_repeats" for r in base.read(output / "comparisons.json", [])):
            runtime.release_comparison(output, "future_top_two_repeats", "Replaced by the frozen top-two repeat identities")
        return dict(phase="repeat", top_two=queue["top_two"], required_repeats=len(repeats))
    if phase == "repeat":
        if not _complete([r for r in queue["runs"] if r["stage"] == "discovery"], results):
            return dict(phase=phase, action="complete both repeated candidates per arm")
        chosen = reporting.freeze_candidates(queue["runs"], results)
        diagnoses = {family: profile.refinement_parameters(family, chosen, results) for family in profile.FAMILY_ORDER}
        profile.freeze(output / "refinement_decisions.json", dict(initial_candidates=chosen, diagnoses=diagnoses))
        queue.update(phase="refinement", optional_cursor=0, optional_decisions=[])
        _save_queue(output, queue)
        return dict(phase="refinement", diagnoses=diagnoses)
    if phase == "refinement":
        if not _complete([r for r in queue["runs"] if r["stage"] == "discovery"], results):
            return dict(phase=phase, action="complete the admitted paired diagnostic block")
        frozen = base.read(output / "refinement_decisions.json")
        order = [(slot, family) for slot in range(2) for family in profile.FAMILY_ORDER]
        cursor = queue["optional_cursor"]
        if cursor < len(order):
            slot, family = order[cursor]
            decision = frozen["diagnoses"][family]
            runs = [profile.make_run(manifest, arm, values[slot], seed=seed, block=f"diagnostic_{slot}_{family}")
                    for arm, values in decision["parameters"].items() if len(values) > slot for seed in profile.SEEDS]
            record = dict(slot=slot, family=family, funded=False)
            if runs:
                try:
                    probes = _profile_optional(output, manifest, queue, runs)
                    if probes:
                        return dict(phase=phase, status="profiling_required", runs=probes)
                    baseline = measured_forecast(output)
                    record["forecast"] = _admit_optional(output, runs, baseline)
                    record["runs"] = [r["id"] for r in _append_unique(queue, runs)]
                    record["funded"] = True
                except ValueError as exc:
                    record["reason"] = str(exc)
            else:
                record["reason"] = "no applicable candidate in this slot"
            queue["optional_decisions"].append(record)
            queue["optional_cursor"] += 1
            _save_queue(output, queue)
            return record
        queue.update(phase="chains", chain_cursor=0, chain_history=[])
        _save_queue(output, queue)
        return dict(phase="chains")
    if phase == "chains":
        return advance_chain(output, manifest, queue, results)
    if phase == "freeze":
        return freeze_confirmation(output, manifest, queue, results)
    raise ValueError(f"Unknown queue phase: {phase}")


def _chain_comparisons(manifest, queue):
    """Only predeclared controls; never an unrestricted post-hoc all-pairs search."""
    comparisons = []
    initial = queue.get("top_two", {})
    by_id = {r["id"]: r for r in queue["runs"]}
    for family in profile.FAMILY_ORDER:
        arms = [a for a in profile.ARMS if a.startswith(family+"_")]
        if all(a in initial for a in arms):
            comparisons.append(dict(family=family, parent_runs={a: by_id[initial[a][1]] for a in arms},
                                    candidate_runs={a: by_id[initial[a][0]] for a in arms}))
        if family == "late":
            parents, candidates = {}, {}
            for arm in arms:
                parents[arm] = next(r for r in queue["runs"] if r["arm"] == arm and r["block"] == "screen"
                                    and r["parameters"]["capacity"] == "reference" and r["lr"] == 3e-5)
                candidates[arm] = next(r for r in queue["runs"] if r["arm"] == arm and r["block"] == "large" and r["seed"] == 42)
            comparisons.append(dict(family=family, parent_runs=parents, candidate_runs=candidates))
    frozen = base.read(Path(manifest["output_dir"]) / "refinement_decisions.json", {})
    for decision in queue.get("optional_decisions", []):
        if not decision["funded"]:
            continue
        candidates = {by_id[ident]["arm"]: by_id[ident] for ident in decision.get("runs", []) if by_id[ident]["seed"] == 42}
        if candidates:
            comparisons.append(dict(family=decision["family"], parent_runs={a: frozen["initial_candidates"][a] for a in candidates},
                                    candidate_runs=candidates))
    comparisons.extend(h["comparison"] for h in queue.get("chain_history", []) if h.get("comparison"))
    return comparisons


def advance_chain(output, manifest, queue, results):
    from serial_metal_campaign import adaptive
    if not _complete([r for r in queue["runs"] if r["stage"] == "discovery"], results):
        return dict(phase="chains", action="complete the admitted matched continuation block")
    if queue["chain_cursor"] >= 3*len(profile.FAMILY_ORDER):
        queue["phase"] = "freeze"
        _save_queue(output, queue)
        return dict(phase="freeze")
    family = profile.FAMILY_ORDER[queue["chain_cursor"] % len(profile.FAMILY_ORDER)]
    used = {arm: sum(r["arm"] == arm and r["block"].startswith("chain_") for r in queue["runs"])
            for arm in profile.ARMS}
    history = [h for h in queue["chain_history"] if h["family"] == family and h.get("funded")]
    previous_steps = [h for h in history if h.get("comparison")]
    comparisons = ([previous_steps[-1]["comparison"]] if previous_steps else
                   [c for c in _chain_comparisons(manifest, queue) if c["family"] == family])
    proposal = adaptive.propose_next(family, comparisons,
                                     results, fits_used_per_arm=used,
                                     active_axis=history[0].get("axis") if history else None,
                                     active_direction=history[0]["proposal"].get("direction") if history else None)
    record = dict(family=family, round=queue["chain_cursor"] // len(profile.FAMILY_ORDER), proposal=proposal, funded=False)
    runs = []
    for control in proposal.get("controls", []):
        runs.extend(profile.make_run(manifest, control["arm"], control["parameters"], seed=seed,
                                     block=f"chain_control_{queue['chain_cursor']}") for seed in control["seeds"])
    for arm, parameters in ({} if proposal.get("controls") else proposal.get("parameters", {})).items():
        runs.extend(profile.make_run(manifest, arm, parameters, seed=seed,
                                     block=f"chain_{queue['chain_cursor']}") for seed in profile.SEEDS)
    expected_runs = list(runs)
    planned_units = {_unit(r) for r in queue["runs"]}
    runs = [r for r in runs if _unit(r) not in planned_units]
    if runs:
        try:
            base.require(all(used[a]+sum(r["arm"] == a for r in runs) <= 6 for a in profile.ARMS), "Continuation allowance exhausted")
            probes = _profile_optional(output, manifest, queue, runs)
            if probes:
                return dict(phase="chains", status="profiling_required", runs=probes)
            record["forecast"] = _admit_optional(output, runs, measured_forecast(output))
            added = _append_unique(queue, runs)
            record.update(funded=True, axis=proposal.get("axis"), runs=[r["id"] for r in added])
            if proposal.get("parameters") and not proposal.get("controls"):
                record["comparison"] = dict(family=family, parent_runs=proposal["parent_runs"],
                                             candidate_runs={r["arm"]: next(q for q in queue["runs"] if _unit(q) == _unit(r))
                                                             for r in expected_runs if r["seed"] == 42})
        except ValueError as exc:
            record["reason"] = str(exc)
    queue["chain_history"].append(record)
    if not (record["funded"] and proposal.get("controls")):
        queue["chain_cursor"] += 1
    _save_queue(output, queue)
    runtime.atomic_json(Path(output) / "chain_decisions.json", queue["chain_history"])
    return record


def freeze_confirmation(output, manifest, queue, results):
    base.require(_complete([r for r in queue["runs"] if r["stage"] == "discovery"], results), "Incomplete discovery")
    chosen = reporting.freeze_candidates(queue["runs"], results)
    forecast = measured_forecast(output, candidates=chosen)
    base.require(forecast["full_training_admitted"], "No affordable complete core confirmation block")
    runs = profile.confirmation_runs(manifest, chosen)
    decision = forecast["confirmation"]
    accepted = [r for r in runs if r["block"] in decision["planned_blocks"]]
    payload = dict(candidates=chosen, runs=accepted, all_runs=runs,
                   planned_blocks=decision["planned_blocks"], deferred_blocks=decision["deferred_blocks"],
                   forecast=forecast, folds_sha256=base.digest(Path(output) / "fold_plan.json"),
                   cohort_sha256=base.read(Path(output) / "preparation.json")["cohort_sha256"],
                   held_out_evaluation=False, evidence_status="exploratory")
    # Closure and final manifest are idempotent, permitting recovery between
    # the two durable writes without reselecting from any fold outcomes.
    profile.freeze(Path(output) / "confirmation_manifest.json", payload)
    return finish_confirmation_freeze(output, queue, payload)


def finish_confirmation_freeze(output, queue, payload):
    """Recover an interrupted freeze using the already frozen identities only."""
    base.require(payload.get("held_out_evaluation") is False and payload.get("runs"), "Invalid frozen confirmation")
    runtime.freeze_discovery(output)
    _append_unique(queue, payload["runs"])
    queue["phase"] = "confirmation"
    _save_queue(output, queue)
    return payload


def next_run(output):
    queue = base.read(Path(output) / "queue.json")
    results = collect_results(output)
    completed = {r["id"] for r in results}
    session = current_session(output)
    attempts = base.read(Path(output) / "attempts.json", [])
    ready = base.read(Path(output) / "readiness" / f"{session['session_id']}.json")
    base.require(ready, "Run readiness on this allocation first")
    confirm = base.read(Path(output) / "confirmation_manifest.json", {})
    attempts_by_run = {}
    for attempt in attempts:
        attempts_by_run.setdefault(attempt["run_id"], []).append(attempt)
    # These operations probes price all possible confirmation blocks. They
    # do not admit their full-fit comparisons or provide selection evidence.
    for run in queue["runs"]:
        if run["stage"] != "operations" or run.get("kind") or run["id"] in completed:
            continue
        # An exhausted optional timing probe must not prevent a later session's
        # mandatory fresh hardware anchor from running. The measured forecast
        # still fails closed when that probe's compatible timing is required.
        if len(attempts_by_run.get(run["id"], [])) >= 2:
            continue
        return run
    stage = "confirmation" if queue["phase"] == "confirmation" else "discovery"
    for run in queue["runs"]:
        if run["stage"] == stage and run["id"] not in completed:
            return run
    return None


def verify_host_guard(output, receipt_path, *, session=None):
    user_control = control.require_running(output)
    receipt = base.read(receipt_path)
    session = current_session(output) if session is None else session
    now = time.time()
    base.require(receipt and receipt.get("session_id") == session["session_id"]
                 and receipt.get("allocation_started_epoch") == session["started_epoch"]
                 and receipt.get("watchdog_verified") is True and receipt.get("endpoint")
                 and receipt.get("campaign_profile") == profile.PROFILE
                 and receipt.get("campaign_manifest_sha256") == base.digest(Path(output) / "campaign_manifest.json"),
                 "A fresh ownership-verified host watchdog receipt is required")
    checked, expiry = receipt.get("checked_epoch", 0), receipt.get("expires_epoch", 0)
    limits = runtime.budget_limits(output)
    base.require(checked <= now <= expiry <= checked+120 and now < receipt["training_stop_epoch"],
                 "Host watchdog receipt is expired or the training window has closed")
    base.require(receipt.get("budget_authorization_sha256") == limits["authorization_sha256"]
                 and receipt.get("total_cap_seconds") == limits["total_seconds"],
                 "Host receipt does not bind the active budget authorization")
    if user_control["has_explicit_control"]:
        base.require(receipt.get("campaign_control_sha256") == user_control["control_sha256"],
                     "Host receipt does not bind the current user pause/resume generation")
    base.require(receipt["hard_stop_epoch"] <= session["started_epoch"]+limits["session_seconds"] and
                 receipt["training_stop_epoch"] <= receipt["hard_stop_epoch"]-limits["closeout_seconds"],
                 "Host receipt weakens session/closeout limits")
    closed_seconds = sum(s["stopped_epoch"]-s["started_epoch"] for s in base.read(Path(output) / "sessions.json", [])
                         if s.get("stopped_epoch") is not None)
    base.require(receipt["hard_stop_epoch"] <= session["started_epoch"]+limits["total_seconds"]-closed_seconds,
                 "Host receipt weakens the cumulative allocation cap")
    return receipt


def operation_confirmation_reserve(output):
    status = runtime.budget_status(output)
    if status["discovery_closed"]:
        return 0.0  # Frozen complete confirmation comparisons have their own reservations.
    return max(0., status["caps_seconds"]["confirmation"]
               - status["used_seconds"]["confirmation"]-status["reserved_seconds"]["confirmation"])


def execute(output, host_receipt):
    output = Path(output)
    manifest = profile.verify_manifest(output)
    evidence.verify_preparation(output)
    guard = verify_host_guard(output, host_receipt)
    runtime.verify_persistence(output, full=False)
    run = next_run(output)
    if run is None:
        return dict(status="advance_required", phase=base.read(output / "queue.json")["phase"])
    if run["stage"] == "operations":
        # Bootstrap bound, not a fictitious measured full-fit duration. Smokes
        # have one fixed epoch; timeout means a failed operations attempt.
        forecast_seconds, reserve, block_id = 600., operation_confirmation_reserve(output), None
    else:
        measured = measured_forecast(output)
        base.require(measured["full_training_admitted"], "Measured cost/operations forecast does not admit full training")
        forecast_seconds = profile.forecast(run, _timings(output))
        reserve = max(measured["budget"]["caps_seconds"]["confirmation"],
                      sum(measured["confirmation"]["buffered_seconds"][b]
                          for b in measured["confirmation"]["planned_blocks"])) if run["stage"] == "discovery" else 0.
        reservations = [r for r in base.read(output / "comparisons.json", []) if run["id"] in r["run_ids"] and not r.get("released_reason")]
        base.require(len(reservations) == 1, "Run admit to reserve the full logical comparison before execution")
        block_id = reservations[0]["block_id"]
    if run["stage"] == "confirmation" and run["block"] == "ring":
        directory = ring_audit_directory(output, run["fold_index"])
        base.require(directory, "Run the bounded ring-audit command for this fold before RING fits")
        evidence.ring_input_audit(output, run, directory=directory)
    intent = None if runtime._is_drive_mount(output) else runtime.prepare_launch_intent(output, run)
    require_persistent_state(output)
    result = runtime.execute_attempt(output, run, manifest["root"],
                 lambda row, directory: evidence.verify_result(output, row, directory),
                 confirmation_reserve_seconds=reserve, block_forecast_seconds=forecast_seconds, block_id=block_id,
                 launch_guard=lambda: verify_host_guard(output, host_receipt),
                 training_deadline_epoch=guard["training_stop_epoch"],
                 launch_intent_id=intent["intent_id"] if intent else None)
    if result.get("artifacts") and runtime._is_drive_mount(result["run_dir"]):
        runtime.record_persistence(output, result["attempt_id"], dict(method="mounted_drive", readback_root=result["run_dir"],
                                                                   artifacts=result["artifacts"]))
    return result


def ring_audit_directory(output, fold):
    for attempt in reversed(base.read(Path(output) / "attempts.json", [])):
        if attempt["run_id"] == f"ring_audit_{fold}" and attempt["status"] == "completed":
            return Path(attempt["run_dir"])
    return None


def prepare_ring_audit(output, fold):
    """Prepare the audit identity before persisting its launch intent."""
    output = Path(output)
    manifest = profile.verify_manifest(output)
    evidence.verify_preparation(output)
    confirmation = base.read(output / "confirmation_manifest.json", {})
    base.require("ring" in confirmation.get("planned_blocks", []), "RING confirmation is not admitted")
    source = next(r for r in confirmation["runs"] if r["arm"] == "gvp_ring_off" and r["fold_index"] == fold)
    queue = base.read(output / "queue.json")
    ident = f"ring_audit_{fold}"
    run = dict(id=ident, arm="gvp_ring_off", family=profile.FAMILIES["gvp"], kind="ring_audit",
               stage="operations", block="ring_audit", epochs=0, fold_index=fold,
               command=[source["command"][0], str(Path(manifest["root"]) / "src/run_metal_single_gpu_campaign.py"),
                        "_ring-audit", "--output-dir", str(output), "--source-run-id", source["id"],
                        "--run-name", f"fold_{fold}"], env={},
               run_dir=str(output / "ring_audits" / f"fold_{fold}"))
    if not any(r["id"] == ident for r in queue["runs"]):
        queue["runs"].append(run)
        _save_queue(output, queue)
    return run, source


def ring_audit(output, fold, host_receipt):
    """One bounded CPU audit inside the allocated-operations ledger."""
    output = Path(output)
    guard = verify_host_guard(output, host_receipt)
    run, source = prepare_ring_audit(output, fold)
    existing = ring_audit_directory(output, fold)
    if existing:
        return evidence.ring_input_audit(output, source, directory=existing)
    intent = None if runtime._is_drive_mount(output) else runtime.prepare_launch_intent(output, run)
    require_persistent_state(output)
    def verify(_row, directory):
        result = evidence.ring_input_audit(output, source, directory=directory)
        base.require(result.get("status") == "passed", "RING audit did not pass")
        return dict(kind="ring_audit", fold_index=fold, receipt=result)
    attempt = runtime.execute_attempt(output, run, base.read(output / "campaign_manifest.json")["root"], verify,
                                      block_forecast_seconds=600, confirmation_reserve_seconds=0,
                                      launch_guard=lambda: verify_host_guard(output, host_receipt),
                                      training_deadline_epoch=guard["training_stop_epoch"],
                                      launch_intent_id=intent["intent_id"] if intent else None)
    if attempt.get("artifacts") and runtime._is_drive_mount(attempt["run_dir"]):
        runtime.record_persistence(output, attempt["attempt_id"], dict(method="mounted_drive", readback_root=attempt["run_dir"],
                                                                      artifacts=attempt["artifacts"]))
    return attempt


def report(output):
    output = Path(output)
    manifest = profile.verify_manifest(output)
    queue = base.read(output / "queue.json")
    results = collect_results(output, verify=True)
    confirmation = base.read(output / "confirmation_manifest.json")
    if confirmation:
        certificates = {}
        for fold in range(5):
            directory = ring_audit_directory(output, fold)
            receipt = base.read(directory / "ring_input_audit.json") if directory else None
            if receipt:
                certificates[str(fold)] = dict(status=receipt.get("status"), cohort_sha256=receipt.get("expected_cohort_sha256"))
        confirmation = {**confirmation, "ring_input_audit": dict(verified=len(certificates) == 5, folds=certificates)}
    confirmation_ids = {r["id"] for r in (confirmation or {}).get("runs", [])}
    run_manifest = {**manifest, "runs": [r for r in queue["runs"] if not r.get("kind")
                                       and r["id"] not in confirmation_ids]}
    return reporting.write_report(output, run_manifest, results, confirmation)
