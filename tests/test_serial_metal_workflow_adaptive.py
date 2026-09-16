"""Scientific scheduler recovery and bounded-chain integration regressions."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serial_metal_campaign import adaptive, evidence, profile, reporting, runtime, workflow
from test_serial_metal_workflow import campaign, scored
from test_serial_metal_profile import manifest


def chain_queue(value):
    return dict(runs=deepcopy(value["runs"]), phase="chains", chain_cursor=0,
                chain_history=[], optional_decisions=[])


def test_latest_failed_chain_step_cannot_fall_back_to_older_positive_trend(campaign, monkeypatch):
    output, value = campaign
    queue = chain_queue(value)
    parents, middle, newest = {}, {}, {}
    for arm in ("gvp_four", "gvp_six"):
        parents[arm] = profile.make_run(value, arm, profile.reference_parameters(learning_rate=1e-4))
        middle[arm] = profile.make_run(value, arm, profile.reference_parameters(learning_rate=3e-4), block="chain_0")
        newest[arm] = profile.make_run(value, arm, profile.reference_parameters(learning_rate=6e-4), block="chain_5")
    old = dict(family="gvp", parent_runs=parents, candidate_runs=middle)
    latest = dict(family="gvp", parent_runs=middle, candidate_runs=newest)
    queue["chain_history"] = [dict(family="gvp", funded=True, axis="learning_rate",
                                   proposal=dict(direction=1), comparison=latest)]
    monkeypatch.setattr(workflow, "_chain_comparisons", lambda *_: [old])
    # Remove unrelated duplicate effective recipes from evidence while retaining
    # declared completion; this test's queue is exactly the compared recipes.
    queue["runs"] = [{**recipe, "seed": seed, "id": recipe["id"] + f"_e{seed}"}
                     for mapping in (parents, middle, newest) for recipe in mapping.values() for seed in profile.SEEDS]
    relevant = [scored(row, .705 if "chain_5" in row["id"] else .71 if "chain_0" in row["id"] else .70)
                for row in queue["runs"]]
    monkeypatch.setattr(workflow, "_profile_optional", lambda *_: pytest.fail("A failed latest trend cannot be profiled"))
    result = workflow.advance_chain(output, value, queue, relevant)
    assert result["proposal"]["status"] == "stop" and not result["funded"]


def test_scheduler_executes_missing_controls_without_parallel_candidate_admission(campaign, monkeypatch):
    output, value = campaign
    queue = chain_queue(value)
    complete = [scored(row) for row in queue["runs"] if row["stage"] == "discovery"]
    parameters = profile.reference_parameters(learning_rate=3e-4)
    proposal = dict(status="controls_required", axis="learning_rate", direction=1,
                    controls=[dict(arm=arm, parameters=profile.reference_parameters(), seeds=[43])
                              for arm in ("gvp_four", "gvp_six")],
                    parameters={arm: parameters for arm in ("gvp_four", "gvp_six")})
    monkeypatch.setattr(adaptive, "propose_next", lambda *a, **k: proposal)
    monkeypatch.setattr(workflow, "_profile_optional", lambda *_: [])
    monkeypatch.setattr(workflow, "measured_forecast", lambda *_: {})
    monkeypatch.setattr(workflow, "_admit_optional", lambda *_: {})
    result = workflow.advance_chain(output, value, queue, complete)
    added = [row for row in queue["runs"] if row["block"].startswith("chain_")]
    assert result["funded"] and len(added) == 2
    assert all(row["block"] == "chain_control_0" and row["seed"] == 43 for row in added)
    assert queue["chain_cursor"] == 0  # Finish controls, then re-evaluate this family's gate.
    assert "comparison" not in result


def test_scheduler_counts_controls_toward_each_arms_six_fit_limit(campaign, monkeypatch):
    output, value = campaign
    queue = chain_queue(value)
    for arm in ("gvp_four", "gvp_six"):
        for index in range(6):
            row = profile.make_run(value, arm, profile.reference_parameters(learning_rate=2e-5 + index*1e-6),
                                   block=f"chain_control_{index}")
            queue["runs"].append(row)
    complete = [scored(row) for row in queue["runs"] if row["stage"] == "discovery"]
    def proposal(*args, **kwargs):
        assert kwargs["fits_used_per_arm"]["gvp_four"] == 6
        assert kwargs["fits_used_per_arm"]["gvp_six"] == 6
        return dict(status="stop", reason="six-fit allowance exhausted")
    monkeypatch.setattr(adaptive, "propose_next", proposal)
    result = workflow.advance_chain(output, value, queue, complete)
    assert not result["funded"]


def test_invalid_operations_plan_then_retry_cannot_duplicate_new_session_probes(campaign, monkeypatch):
    output, value = campaign
    hardware = {"gpu": "synthetic"}
    runtime.atomic_json(output / "sessions.json", [dict(session_id="old", started_epoch=1, stopped_epoch=2, hardware=hardware),
                                                  dict(session_id="new", started_epoch=3, stopped_epoch=None, hardware=hardware)])
    monkeypatch.setattr(evidence, "hardware_probe", lambda: hardware)
    plan = dict(setup_seconds=1, smoke_seconds=1, audit_seconds=1, persistence_seconds=1,
                recovery_seconds=1, shutdown_seconds=0, sessions_remaining=1, basis="synthetic measured setup")
    with pytest.raises(ValueError, match="shutdown"):
        workflow.readiness(output, plan)
    workflow.readiness(output, {**plan, "shutdown_seconds": 900})
    queue = runtime.read_json(output / "queue.json")
    assert len(queue["runs"]) == len({row["id"] for row in queue["runs"]})
    assert len([row for row in queue["runs"] if row["block"].startswith("smoke_")]) == 1


def test_completed_ring_audit_is_not_dispatched_through_training_executor(campaign, monkeypatch):
    output, value = campaign
    runtime.atomic_json(output / "sessions.json", [dict(session_id="live", stopped_epoch=None)])
    runtime.atomic_json(output / "readiness/live.json", {"status": "ready"})
    wanted = profile.make_run(value, "gvp_four", profile.reference_parameters(), stage="confirmation", block="core", fold_index=0)
    audit = dict(id="ring_audit_0", stage="operations", kind="ring_audit", block="ring_audit")
    queue = dict(runs=[*value["runs"], wanted, audit], phase="confirmation")
    runtime.atomic_json(output / "queue.json", queue)
    runtime.atomic_json(output / "attempts.json", [dict(run_id="ring_audit_0", status="completed")])
    monkeypatch.setattr(workflow, "collect_results", lambda *_: [{"id": row["id"]} for row in value["runs"]])
    assert workflow.next_run(output)["id"] == wanted["id"]


def test_exhausted_operations_probe_does_not_block_fresh_session_anchor(campaign, monkeypatch):
    output, value = campaign
    runtime.atomic_json(output / "sessions.json", [dict(session_id="live", stopped_epoch=None)])
    runtime.atomic_json(output / "readiness/live.json", {"status": "ready"})
    exhausted = next(
        row for row in value["runs"]
        if row["stage"] == "operations" and row["arm"] == "gvp_ring_on"
    )
    anchor = profile.make_run(
        value,
        "gvp_four",
        profile.reference_parameters(),
        stage="operations",
        block="smoke_fresh_session",
        epochs=1,
    )
    runtime.atomic_json(output / "queue.json", {"runs": [exhausted, anchor], "phase": "screen"})
    runtime.atomic_json(output / "attempts.json", [
        {"run_id": exhausted["id"], "status": "failed"},
        {"run_id": exhausted["id"], "status": "interrupted"},
    ])
    monkeypatch.setattr(workflow, "collect_results", lambda *_: [])
    assert workflow.next_run(output)["id"] == anchor["id"]


def test_interrupted_confirmation_freeze_recovers_frozen_candidates_without_reselection(campaign, monkeypatch):
    output, value = campaign
    queue = dict(runs=deepcopy(value["runs"]), phase="freeze")
    runtime.atomic_json(output / "queue.json", queue)
    candidates = {arm: next(row for row in value["runs"] if row["arm"] == arm and row["block"] == "screen")
                  for arm in profile.ARMS}
    all_runs = profile.confirmation_runs(value, candidates)
    accepted = [row for row in all_runs if row["block"] == "core"]
    payload = dict(candidates=candidates, runs=accepted, all_runs=all_runs, planned_blocks=["core"],
                   deferred_blocks=["five", "fusion", "ring"], forecast={"frozen": True},
                   folds_sha256="frozen-folds", cohort_sha256="frozen-cohort", held_out_evaluation=False,
                   evidence_status="exploratory")
    runtime.atomic_json(output / "confirmation_manifest.json", payload)
    monkeypatch.setattr(workflow, "collect_results", lambda *a, **k: [scored(r) for r in value["runs"] if r["stage"] == "discovery"])
    monkeypatch.setattr(reporting, "freeze_candidates", lambda *a, **k: pytest.fail("Frozen candidates must not be reselected"))
    monkeypatch.setattr(workflow, "measured_forecast", lambda *a, **k: pytest.fail("Frozen coverage must not be replanned"))
    closed = []
    monkeypatch.setattr(runtime, "freeze_discovery", lambda *_: closed.append(True))
    workflow.advance(output)
    recovered = runtime.read_json(output / "queue.json")
    assert recovered["phase"] == "confirmation" and closed
    assert {r["id"] for r in recovered["runs"] if r["stage"] == "confirmation"} == {r["id"] for r in accepted}
    assert runtime.read_json(output / "confirmation_manifest.json") == payload


def test_interrupted_top_two_transition_releases_placeholder_after_recovery(campaign, monkeypatch):
    output, value = campaign
    runtime.atomic_json(output / "comparisons.json", [dict(block_id="future_top_two_repeats", run_ids=["future_repeat_0"],
                                                          stage="discovery", forecasts={"future_repeat_0": 10})])
    complete = [scored(row) for row in value["runs"] if row["stage"] == "discovery"]
    monkeypatch.setattr(workflow, "collect_results", lambda *a, **k: complete)
    release = runtime.release_comparison
    def crash(*args, **kwargs):
        raise RuntimeError("simulated loss after queue commit")
    monkeypatch.setattr(runtime, "release_comparison", crash)
    with pytest.raises(RuntimeError, match="simulated"):
        workflow.advance(output)
    assert runtime.read_json(output / "queue.json")["phase"] == "repeat"
    monkeypatch.setattr(runtime, "release_comparison", release)
    workflow.advance(output)
    assert runtime.read_json(output / "comparisons.json")[0].get("released_reason")


def test_repeated_advance_waits_for_existing_capacity_probe_without_consuming_decision(campaign, monkeypatch):
    output, value = campaign
    queue = chain_queue(value)
    complete = [scored(row) for row in queue["runs"] if row["stage"] == "discovery"]
    parents = {arm: next(row for row in queue["runs"] if row["arm"] == arm and row["block"] == "screen")
               for arm in ("gvp_four", "gvp_six")}
    proposal = dict(status="continue", axis="learning_rate", direction=1, controls=[],
                    parent_runs=parents, parameters={arm: profile.reference_parameters(learning_rate=3e-4)
                                                   for arm in parents})
    session = {"session_id": "allocation-one"}
    monkeypatch.setattr(adaptive, "propose_next", lambda *a, **k: proposal)
    monkeypatch.setattr(workflow, "_timings", lambda *_: [])
    monkeypatch.setattr(workflow, "current_session", lambda *_: session)
    monkeypatch.setattr(workflow, "collect_results", lambda *_: complete)
    monkeypatch.setattr(workflow, "operation_confirmation_reserve", lambda *_: 36000)
    monkeypatch.setattr(runtime, "admission", lambda *a, **k: {"allowed": True, "reason": "within budget"})
    first = workflow.advance_chain(output, value, queue, complete)
    second = workflow.advance_chain(output, value, queue, complete)
    assert first["status"] == second["status"] == "profiling_required"
    assert first["runs"] == second["runs"]
    assert queue["chain_cursor"] == 0 and not queue["chain_history"]
    assert len(queue["runs"]) == len({row["id"] for row in queue["runs"]})
    # Finished probes on another allocation cannot masquerade as current timing.
    complete.extend(scored(row) for row in queue["runs"] if row["id"] in first["runs"])
    session["session_id"] = "allocation-two"
    third = workflow.advance_chain(output, value, queue, complete)
    assert third["status"] == "profiling_required"
    assert not set(third["runs"]) & set(first["runs"])
    assert queue["chain_cursor"] == 0 and not queue["chain_history"]


def test_opposite_target_probe_keeps_reused_target_in_next_frozen_comparison(campaign, monkeypatch):
    output, value = campaign
    queue = chain_queue(value)
    queue["chain_cursor"] = profile.FAMILY_ORDER.index("late")
    parents, children, complete = {}, {}, []
    for arm, parent_lr, child_lr in (("late_four", 1e-5, 5e-6), ("late_six", 1e-4, 2e-4)):
        parents[arm] = profile.make_run(value, arm, profile.reference_parameters(learning_rate=parent_lr))
        children[arm] = profile.make_run(value, arm, profile.reference_parameters(learning_rate=child_lr), block="diagnostic_0_late")
        for recipe, score in ((parents[arm], .70), (children[arm], .71)):
            for seed in profile.SEEDS:
                run = profile.make_run(value, arm, recipe["parameters"], seed=seed, block=recipe["block"])
                complete.append(scored(run, score))
    queue["runs"] = [dict(row) for row in complete]
    comparison = dict(family="late", parent_runs=parents, candidate_runs=children)
    monkeypatch.setattr(workflow, "_chain_comparisons", lambda *_: [comparison])
    monkeypatch.setattr(workflow, "_profile_optional", lambda *_: [])
    monkeypatch.setattr(workflow, "measured_forecast", lambda *_: {})
    monkeypatch.setattr(workflow, "_admit_optional", lambda *_: {})
    record = workflow.advance_chain(output, value, queue, complete)
    assert record["funded"] and len(record["runs"]) == 2
    assert set(record["comparison"]["candidate_runs"]) == {"late_four", "late_six"}
    assert record["comparison"]["candidate_runs"]["late_six"]["id"] == parents["late_six"]["id"]
    assert record["proposal"]["required_fits_per_arm"] == {"late_four": 2, "late_six": 0}
