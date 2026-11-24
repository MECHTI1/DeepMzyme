"""Synthetic proofs of the fold-0 futility boundary; no provider, network, GPU or campaign data is used.

The pipeline, the coordinator's dispatch, the session closeout order, the resume gate and the launch-window
ledger run for real. The pod is a small local model: it writes the futility record once into both preserved
copies and afterwards only returns it, as a finished unit is recovered on the real pod.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from ec_clean30 import campaign, execution_v5 as v5, phase3, phase4, workstation
from ec_clean30_fold0_fixtures import IDENTITY, SOURCE, approved, build, cached_driver, sha, write  # noqa: F401

UNIT = v5.FUTILITY_UNIT
BEFORE = ["clean_embeddings", "clean_inner", "held_out_sanity", "verify_sanity", "s1", "select_s1", "select_s2",
          "futility"]
# Nothing below may ever be dispatched by a fold-0 session: S4 selection, the refits, final CLEAN, a later fold.
FORBIDDEN = {"select_s4", "s4", "final_embeddings", "clean_final", "attach_fold"}
STATES = {"continue": ("futility_boundary", "stopped_at_futility_boundary"),
          "stop_futility": ("stop_futility", "stopped_futility")}
DECISIONS = pytest.mark.parametrize("decision", ["continue", "stop_futility"])


class Pod:
    """The remote side as the pipeline sees it; every dispatch is recorded and finished work is never repeated."""

    def __init__(self, tmp_path, decision, events=None):
        self.roots = [tmp_path / "phase4-primary", tmp_path / "phase4-mirror"]
        for root in self.roots:
            root.mkdir()
        self.decision, self.events, self.computed, self.fault = decision, [] if events is None else events, 0, None

    def install(self, monkeypatch):
        for name, function in (("clean_embeddings", "run_embedding_queue"), ("clean_inner", "run_clean_remote_queue"),
                               ("s1", "run_s1_queue"), ("s4", "run_refit_queue"), ("attach_fold", "run_fold_attachment")):
            monkeypatch.setattr(phase4, function, self.queue(name), raising=False)
        monkeypatch.setattr(phase4, "run_scientific_step", self.step, raising=False)
        monkeypatch.setattr(phase3, "verify_execution_source", lambda *args, **kwargs: SOURCE)
        return self

    def queue(self, name):
        def call(request, **options):
            final = request.get("clean_scope") == "final"
            self.events.append({"clean_embeddings": "final_embeddings", "clean_inner": "clean_final"}[name]
                               if final else name)
            return {"status": "passed"}
        return call

    def unit(self, root):
        return root / "runs" / UNIT

    def step(self, request, *, step, **options):
        self.events.append(step)
        if step != "futility":
            return {"status": "passed", **IDENTITY}
        complete = self.unit(self.roots[0]) / "complete.json"
        if complete.exists():  # A finished unit is returned from its preserved record; nothing is computed again.
            return json.loads(complete.read_text())
        if (self.unit(self.roots[0]) / "attempt1").exists():
            raise RuntimeError("An interrupted scientific action needs explicit recovery, never blind replay.")
        self.computed += 1
        for root in self.roots:
            (self.unit(root) / "attempt1").mkdir(parents=True)
        if self.fault == "interrupted":
            raise RuntimeError("The pod was lost during the futility step.")
        snapshots = []
        for index, (suffix, text) in enumerate(((".jsonl", "step chain\n"),
                                               (".json", json.dumps({"decision": self.decision}) + "\n"),
                                               (".json", json.dumps({"pointer": 2}) + "\n"))):
            name = f"output{index}{suffix}"
            for root in self.roots:
                write(self.unit(root) / "attempt1" / name, text)
            snapshots.append({"path": "/workspace/campaign/" + name, "snapshot": name,
                              "sha256": sha(self.unit(self.roots[0]) / "attempt1" / name)})
        result = {"status": "passed", "unit_id": UNIT, "kind": "scientific", "scientific_step": "futility",
                  "snapshots": snapshots, **IDENTITY, "decision": self.decision,
                  "record_sha256": snapshots[1]["sha256"], "pointer_sequence": 2, "attempt": 1,
                  "persistence_receipt": "transfers/1.json"}
        for root in self.roots:
            write(self.unit(root) / "complete.json", result)
        if callable(self.fault):
            self.fault(self)
        if self.fault == "coordinator_lost":
            raise RuntimeError("The coordinator was lost after the record reached both copies.")
        return json.loads(complete.read_text())

    def request(self, **changes):
        return {"execution_manifest": "manifest", "amendment_chain_receipt": "receipt",
                "expected_chain": SOURCE["amendment_chain"], **IDENTITY, "provider": "runpod", "fold": 0,
                "remote_campaign": "/workspace/campaign", "run_final_clean": False,
                "durable_root": str(self.roots[0]), "mirror_root": str(self.roots[1]),
                "remote_request_template": {"session_id": "owned", "clean_inputs": "inner",
                                            "execution_root": "/workspace/campaign/runs/fold0/phase4"}, **changes}

    def run(self, **changes):
        return v5.run_phase4_pipeline(self.request(**changes), transport=object(), check_session=lambda: None,
                                      record_progress=lambda name, result: None)


def relink(path, target):
    path.unlink()
    path.symlink_to(target)


@pytest.fixture
def pod(tmp_path, monkeypatch):
    return lambda decision, events=None: Pod(tmp_path, decision, events).install(monkeypatch)


# 1. The pipeline: one synchronous boundary for either decision.

@DECISIONS
def test_fold0_ends_at_its_futility_record_for_either_decision(pod, decision):
    model = pod(decision)
    result = model.run()
    assert model.events == BEFORE and not FORBIDDEN.intersection(model.events)
    assert result["status"] == STATES[decision][0]
    boundary = result["futility"]
    record = model.unit(model.roots[0]) / "attempt1" / "output1.json"
    assert boundary == {"unit_id": UNIT, "decision": decision, "record_sha256": sha(record),
                        "pointer_sequence": 2, "attempt": 1, "preserved_copies": 2,
                        "complete_sha256": sha(model.unit(model.roots[1]) / "complete.json")}
    # The decision is the scientific step's own; the boundary only reports it.
    assert json.loads(record.read_text())["decision"] == decision
    assert phase3.source_identity(SOURCE).items() <= result.items()


@DECISIONS
@pytest.mark.parametrize("damage,match", [
    (lambda pod: (pod.unit(pod.roots[1]) / "complete.json").unlink(), "lacks the terminal futility record"),
    (lambda pod: write(pod.unit(pod.roots[1]) / "complete.json", {"status": "passed"}), "lacks the terminal"),
    (lambda pod: (pod.unit(pod.roots[1]) / "attempt1" / "output1.json").unlink(), "lacks a futility output"),
    (lambda pod: write(pod.unit(pod.roots[1]) / "attempt1" / "output1.json", "changed\n"), "lacks a futility output"),
    (lambda pod: write(pod.unit(pod.roots[0]) / "attempt1" / "output2.json", "changed\n"), "lacks a futility output"),
    # The same record in other bytes: both copies parse to the result, and only their digests differ.
    (lambda pod: (pod.unit(pod.roots[1]) / "complete.json").write_text(json.dumps(json.loads(
        (pod.unit(pod.roots[0]) / "complete.json").read_text()), indent=4)), "two preserved futility records differ"),
    # A link into the other copy is not a second copy.
    (lambda pod: relink(pod.unit(pod.roots[1]) / "complete.json", pod.unit(pod.roots[0]) / "complete.json"),
     "lacks the terminal futility record"),
    (lambda pod: relink(pod.unit(pod.roots[1]) / "attempt1" / "output1.json",
                        pod.unit(pod.roots[0]) / "attempt1" / "output1.json"), "lacks a futility output"),
])
def test_the_boundary_needs_the_record_in_both_preserved_copies(pod, decision, damage, match):
    model = pod(decision)
    model.fault = damage
    with pytest.raises(v5.LaunchWindowError, match=match):
        model.run()
    assert model.events == BEFORE and not FORBIDDEN.intersection(model.events)


@pytest.mark.parametrize("change,match", [
    (lambda result: result.update(record_sha256="0" * 64), "does not bind its preserved record"),
    (lambda result: result.update(decision="proceed"), "no binding decision"),
    (lambda result: result.update(unit_id="ecc30_science_fold0_select_s2"), "no binding decision"),
    (lambda result: result["snapshots"][1].update(snapshot="../output1.json"), "does not bind"),
    (lambda result: result.update(attempt="1"), "does not bind"),
])
def test_the_boundary_refuses_a_result_that_is_not_the_preserved_futility_record(pod, change, match):
    model = pod("continue")
    model.run()
    result = json.loads((model.unit(model.roots[0]) / "complete.json").read_text())
    change(result)
    with pytest.raises(v5.LaunchWindowError, match=match):
        v5.verify_futility_boundary(model.request(), result)
    for roots in ({"mirror_root": None}, {"mirror_root": str(model.roots[0])}):
        with pytest.raises(v5.LaunchWindowError, match="both independent preserved copies"):
            v5.verify_futility_boundary(model.request(**roots),
                                        json.loads((model.unit(model.roots[0]) / "complete.json").read_text()))


@DECISIONS
def test_a_repeated_entry_returns_at_the_same_record_and_never_computes_it_again(pod, decision):
    model = pod(decision)
    first = model.run()
    for _ in range(3):  # The identical re-run of an open window re-enters the pipeline from the top.
        assert model.run() == first
    assert model.computed == 1 and model.events == BEFORE * 4 and not FORBIDDEN.intersection(model.events)


@DECISIONS
def test_an_interrupted_futility_step_is_never_followed_by_s4_and_never_run_again(pod, decision):
    model = pod(decision)
    model.fault = "interrupted"
    with pytest.raises(RuntimeError, match="lost during the futility step"):
        model.run()
    model.fault = None
    for _ in range(2):
        with pytest.raises(RuntimeError, match="explicit recovery, never blind replay"):
            model.run()
    assert model.computed == 1 and not FORBIDDEN.intersection(model.events)
    assert not (model.unit(model.roots[0]) / "complete.json").exists()


@DECISIONS
def test_a_record_preserved_before_the_coordinator_was_lost_still_ends_the_session(pod, decision):
    model = pod(decision)
    model.fault = "coordinator_lost"
    with pytest.raises(RuntimeError, match="coordinator was lost"):
        model.run()
    model.fault = None
    assert model.run()["futility"]["decision"] == decision
    assert model.computed == 1 and not FORBIDDEN.intersection(model.events)


def test_fold0_refuses_a_final_clean_request_before_any_dispatch(pod):
    model = pod("continue")
    with pytest.raises(v5.LaunchWindowError, match="Fold 0 ends at its futility record"):
        model.run(run_final_clean=True, final_clean_request={"clean_scope": "final"})
    assert model.events == []


def test_the_boundary_names_the_unit_that_the_real_dispatch_runs(monkeypatch):
    seen = []

    def queue(request, units, **options):
        seen.extend(unit["unit_id"] for unit in units)
        raise LookupError("observed")

    monkeypatch.setattr(phase4, "_run_remote_queue", queue)
    with pytest.raises(LookupError):
        phase4.run_scientific_step({"fold": 0, "scientific_forecasts": {"futility": 1},
                                    "scientific_timeouts": {"futility": 1}}, step="futility")
    assert seen == [v5.FUTILITY_UNIT]


@pytest.mark.parametrize("change,error,match", [
    (lambda result: result.update(decision="proceed"), v5.LaunchWindowError, "no binding decision"),
    (lambda result: result.pop("decision"), v5.LaunchWindowError, "no binding decision"),
    (lambda result: result.update(status="awaiting_host_ack"), v5.LaunchWindowError, "unverified futility"),
    (lambda result: result.update(scientific_identity_sha256="0" * 64), phase3.Phase3Error, "another scientific identity"),
])
def test_a_futility_result_that_is_not_binding_stops_the_session_before_s4(pod, monkeypatch, change, error, match):
    model = pod("continue")
    real = model.step

    def step(request, *, step, **options):
        result = real(request, step=step, **options)
        if step == "futility":
            change(result)
        return result

    monkeypatch.setattr(phase4, "run_scientific_step", step)
    with pytest.raises(error, match=match):
        model.run()
    assert model.events == BEFORE and not FORBIDDEN.intersection(model.events)


@DECISIONS
def test_a_failed_progress_write_at_the_futility_step_dispatches_nothing_further(pod, decision):
    model = pod(decision)

    def progress(name, result):  # The session check inside the progress write fails once the record exists.
        if name == "futility":
            raise RuntimeError("The session deadline passed.")

    with pytest.raises(RuntimeError, match="session deadline"):
        v5.run_phase4_pipeline(model.request(), transport=object(), check_session=lambda: None,
                               record_progress=progress)
    assert model.events == BEFORE and not FORBIDDEN.intersection(model.events)
    assert model.run()["futility"]["decision"] == decision and model.computed == 1  # The re-run ends at the record.


# 2. The coordinator: truthful terminal states, a write-once boundary, and the same stop on recovery.

@pytest.fixture
def coordinator(tmp_path, monkeypatch, pod):
    mirrors = {tmp_path / "g2-mirror", tmp_path / "phase4-mirror"}
    original_stat = Path.stat

    def stat(path, **kwargs):  # Device separation is synthetic; every byte and hash check stays real.
        value = original_stat(path, **kwargs)
        if path in mirrors:
            fields = list(value)
            fields[2] += 1
            return os.stat_result(fields)
        return value

    monkeypatch.setattr(Path, "stat", stat)
    monkeypatch.setattr(campaign, "timing_recipes_v2", lambda: (("science_recipe_0", object()),), raising=False)
    for name in ("g2-primary", "g2-mirror"):
        (tmp_path / name).mkdir()

    def make(decision, events=None):
        model = pod(decision, events)
        config = {"execution_manifest": str(tmp_path / "manifest.json"), "hardware_profile": "runpod_rtx4090",
                  "remote_campaign": "/workspace/campaign", "durable_root": str(tmp_path / "g2-primary"),
                  "mirror_root": str(tmp_path / "g2-mirror"), "state_root": str(tmp_path / "state"),
                  "execution_scope": "g2_then_fold0_v5"}
        session = {"pod_id": "pod", "session_id": "session", "allocation_started_unix": 1000,
                   "deadline_unix": 20000, "max_total_seconds": 20000}
        prepared = {**IDENTITY, "fold": 0, "execution_manifest": config["execution_manifest"],
                    "queue_actions": ["clean_embeddings", "clean_fits", "s1"], "run_final_clean": False,
                    "durable_root": str(model.roots[0]), "mirror_root": str(model.roots[1]),
                    "host_state_root": str(tmp_path / "fold-state"), "max_download_bytes": 1000, "stage_files": [],
                    "remote_campaign": config["remote_campaign"],
                    "remote_request_template": {"execution_root": "/workspace/campaign/runs/fold0/phase4",
                                                "durable_root": str(model.roots[0])}}
        state = {"completed": [name for _, name in workstation.measurement_queue(SOURCE)],
                 "freeze1c": {"remote": "/workspace/campaign/g2_requests/freeze1c_v2.json"}}
        owner = SimpleNamespace(config_sha="a" * 64, register_output_roots=lambda roots: None)
        world = SimpleNamespace(pod=model, config=config, session=session, prepared=prepared, state=state,
                                state_path=tmp_path / "driver_state.json",
                                transport=SimpleNamespace(stop_controller=owner))
        world.dispatch = lambda: workstation._continue_runpod_phase4(
            world.config, world.transport, world.state, world.state_path, world.session, {"source": SOURCE},
            "remote_manifest", "remote_probe", lambda: world.session, lambda *args: dict(world.prepared))
        world.saved = lambda: json.loads(world.state_path.read_text())
        return world

    return make


@DECISIONS
def test_the_coordinator_records_a_truthful_terminal_state(coordinator, decision):
    world = coordinator(decision)
    world.dispatch()
    saved = world.saved()
    assert saved["status"] == STATES[decision][1] != "phase4_complete"
    assert saved["phase4"]["result"]["status"] == STATES[decision][0]
    assert saved["futility_boundary"] == saved["phase4"]["result"]["futility"]
    assert saved["futility_boundary"]["decision"] == decision and saved["futility_boundary"]["preserved_copies"] == 2
    assert sorted(saved["phase4"]["completed_queues"]) == sorted(BEFORE)  # The driver file sorts its keys.
    assert world.pod.events == BEFORE and not FORBIDDEN.intersection(world.pod.events)


@DECISIONS
@pytest.mark.parametrize("lost", ["coordinator_lost", "interrupted"])
def test_same_window_recovery_never_dispatches_past_the_boundary(coordinator, decision, lost):
    world = coordinator(decision)
    world.pod.fault = lost
    with pytest.raises(RuntimeError):
        world.dispatch()
    assert "futility_boundary" not in world.state and world.state.get("status") is None
    world.pod.fault = None
    if lost == "interrupted":  # No record exists: the step is refused, not repeated, and nothing follows it.
        with pytest.raises(RuntimeError, match="never blind replay"):
            world.dispatch()
        assert "futility_boundary" not in world.state
    else:  # The preserved record is read again and the session ends at it.
        world.dispatch()
        first = world.saved()
        assert first["status"] == STATES[decision][1]
        world.state = world.saved()  # A new coordinator process starts from the saved driver file.
        world.dispatch()
        assert world.saved() == first  # A further identical entry changes nothing.
    assert world.pod.computed == 1 and not FORBIDDEN.intersection(world.pod.events)


def test_recovery_cannot_change_the_reviewed_request_or_the_recorded_boundary(coordinator):
    world = coordinator("continue")
    world.dispatch()
    events = len(world.pod.events)
    world.prepared["run_final_clean"] = True
    with pytest.raises(workstation.WorkstationError, match="cannot change during recovery"):
        world.dispatch()
    world.prepared["run_final_clean"] = False
    world.state["futility_boundary"] = {**world.state["futility_boundary"], "decision": "stop_futility"}
    with pytest.raises(workstation.WorkstationError, match="futility boundary changed during recovery"):
        world.dispatch()
    assert not FORBIDDEN.intersection(world.pod.events[events:])


RECORD = {"unit_id": UNIT, "decision": "continue", "record_sha256": "1" * 64, "preserved_copies": 2}


@pytest.mark.parametrize("result,match", [
    ({"status": "passed", "futility": RECORD}, "verified terminal boundary"),  # Fold 0 never completes Phase 4.
    ({"status": "phase4_complete", "futility": RECORD}, "verified terminal boundary"),
    ({"futility": RECORD}, "verified terminal boundary"),
    ({"status": "futility_boundary"}, "futility boundary changed"),  # A boundary without its record.
    ({"status": "stop_futility", "futility": "stop_futility"}, "futility boundary changed"),
])
def test_fold0_accepts_only_a_boundary_result_with_its_record(coordinator, monkeypatch, result, match):
    world = coordinator("continue")
    monkeypatch.setattr(v5, "run_phase4_pipeline", lambda *args, **kwargs: dict(result))
    with pytest.raises(workstation.WorkstationError, match=match):
        world.dispatch()
    assert world.state.get("status") != "phase4_complete" and "futility_boundary" not in world.state


@pytest.mark.parametrize("result,status", [({"status": "passed"}, "phase4_complete"),
                                           ({"status": "futility_boundary", "futility": RECORD}, None),
                                           ({"status": "stop_futility", "futility": RECORD}, None)])
def test_a_later_fold_keeps_its_own_terminal_state(coordinator, monkeypatch, result, status):
    world = coordinator("continue")
    world.prepared.update(fold=1, remote_request_template={
        **world.prepared["remote_request_template"], "execution_root": "/workspace/campaign/runs/fold1/phase4"})
    monkeypatch.setattr(v5, "run_phase4_pipeline", lambda *args, **kwargs: dict(result))
    if status is None:  # A boundary result belongs to fold 0 only.
        with pytest.raises(workstation.WorkstationError, match="verified terminal boundary"):
            world.dispatch()
        assert world.state.get("status") is None
    else:
        world.dispatch()
        assert world.saved()["status"] == status and "futility_boundary" not in world.saved()


@DECISIONS
def test_the_session_closes_out_at_the_boundary(cached_driver, coordinator, monkeypatch, decision):
    driver = cached_driver
    world = coordinator(decision, driver.events)
    driver.session["max_total_seconds"] = 7020
    driver.config.update({key: world.config[key] for key in ("execution_manifest", "remote_campaign", "durable_root",
                                                             "mirror_root", "execution_scope")},
                         g2_max_download_bytes=10 * workstation.GIB)
    driver.ready["source"].update(SOURCE)
    write(driver.path, driver.config)
    driver.transport.stop_controller = world.transport.stop_controller
    state_path = Path(driver.config["state_root"]) / "driver_state.json"
    proof = write(Path(driver.config["state_root"]) / "preservation_set.json", {"format": "fixture"})
    monkeypatch.setattr(workstation, "_stage_runpod_runtime", lambda *a: None)
    monkeypatch.setattr(workstation, "_prepare_g2_v5", lambda *a: None)
    monkeypatch.setattr(workstation, "_execute_queue", lambda config, transport, state, *a: state.update(
        completed=world.state["completed"]))
    monkeypatch.setattr(workstation, "_complete_g2_v5", lambda config, transport, state, *a: state.update(
        freeze1c=world.state["freeze1c"]))
    monkeypatch.setattr(workstation, "_preserve_runpod_session_outputs",
                        lambda *a: driver.events.append("preserve") or proof)

    def closeout(path):
        driver.events.append("closeout")
        return {"status": "DELETED", "pod_id": "owned-pod", "provider_absence_verified": True,
                "preservation_sha256": sha(path)}

    driver.controller.closeout = closeout
    gate = lambda config, path, ready: {"config_sha256": workstation._sha(path), "execution_manifest_sha256": "9" * 64}
    state = workstation._run_runpod_g2(driver.config, driver.path, controller=driver.controller,
                                       transport=driver.transport, pre_allocation=gate,
                                       prepare_phase4=lambda *a: dict(world.prepared),
                                       prepare_g2=lambda *a: {}, complete_g2=lambda *a: {})
    saved = json.loads(state_path.read_text())
    assert saved == state and saved["status"] == STATES[decision][1]
    assert saved["closed"] is True and saved["provider_stop_verified"] is True
    assert saved["closeout"]["status"] == "DELETED" and "closeout_pending" not in saved
    assert saved["futility_boundary"]["decision"] == decision and not saved.get("preservation_error")
    events = driver.events
    assert [name for name in events if name in BEFORE] == BEFORE and not FORBIDDEN.intersection(events)
    # One allocation; the record is preserved, then the pod is stopped, then the two-copy closeout runs.
    assert events.count("start") == events.count("stop") == events.count("closeout") == 1
    assert events.index("futility") < events.index("preserve") < events.index("stop") < events.index("closeout")
    assert not workstation.closeout_interrupted(saved)


# 3. The launch gate: what fold 0 may request, and which launch-window outcome the boundary takes.

@pytest.mark.parametrize("change", ["run_final_clean", "keep_alive_for_operator"])
def test_the_launch_gate_refuses_work_behind_the_boundary_before_any_allocation(approved, monkeypatch, change):
    config, path, auth_path, authorization, source = approved
    monkeypatch.setattr(workstation, "_run_runpod_g2", lambda *a, **k: pytest.fail("Provider dispatch reached"))
    request_path = Path(config["phase4"]["request_path"])
    request = json.loads(request_path.read_text())

    def bind():  # The maintainer's authorization binds exactly these reviewed bytes.
        write(request_path, request)
        config["phase4"]["request_sha256"] = sha(request_path)
        write(path, config)
        authorization["config_sha256"] = sha(path)
        write(auth_path, authorization)

    bind()
    assert v5.validate_launch_authorization(config, path, auth_path, source) == authorization
    if change == "run_final_clean":
        request["run_final_clean"] = True
    else:
        config["runpod"]["keep_alive_for_operator"] = True
    bind()
    with pytest.raises(v5.LaunchWindowError, match="Fold 0 ends at its futility record"):
        v5.run_v5(config, path, authorization_path=auth_path, window_id="window")
    ledger = v5.FailedWindowLedger(authorization["failure_ledger"], authorization["failure_ledger_authorization_sha256"])
    assert ledger.status()["failed_windows"] == 0 and ledger.status()["unresolved_window"] is None


@pytest.mark.parametrize("status,failed", [("stopped_at_futility_boundary", 0), ("stopped_futility", 0),
                                           ("interrupted", 1)])
def test_a_preserved_boundary_stop_is_a_completed_launch_window(approved, monkeypatch, status, failed):
    config, path, auth_path, authorization, source = approved

    def session(config, path, *, pre_allocation, **callbacks):
        pre_allocation(config, path, {"source": source})
        state = {"start_attempted": True, "status": status, "pod_id": "owned", "closed": True,
                 "closeout": {"status": "DELETED", "pod_id": "owned", "provider_absence_verified": True,
                              "preservation_sha256": "1" * 64}}
        write(Path(config["state_root"]) / "driver_state.json", state)
        return state

    monkeypatch.setattr(workstation, "_run_runpod_g2", session)
    assert v5.run_v5(config, path, authorization_path=auth_path, window_id="window")["status"] == status
    ledger = v5.FailedWindowLedger(authorization["failure_ledger"], authorization["failure_ledger_authorization_sha256"])
    assert ledger.status()["failed_windows"] == failed and ledger.status()["unresolved_window"] is None


# 4. Later sessions: a pass whose records hold a futility step is not resumed under this manifest.

def _preserve_members(world, members):
    """Add files to the earlier session's preserved Phase-4 set, with every receipt and digest binding kept."""
    receipt_path = world.tmp / "receipt_p4.json"
    receipt = json.loads(receipt_path.read_text())
    inventory = receipt["remote_inventory_path"]
    roots = [Path(receipt["primary_root"]), Path(receipt["mirror_root"])]
    rows = [row for row in receipt["files"] if row["path"] != inventory]
    for relative, text in members.items():
        for root in roots:
            write(root / relative, text)
        rows.append({"path": relative, "size_bytes": len(text.encode()), "sha256": sha(roots[0] / relative)})
    for root in roots:
        write(root / inventory, {"session_id": "s1", "files": rows, "pending_transfer": None, "pending_unit": None})
    rows.append({"path": inventory, "size_bytes": (roots[0] / inventory).stat().st_size,
                 "sha256": sha(roots[0] / inventory)})
    write(receipt_path, {**receipt, "files": rows})
    proof = json.loads(world.proof.read_text())
    proof["receipts"][1]["sha256"] = sha(receipt_path)
    write(world.proof, proof)
    world.driver["closeout"]["preservation_sha256"] = sha(world.proof)


def _later_fold(world):
    """The same earlier session, named by a later-fold configuration that meets that fold's own requirements."""
    world.driver.update(completed=[name for _, name in workstation.measurement_queue(SOURCE)],
                        pretrain_preparation={"sha256": sha(world.pretrain), "local_path": str(world.pretrain),
                                              "record": dict(IDENTITY)})
    world.config["execution_fold"] = 1


RESULT = json.dumps({"status": "passed", "unit_id": UNIT, "decision": "continue"}) + "\n"
PRESERVED = {
    # The terminal record with its attempt, as a finished futility unit is preserved.
    "terminal_record": {f"runs/{UNIT}/complete.json": RESULT, f"runs/{UNIT}/attempt1/unit_result.json": RESULT},
    # The step produced its result and was acknowledged, but the session ended before its terminal record.
    "result_without_terminal_record": {f"runs/{UNIT}/attempt1/unit_result.json": RESULT,
                                       f"runs/{UNIT}/attempt1/output1.json": "{}\n"},
    # The step started and left neither.
    "started_only": {f"runs/{UNIT}/attempt1/scientific_outputs_pending.json": "{}\n"},
}
STATE_EVIDENCE = [
    ({"status": "stopped_at_futility_boundary"}, "holds its futility record"),
    ({"status": "interrupted", "futility_boundary": {"decision": "continue"}}, "holds its futility record"),
    ({"status": "stopped_futility"}, "binding futility stop cannot resume"),
    ({"status": "interrupted", "futility_boundary": {"decision": "stop_futility"}}, "binding futility stop"),
]


@pytest.mark.parametrize("fold", [0, 1])
@pytest.mark.parametrize("files", ["records", "full"])
@pytest.mark.parametrize("evidence,match", STATE_EVIDENCE)
def test_no_later_session_resumes_a_pass_whose_state_holds_the_boundary(build, fold, files, evidence, match):
    world = build()
    if fold:
        _later_fold(world)
        world.rebind()
    assert v5.validate_resume(world.config, SOURCE, files=files)["state"]  # The same pass resumes without it.
    world.driver.update(evidence)
    world.rebind()
    with pytest.raises(v5.LaunchWindowError, match=match):
        v5.validate_resume(world.config, SOURCE, files=files)


@pytest.mark.parametrize("fold", [0, 1])
@pytest.mark.parametrize("files", ["records", "full"])
@pytest.mark.parametrize("preserved", sorted(PRESERVED))
def test_no_later_session_resumes_a_pass_whose_records_hold_a_futility_step(build, fold, files, preserved):
    # A lost driver write: the state shows only an interruption, and the preserved records show the step.
    world = build()
    if fold:
        _later_fold(world)
        world.rebind()
    assert v5.validate_resume(world.config, SOURCE, files=files)["state"]["status"] == "interrupted"
    _preserve_members(world, PRESERVED[preserved])
    world.rebind()
    with pytest.raises(v5.LaunchWindowError, match="preserved fold-0 records hold a futility step"):
        v5.validate_resume(world.config, SOURCE, files=files)


def test_a_futility_request_that_never_started_keeps_its_first_execution(build):
    # Only the uploaded request is preserved: the step never started on the pod, so a later session may run it once.
    world = build()
    _preserve_members(world, {f"commands/{UNIT}_attempt1.json": "{}\n"})
    world.rebind()
    assert v5.validate_resume(world.config, SOURCE, files="full")["attempts"][UNIT] == 2


def test_resume_attempts_never_repeat_a_started_futility_step(tmp_path):
    root = tmp_path / "preserved"
    write(root / f"commands/{UNIT}_attempt1.json", "{}")
    # Only the uploaded request exists: the step never started on the pod, so this is its first execution.
    assert v5.resume_attempts(root) == {UNIT: 2}
    (root / "runs" / UNIT / "attempt1").mkdir(parents=True)
    # The step started and left no result: diagnosis, never a second run.
    with pytest.raises(v5.LaunchWindowError, match="Partial scientific outputs require diagnosis"):
        v5.resume_attempts(root)
    write(root / "runs" / UNIT / "attempt1" / "unit_result.json", {"status": "passed"})
    write(root / "runs" / UNIT / "complete.json", {"status": "passed", "attempt": 1})
    assert v5.resume_attempts(root) == {UNIT: 1}  # A finished record keeps its attempt and is only returned.


# 5. The pod's own refusal, unchanged by this correction: a started futility step is never replayed.

@pytest.mark.parametrize("left", ["attempt_only", "partial_outputs"])
def test_the_pod_refuses_to_replay_a_started_futility_step(tmp_path, monkeypatch, left):
    from contextlib import contextmanager

    root = tmp_path / "phase4"
    request = {"execution_root": str(root), "unit_id": UNIT, "scientific_step": "futility", "attempt": 1}

    @contextmanager
    def execution(request, policy):
        yield SimpleNamespace(state={}, admit=lambda *a: pytest.fail("The step was admitted again."))

    monkeypatch.setattr(phase4, "_runtime", lambda request, **kwargs: (SOURCE, None, None, None, None))
    monkeypatch.setattr(phase4.hot_storage, "require_restored_attempts", lambda root: None)
    monkeypatch.setattr(phase4, "require_restored_scientific_outputs", lambda root: None)
    monkeypatch.setattr(phase4, "_execution", execution)
    monkeypatch.setattr(phase4.subprocess, "run", lambda *a, **k: pytest.fail("A worker was started."))
    attempt = root / "runs" / UNIT / "attempt1"
    attempt.mkdir(parents=True)
    if left == "partial_outputs":
        write(attempt / "output1.json", "{}\n")
    with pytest.raises(phase4.Phase4Error, match="never blind replay"):
        phase4.run_scientific_unit(request)
    assert not (root / "commands").exists() and not (root / "runs" / UNIT / "complete.json").exists()
