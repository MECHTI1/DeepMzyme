"""Failure-oriented recovery checks use local archives and a fake provider."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import resume_gvp_capacity_colab as recovery


def save(path, value):
    recovery.atomic_json(path, value)


def receipt():
    return dict(session="owned", endpoint="endpoint-one", observed_at_unix=500,
                returncode=0, command=["colab", "sessions"],
                stdout="[colab] No active sessions found on server.\n", stderr="")


def ledger():
    return dict(ceiling_seconds=21600, allocations=[dict(session="owned", started_at_unix=100)])


def test_reconciliation_is_idempotent_and_preserves_the_before_state():
    before = ledger()
    result = recovery.reconciled_ledger(before, receipt())
    assert before == ledger()
    assert result["allocations"][0]["stopped_at_unix"] == 500
    assert result["allocations"][0]["started_at_unix"] == 100
    assert result["allocations"][0]["termination_kind"] == "provider_loss"
    assert recovery.reconciled_ledger(result, receipt()) == result
    assert result["ceiling_seconds"] == 21600


@pytest.mark.parametrize("change", [dict(session="unknown"), dict(observed_at_unix=90),
                                    dict(returncode=1), dict(command=["colab", "stop"]),
                                    dict(stdout="[owned] endpoint-one | Hardware: G4 | Variant: GPU")])
def test_invalid_absence_never_closes_an_allocation(change):
    before = ledger()
    with pytest.raises(ValueError):
        recovery.reconciled_ledger(before, receipt() | change)
    assert before == ledger()


def test_named_or_orphaned_owned_endpoint_blocks_reallocation():
    with pytest.raises(ValueError, match="present or ambiguous"):
        recovery.confirm_absence("[?] endpoint-one | Hardware: G4 | Variant: GPU", "owned", "endpoint-one")
    recovery.confirm_absence("[?] unrelated | Hardware: G4 | Variant: GPU", "owned", "endpoint-one")
    with pytest.raises(ValueError, match="Unrecognized"):
        recovery.confirm_absence("", "owned")


def test_duplicate_allocation_identity_is_rejected():
    data = ledger()
    data["allocations"].append(deepcopy(data["allocations"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        recovery.reconciled_ledger(data, receipt())


def test_a_second_controller_cannot_take_the_host_lock(tmp_path):
    with recovery.ownership_lock(tmp_path):
        with pytest.raises(BlockingIOError):
            with recovery.ownership_lock(tmp_path):
                pytest.fail("Second owner admitted")
    with recovery.ownership_lock(tmp_path):
        pass


def test_only_one_restart_of_an_interrupted_scientific_cell(tmp_path):
    spec = dict(id="discovery_depth2_split_s43", phase="discovery", seed=43, changes={"gvp_layers": 2})
    assert recovery.prior_attempts(tmp_path, spec) == []
    first = tmp_path / "session_1" / ("launch_intent_" + spec["id"] + ".json")
    save(first, spec)
    assert len(recovery.prior_attempts(tmp_path, spec)) == 1
    with pytest.raises(ValueError, match="scientific identity"):
        recovery.prior_attempts(tmp_path, spec | {"seed": 42})
    save(tmp_path / "session_2" / first.name, spec)
    with pytest.raises(ValueError, match="retry limit"):
        recovery.prior_attempts(tmp_path, spec)


def test_receipts_cannot_be_overwritten(tmp_path):
    path = tmp_path / "receipt.json"
    recovery.write_once(path, {"charge": 400})
    recovery.write_once(path, {"charge": 400})
    with pytest.raises(ValueError, match="Conflicting"):
        recovery.write_once(path, {"charge": 0})
    assert recovery.read_json(path) == {"charge": 400}


@pytest.fixture
def frozen_archive(tmp_path):
    root = tmp_path / "root"
    output = root / recovery.RELATIVE_OUTPUT
    execution = output / "execution"
    execution.mkdir(parents=True)
    source = root / "src/example.py"
    source.parent.mkdir()
    source.write_text("original source\n")
    controller = root / recovery.CONTROLLER
    controller.parent.mkdir()
    controller.write_text("original controller\n")
    save(output / "splits.json", {"frozen": True})
    save(output / "discovery_matrix.json", [])
    manifest = dict(study="gvp_capacity_diagnostic_v1", gpu_ceiling_seconds=21600,
                    held_out_evaluation=False, promotion=False, reduced_maximum_fits=34,
                    source_files={"src/example.py": recovery.digest(source)})
    save(output / "manifest.json", manifest)
    paths = [source, controller, output / "manifest.json", output / "splits.json", output / "discovery_matrix.json"]
    inventory = {str(p.relative_to(root)): recovery.digest(p) for p in paths}
    save(execution / "input_inventory.json", inventory)
    with tarfile.open(execution / "inputs.tar.gz", "w:gz") as archive:
        for path in paths + [execution / "input_inventory.json"]:
            archive.add(path, arcname=str(path.relative_to(root)))
    part = execution / "part_000"
    part.write_bytes((execution / "inputs.tar.gz").read_bytes())
    save(execution / "transfer.json", dict(archive_sha256=recovery.digest(part),
         study_manifest_sha256=recovery.digest(output / "manifest.json"),
         parts=[dict(name=part.name, sha256=recovery.digest(part))]))
    return root, output, inventory


def test_archive_restores_frozen_source_without_reverting_live_edits(frozen_archive):
    root, output, inventory = frozen_archive
    (root / "src/example.py").write_text("user's newer source\n")
    recovery.verify_archive(output)
    staged = recovery.stage_source(output, inventory)
    assert (staged / "src/example.py").read_text() == "original source\n"
    assert (root / "src/example.py").read_text() == "user's newer source\n"
    (staged / "src/example.py").write_text("tampered\n")
    with pytest.raises(ValueError, match="Staged input changed"):
        recovery.stage_source(output, inventory)


def test_a_changed_scientific_manifest_or_input_is_refused(frozen_archive):
    _, output, _ = frozen_archive
    original = (output / "manifest.json").read_bytes()
    data = recovery.read_json(output / "manifest.json")
    save(output / "manifest.json", data | {"held_out_evaluation": True})
    with pytest.raises(ValueError, match="scientific scope"):
        recovery.verify_archive(output)
    (output / "manifest.json").write_bytes(original)
    (output / "execution/part_000").write_bytes(b"broken")
    with pytest.raises(ValueError, match="transfer part"):
        recovery.verify_archive(output)


def result_fixture(output, spec):
    run = output / "runs" / spec["id"]
    run.mkdir(parents=True)
    (run / "checkpoint.pt").write_bytes(b"synthetic checkpoint bytes")
    data = dict(spec=spec, held_out_evaluation=False, manifest_sha256=recovery.digest(output / "manifest.json"),
                files={"checkpoint.pt": recovery.digest(run / "checkpoint.pt")})
    save(run / "capacity_result.json", data)
    retrieval = output / "execution/retrieval" / spec["id"]
    retrieval.mkdir(parents=True, exist_ok=True)
    with tarfile.open(retrieval / "run.tar.gz", "w:gz") as archive:
        archive.add(run, arcname=spec["id"])
    save(retrieval / "verified.json", dict(id=spec["id"], archive_sha256=recovery.digest(retrieval / "run.tar.gz")))
    return data


def test_completion_requires_the_result_record_and_bound_archive(tmp_path):
    save(tmp_path / "manifest.json", {"study": "frozen"})
    spec = dict(id="one", phase="discovery", seed=42)
    save(tmp_path / "discovery_matrix.json", [spec])
    result_fixture(tmp_path, spec)
    assert len(recovery.verified_results(tmp_path)) == 1
    path = tmp_path / "runs/one/capacity_result.json"
    data = recovery.read_json(path)
    save(path, data | {"invented_score": 1})
    with pytest.raises(ValueError, match="differs from retrieved archive"):
        recovery.verified_results(tmp_path)
    save(path, data)
    (tmp_path / "runs/one/checkpoint.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Corrupt result"):
        recovery.verified_results(tmp_path)


def test_partial_retrieval_is_preserved_and_only_transfer_is_retried(tmp_path):
    save(tmp_path / "manifest.json", {"study": "frozen"})
    spec = dict(id="one", phase="discovery", seed=42)
    save(tmp_path / "discovery_matrix.json", [spec])
    partial = tmp_path / "execution/retrieval/one"
    partial.mkdir(parents=True)
    (partial / "part_000").write_bytes(b"old interrupted download")
    class Controller:
        calls = 0
        def retrieve(self, provider, output, status):
            self.calls += 1
            if self.calls == 1:
                partial.mkdir()
                (partial / "part_000").write_bytes(b"new interrupted download")
                raise TimeoutError("download interrupted")
            result_fixture(output, spec)
    controller = Controller()
    result = recovery.retrieve_with_recovery(controller, None, tmp_path, {"id": "one"})
    assert result["spec"] == spec and controller.calls == 2
    saved = list((tmp_path / "execution/recovery/partial_retrievals/one").glob("*/part_000"))
    assert {p.read_bytes() for p in saved} == {b"old interrupted download", b"new interrupted download"}


@pytest.mark.parametrize("still_present", [False, True])
def test_shutdown_needs_provider_absence_even_when_stop_fails(tmp_path, still_present):
    data = ledger()
    class Provider:
        session = "owned"
        def command(self, *args, **kwargs):
            raise RuntimeError("session unavailable")
        def call(self, *args, **kwargs):
            return "[owned] endpoint-one | Hardware: G4" if still_present else receipt()["stdout"]
    if still_present:
        with pytest.raises(RuntimeError, match="absence unverified"):
            recovery.stop_confirmed(recovery, Provider(), tmp_path, data["allocations"][0], data, tmp_path / "ledger.json")
        assert "stopped_at_unix" not in data["allocations"][0]
    else:
        recovery.stop_confirmed(recovery, Provider(), tmp_path, data["allocations"][0], data, tmp_path / "ledger.json")
        assert data["allocations"][0]["provider_listing_verified_absent"]
        assert data["allocations"][0]["termination_kind"] == "provider_loss"


def test_ambiguous_launch_is_not_retried_and_teardown_runs(tmp_path, monkeypatch):
    output = tmp_path / "output"
    execution = output / "execution"
    save(execution / "allocation_ledger.json", ledger())
    save(execution / "recovery/initial_provider_absence.json", receipt())
    save(execution / "progress.json", {"reduced": True})
    save(output / "manifest.json", {"study": "frozen"})
    save(output / "discovery_matrix.json", [dict(id="one", phase="discovery", seed=42)])
    monkeypatch.setattr(recovery, "verify_archive", lambda _: ({}, {recovery.CONTROLLER: "frozen"}))
    monkeypatch.setattr(recovery, "stage_source", lambda *_: tmp_path)
    monkeypatch.setattr(recovery, "verified_results", lambda _: [])
    monkeypatch.setattr(recovery.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=receipt()["stdout"], stderr=""))
    guardian = SimpleNamespace(pid=123, poll=lambda: None, terminate=lambda: None, wait=lambda **k: None)
    monkeypatch.setattr(recovery.subprocess, "Popen", lambda *a, **k: guardian)
    monkeypatch.setattr(recovery.signal, "signal", lambda *a: None)
    class Provider:
        def __init__(self, session, log):
            self.session = session
        def command(self, action, *a, **k):
            return f"[{self.session}] endpoint-two | Hardware: G4 | Variant: GPU" if action == "status" else "ok"
        def call(self, *a, **k):
            return receipt()["stdout"]
    launches = []
    def launch(*args):
        launches.append(args)
        raise TimeoutError("launch may have succeeded")
    controller = SimpleNamespace(Provider=Provider, SESSION_CEILING=14400, SHUTDOWN_RESERVE=900,
        charge=lambda rows, now: 400, atomic_json=recovery.atomic_json,
        cost_gate=lambda *a: dict(admitted=True, worst_measured_fit_seconds=250),
        setup_remote=lambda *a: {"gpu": "fake"}, launch_fit=launch)
    monkeypatch.setattr(recovery, "load_controller", lambda *a: controller)
    with pytest.raises(TimeoutError, match="launch may"):
        recovery.resume(tmp_path, output)
    assert len(launches) == 1
    assert recovery.read_json(execution / "closeout.json")["all_owned_sessions_provider_verified_stopped"]
    assert recovery.read_json(execution / "closeout.json")["status"] == "incomplete"


@pytest.mark.parametrize("action", ["status", "resume"])
def test_missing_study_cannot_create_state_or_touch_a_provider(tmp_path, monkeypatch, action):
    output = tmp_path / "missing_study"
    def forbidden(*args, **kwargs):
        pytest.fail("Missing study contacted a provider")
    monkeypatch.setattr(recovery.subprocess, "run", forbidden)
    monkeypatch.setattr(recovery.subprocess, "Popen", forbidden)
    call = recovery.describe if action == "status" else recovery.resume
    with pytest.raises(ValueError, match="missing"):
        call(tmp_path, output)
    assert not output.exists()


def test_budget_exhaustion_uses_all_prior_charges_and_preserves_matrix():
    data = dict(allocations=[dict(session="old", started_at_unix=100, stopped_at_unix=21600)])
    seen = []
    def gate(allocated, measured, completed, maximum):
        seen.append((allocated, measured, completed, maximum))
        return {"admitted": False}
    controller = SimpleNamespace(charge=lambda rows, now: sum(r["stopped_at_unix"]-r["started_at_unix"] for r in rows),
                                 cost_gate=gate)
    with pytest.raises(ValueError, match="six-hour ceiling"):
        recovery.require_admitted(controller, data, [{"total_seconds": 250}])
    assert seen == [(21500, [250], 1, 34)]


def test_absence_receipt_is_captured_once_without_replacing_an_earlier_observation(tmp_path):
    status = tmp_path / "execution/session_1/provider_status.txt"
    status.parent.mkdir(parents=True)
    status.write_text("[owned] endpoint-one | Hardware: G4 | Variant: GPU")
    calls = []
    class Provider:
        def __init__(self, session, log):
            assert session == "owned"
        def call(self, *args, **kwargs):
            calls.append(args)
            return receipt()["stdout"]
    first = recovery.absence_receipt(tmp_path, ledger(), SimpleNamespace(Provider=Provider))
    second = recovery.absence_receipt(tmp_path, ledger(), SimpleNamespace(Provider=Provider))
    assert first == second and len(calls) == 1
    assert first["endpoint"] == "endpoint-one"
