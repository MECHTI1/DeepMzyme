"""Synthetic checks of the resumed-session execution path: launch ordering, pre-stage, restore and closeout.

The pod side of the restore and of the closeout inventory runs as the same programs on local folders, and
ssh, scp and rsync are replaced by local copies; no provider, network, GPU or campaign data is used.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from ec_clean30 import amendment_rules, execution_v5 as v5, phase3, phase4, runpod_controller as rc, workstation
from ec_clean30_fold0_fixtures import (  # noqa: F401
    BASE, HOUR, IDENTITY, MASKED, SOURCE, Transport, build, cached_driver, fold_world, local_provider,
    preserve_fold, sha, write,
)

REPO = Path(__file__).resolve().parents[1]
REAL_RUN = subprocess.run  # The restore program runs for real even while ssh/scp/rsync are local copies.
RECIPE = "ecc30timing_science_recipe_0"


class RestoreTransport(Transport):
    def remote(self, argv, *, timeout):
        assert argv[:3] == [sys.executable, "-c", v5.RESTORE_PROGRAM] and timeout > 0
        self.commands.append((argv, timeout))
        result = REAL_RUN(argv, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr[-2000:])
        return result.stdout


def tree(*roots):
    """Path -> (size, digest) of every file under the roots."""
    rows = {}
    for root in roots:
        for path in sorted(Path(root).rglob("*")):
            if path.is_file():
                rows[str(path)] = (path.stat().st_size, sha(path))
    return rows


def flip(path):
    """Change one byte and keep the size."""
    data = bytearray(Path(path).read_bytes())
    data[0] = (data[0] + 1) % 256
    Path(path).write_bytes(bytes(data))


def g2_runs(extra=None):
    """A preserved G2 root shaped like the earlier session's: each unit's runner record beside its run folder."""
    files = {f"runs/{RECIPE}_attempt1/g2_complete.json": json.dumps(IDENTITY),
             f"runs/{RECIPE}_attempt1/g2_timing.json": json.dumps(IDENTITY),
             f"runs/{RECIPE}_attempt1.runner.json": json.dumps({"returncode": 0}),
             f"commands/{RECIPE}_attempt1.json": "{}", f"commands/{RECIPE}_attempt1.log": "ran\n",
             "runs/ecc30_pretrain_fold0_attempt1/pretrain_complete.json": json.dumps(IDENTITY),
             "runs/ecc30_pretrain_fold0_attempt1.runner.json": json.dumps({"returncode": 0}),
             "runs/ecc30timing_science_recipe_1_attempt1.runner.json": json.dumps({"returncode": 1}),
             **{f"runs/{RECIPE}_attempt1/cache/{index:04d}.pt": f"payload {index}\n" for index in range(300)}}
    files.update(extra or {})
    return files


@pytest.fixture
def devices(build, monkeypatch):
    """The new pass's mirror roots on their own synthetic devices; every byte and hash check stays real."""
    extra = set()
    current = Path.stat

    def stat(path, **kwargs):
        value = current(path, **kwargs)
        if path in extra:
            fields = list(value)
            fields[2] += 7
            return os.stat_result(fields)
        return value

    monkeypatch.setattr(Path, "stat", stat)
    return extra


def resumed(build, devices, monkeypatch, **options):
    world = build(**options)
    world.config["g2_max_download_bytes"] = 10 ** 9
    world.rebind()
    devices.update({world.tmp / "new_g2_mirror", world.tmp / "new_p4_mirror"})
    monkeypatch.setattr(workstation, "preservation_storage", lambda config, source, request: [])
    world.outputs, _ = v5._resume_outputs(world.config, v5.validate_resume(world.config, SOURCE, files="records"))
    return world


# B1: the earlier session's runner records.

def test_runner_records_count_as_their_attempt_and_other_suffixes_stay_refused(build, tmp_path):
    build()  # The fixture's measurement queue.
    root = tmp_path / "w11_g2"
    for name in g2_runs():
        write(root / name, "x")
    attempts = v5.g2_resume_attempts(root, SOURCE)
    # A terminal unit keeps its attempt; a unit with only its runner record gets its single identical retry.
    assert attempts == {"pretrain": 1, "science_recipe_0": 1, "science_recipe_1": 2}
    for bad in (f"runs/{RECIPE}_attempt1.runner.json.partial", f"runs/{RECIPE}_attempt3.runner.json",
                f"commands/{RECIPE}_attempt1.json.tmp", f"runs/{RECIPE}_attempt1.runner"):
        write(root / bad, "x")
        with pytest.raises(v5.LaunchWindowError, match="unsupported retry history"):
            v5.g2_resume_attempts(root, SOURCE)
        (root / bad).unlink()
    write(root / f"runs/{RECIPE}_attempt2.runner.json", "x")
    write(root / f"runs/{RECIPE}_attempt2/g2_complete.json", "x")
    with pytest.raises(v5.LaunchWindowError, match="multiple results"):
        v5.g2_resume_attempts(root, SOURCE)


@pytest.mark.parametrize("files", ["records", "full"])
def test_resume_validation_returns_the_attempt_maps_and_refuses_bad_history_first(build, files):
    world = build(g2_files=g2_runs())
    resume = v5.validate_resume(world.config, SOURCE, files=files)
    assert resume["g2_attempts"] == {"pretrain": 1, "science_recipe_0": 1, "science_recipe_1": 2}
    assert resume["attempts"] == {BASE: 1, MASKED: 2}
    with pytest.raises(v5.LaunchWindowError, match="either the full copies or the records"):
        v5.validate_resume(world.config, SOURCE, files="staged")
    bad = build(g2_files=g2_runs({f"runs/{RECIPE}_attempt1.runner.json.partial": "x"}))
    with pytest.raises(v5.LaunchWindowError, match="unsupported retry history"):
        v5.validate_resume(bad.config, SOURCE, files=files)


def test_runner_records_restore_and_name_their_g2_evidence(build):
    world = build(g2_files=g2_runs())
    state = {}
    transport = RestoreTransport()
    v5.restore_session(world.config, transport, state, world.state_path, SOURCE, lambda: None)
    assert state["resume_g2_attempts"] == {"pretrain": 1, "science_recipe_0": 1, "science_recipe_1": 2}
    evidence = state["g2_evidence"]["science_recipe_0"]
    # The recorded path stays the earlier primary copy; its digest is the receipt's.
    assert evidence["local_path"] == str(world.g2_primary / f"runs/{RECIPE}_attempt1/g2_complete.json")
    assert evidence["sha256"] == sha(evidence["local_path"]) and state["completed"] == ["science_recipe_0"]
    pod = world.remote / "runs/fold0/g2"
    assert (pod / f"runs/{RECIPE}_attempt1.runner.json").is_file()


# B2: the launch checks the records first and both full copies right before it reserves a window.

def launch_setup(world, tmp_path, monkeypatch, coordinator):
    monkeypatch.setattr(phase3, "authenticated_chain", lambda source: source)
    monkeypatch.setattr(workstation, "validate_phase4_plan", lambda *args: None)
    monkeypatch.setattr(workstation, "prepare_session_chain_receipt", lambda config: None)
    monkeypatch.setattr(phase3, "verify_execution_source", lambda *args, **kwargs: SOURCE)
    monkeypatch.setattr(workstation, "_run_runpod_g2", coordinator)
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o700)
    ledger = shared / "ledger.json"
    write(ledger, v5.empty_failure_ledger(ledger, "f" * 64)).chmod(0o600)
    world.config.update(provider="runpod", execution_scope="g2_then_fold0_v5")
    world.rebind()
    authorization = write(tmp_path / "authorization.json", {
        "format": v5.AUTHORIZATION_FORMAT, "launch_authorized": True, "provider": "runpod",
        "scope": "g2_then_fold0_v5", "approved_by": "maintainer-fixture", **IDENTITY,
        "execution_manifest_sha256": SOURCE["execution_manifest_sha256"], "fold0_cost_ceiling_usd": 85,
        "failed_window_limit": 2, "failure_ledger": str(ledger), "failure_ledger_authorization_sha256": "f" * 64,
        "config_sha256": sha(world.config_path)})
    return authorization, v5.FailedWindowLedger(ledger, "f" * 64)


def spy(monkeypatch, calls):
    real_resume, real_plan, real_begin = v5.validate_resume, v5.plan_restore, v5.FailedWindowLedger.begin_window
    monkeypatch.setattr(v5, "validate_resume",
                        lambda config, source, *, files="full": calls.append(files) or real_resume(config, source, files=files))
    monkeypatch.setattr(v5, "plan_restore", lambda *args: calls.append("plan") or real_plan(*args))
    monkeypatch.setattr(v5.FailedWindowLedger, "begin_window",
                        lambda self, *args, **kwargs: calls.append("window") or real_begin(self, *args, **kwargs))


def coordinator_for(calls):
    def coordinator(config, config_path, *, pre_allocation, **kwargs):
        calls.append("quote")  # _prepared_preflight's quote check precedes the launch gate.
        pre_allocation(config, config_path, {"source": SOURCE})
        calls.append("allocate")
        write(Path(config["state_root"]) / "driver_state.json", {"completed": []})  # Nothing was started.
        return {}
    return coordinator


def test_launch_hashes_both_copies_only_after_the_quote_and_before_the_window(build, tmp_path, monkeypatch):
    world = build()
    calls = []
    authorization, ledger = launch_setup(world, tmp_path, monkeypatch, coordinator_for(calls))
    spy(monkeypatch, calls)
    v5.run_v5(world.config, world.config_path, authorization_path=authorization, window_id="resume-fixture")
    assert calls == ["records", "quote", "full", "records", "plan", "window", "allocate"]
    assert not (world.remote / "runs").exists()  # The dry plan staged nothing.


@pytest.mark.parametrize("copy", ["primary", "mirror"])
def test_a_changed_preserved_byte_is_refused_before_any_window(build, tmp_path, monkeypatch, copy):
    world = build()
    calls = []
    authorization, ledger = launch_setup(world, tmp_path, monkeypatch, coordinator_for(calls))
    spy(monkeypatch, calls)
    root = world.p4_primary if copy == "primary" else Path(str(world.p4_primary) + "_mirror")
    flip(root / f"runs/{BASE}/attempt1/unit_result.json")
    with pytest.raises(v5.LaunchWindowError, match="missing or changed"):
        v5.run_v5(world.config, world.config_path, authorization_path=authorization, window_id="resume-fixture")
    assert calls == ["records", "quote", "full"] and ledger.status()["failed_windows"] == 0
    assert ledger.status()["unresolved_window"] is None


def test_unsupported_g2_history_is_refused_before_the_coordinator(build, tmp_path, monkeypatch):
    world = build(g2_files=g2_runs({f"runs/{RECIPE}_attempt1.runner.json.partial": "x"}))
    calls = []
    authorization, ledger = launch_setup(world, tmp_path, monkeypatch, coordinator_for(calls))
    with pytest.raises(v5.LaunchWindowError, match="unsupported retry history"):
        v5.run_v5(world.config, world.config_path, authorization_path=authorization, window_id="resume-fixture")
    assert calls == [] and ledger.status()["unresolved_window"] is None


def test_restore_plan_is_pure_and_refuses_a_bad_mirror_byte_before_any_upload(build):
    world = build()
    before = tree(world.tmp)
    plan = v5.plan_restore(world.config, v5.validate_resume(world.config, SOURCE, files="records"), SOURCE)
    assert tree(world.tmp) == before and plan["attempts"] == {BASE: 1, MASKED: 2}
    # Archived bytes are read from the mirror copies.
    assert all(row["local"].startswith(str(world.tmp / "old_")) and "_mirror/" in row["local"] for row in plan["rows"])
    flip(world.tmp / "old_p4_mirror" / f"runs/{BASE}/complete.json")
    transport = RestoreTransport()
    with pytest.raises(v5.LaunchWindowError, match="changed before it was archived"):
        v5.restore_session(world.config, transport, {}, world.state_path, SOURCE, lambda: None)
    assert not transport.uploads and not transport.commands


def test_restore_uses_the_mirror_copy_and_its_own_per_file_allowance(build):
    world = build(g2_files=g2_runs())
    flip(world.p4_primary / f"runs/{MASKED}/attempt1/farm/partial.pt")  # Only the primary copy decayed.
    transport = RestoreTransport()
    v5.restore_session(world.config, transport, {}, world.state_path, SOURCE, lambda: None)
    restored = world.root / f"runs/{MASKED}/attempt1/farm/partial.pt"
    assert restored.read_text() == "lost before its result"
    (argv, timeout), = transport.commands
    assert timeout >= workstation.INVENTORY_MIN_SECONDS + int(argv[6]) * v5.RESTORE_FILE_SECONDS
    timings = json.loads((Path(world.config["state_root"]) / "restore_timings.json").read_text())["restore"]
    assert timings["files"] == int(argv[6]) and timings["batches"][0]["restored"] == int(argv[6])


# B3: an unbilled pre-stage of the inherited files, and closeout witnesses.

def test_prestage_copies_inherited_files_into_both_new_copies_and_never_writes_the_old_roots(
        build, devices, monkeypatch):
    world = resumed(build, devices, monkeypatch, g2_files=g2_runs())
    old = tree(*(world.tmp / name for name in ("old_g2", "old_g2_mirror", "old_p4", "old_p4_mirror")))
    result = v5.prestage_resume(world.config, world.config_path, source=SOURCE)
    assert tree(*(world.tmp / name for name in ("old_g2", "old_g2_mirror", "old_p4", "old_p4_mirror"))) == old
    proof = json.loads(Path(result["proof"]).read_text())
    assert proof["preservation_sha256"] == world.config["resume"]["preservation_sha256"]
    assert [row["execution_root"] for row in proof["roots"]] == [row["execution_root"] for row in world.outputs]
    for row in proof["roots"]:
        receipt = json.loads(Path(row["receipt_path"]).read_text())
        assert sha(row["receipt_path"]) == row["receipt_sha256"]
        copied = [entry for entry in receipt["files"] if v5._prestaged_member(entry["path"])]
        assert row["files"] == len(copied) > 0
        for entry in receipt["files"]:
            for root in (row["primary_root"], row["mirror_root"]):
                target = Path(root) / entry["path"]
                if v5._prestaged_member(entry["path"]):
                    assert sha(target) == entry["sha256"] and target.stat().st_size == entry["size_bytes"]
                else:
                    assert not target.exists()  # The restore and closeout own these.
        items = v5.prestaged_witnesses(world.config, row["execution_root"], Path(row["primary_root"]),
                                       Path(row["mirror_root"]))
        assert {tuple(item[:3]) for item in items} == {
            (entry["path"], entry["size_bytes"], entry["sha256"]) for entry in copied}
        # Each witness carries the identity of both synced copies, as they are now.
        assert all(item[3:] == [workstation.copy_fingerprint(Path(root) / item[0])
                                for root in (row["primary_root"], row["mirror_root"])] for item in items)
    assert not list(world.tmp.rglob("*.prestage-partial"))
    # Another root pair or a changed receipt binds no witness.
    row = proof["roots"][0]
    assert v5.prestaged_witnesses(world.config, row["execution_root"], Path(row["mirror_root"]),
                                  Path(row["primary_root"])) == []
    # A repeated pre-stage keeps every file and adds nothing.
    stamps = {path: path.stat().st_mtime_ns for path in Path(row["primary_root"]).rglob("*") if path.is_file()}
    again = v5.prestage_resume(world.config, world.config_path, source=SOURCE)
    assert again["proof_sha256"] != "" and all(path.stat().st_mtime_ns == stamp for path, stamp in stamps.items())
    timings = json.loads((Path(world.config["state_root"]) / "prestage_timings.json").read_text())["prestage"]
    assert all(root["copied"] == 0 and root["kept"] == root["files"] for root in timings["roots"])


def test_interrupted_prestage_resumes_after_a_torn_journal_line(build, devices, monkeypatch):
    world = resumed(build, devices, monkeypatch, g2_files=g2_runs())
    monkeypatch.setattr(v5, "PRESTAGE_BATCH_FILES", 50)
    real, copies = v5._prestage_copy, []

    def interrupted(source, targets, entry):
        if len(copies) == 120:
            raise KeyboardInterrupt("synthetic stop")
        copies.append(entry["path"])
        return real(source, targets, entry)

    monkeypatch.setattr(v5, "_prestage_copy", interrupted)
    with pytest.raises(KeyboardInterrupt):
        v5.prestage_resume(world.config, world.config_path, source=SOURCE)
    journal = Path(world.config["state_root"]) / v5.PRESTAGE_JOURNAL
    assert len(journal.read_text().splitlines()) == 2 and not (Path(world.config["state_root"]) / v5.PRESTAGE_PROOF).exists()
    with journal.open("a") as handle:
        handle.write('{"format": "' + v5.PRESTAGE_FORMAT + '", "files": [["runs/x"')  # A torn last line.
    monkeypatch.setattr(v5, "_prestage_copy", real)
    result = v5.prestage_resume(world.config, world.config_path, source=SOURCE)
    for row in result["roots"]:
        receipt = json.loads(Path(row["receipt_path"]).read_text())
        for entry in receipt["files"]:
            if v5._prestaged_member(entry["path"]):
                for root in (row["primary_root"], row["mirror_root"]):
                    assert sha(Path(root) / entry["path"]) == entry["sha256"]


@pytest.mark.parametrize("problem", ["differing_target", "bad_source", "started", "locked", "storage", "devices"])
def test_prestage_refusals(build, devices, monkeypatch, problem):
    world = resumed(build, devices, monkeypatch, g2_files=g2_runs())
    name = f"runs/{RECIPE}_attempt1/cache/0007.pt"
    target = world.tmp / "new_g2" / name
    match = {"differing_target": "never replaced", "bad_source": "differs from its receipt",
             "started": "before this pass attempts a start", "locked": "driver lock",
             "storage": "too little space", "devices": "two independent devices"}[problem]
    if problem == "differing_target":
        write(target, "payload 7?\n")
    elif problem == "bad_source":
        flip(world.tmp / "old_g2_mirror" / name)
    elif problem == "started":
        write(world.state_path, {"start_attempted": True})
    elif problem == "storage":
        monkeypatch.setattr(workstation, "preservation_storage",
                            lambda config, source, request: [(os.stat(world.tmp).st_dev, 10, 0)])
    elif problem == "devices":
        devices.clear()
    if problem == "locked":
        Path(world.config["state_root"]).mkdir(parents=True)
        import fcntl
        with (Path(world.config["state_root"]) / "driver.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with pytest.raises(v5.LaunchWindowError, match=match):
                v5.prestage_resume(world.config, world.config_path, source=SOURCE)
        return
    with pytest.raises(v5.LaunchWindowError, match=match):
        v5.prestage_resume(world.config, world.config_path, source=SOURCE)
    assert not (Path(world.config["state_root"]) / v5.PRESTAGE_PROOF).exists()
    if problem == "differing_target":
        assert target.read_text() == "payload 7?\n"
    if problem in {"storage", "devices", "started"}:
        assert not [path for path in (world.tmp / "new_g2").rglob("*") if path.is_file()]
    assert not list(world.tmp.rglob("*.prestage-partial"))


def test_a_changed_prestaged_copy_voids_the_proof_and_is_never_kept_unread(build, devices, monkeypatch):
    world = resumed(build, devices, monkeypatch, g2_files=g2_runs())
    v5.prestage_resume(world.config, world.config_path, source=SOURCE)
    assert v5.verify_prestage(world.config)["status"] == "prestaged_unchanged"
    row = world.outputs[0]
    name = f"runs/{RECIPE}_attempt1/cache/0007.pt"
    for key in ("primary_root", "mirror_root"):
        copy = Path(row[key]) / name
        original = copy.read_bytes()
        flip(copy)  # Same size on either new device.
        with pytest.raises(v5.LaunchWindowError, match="changed since the pre-stage"):
            v5.verify_prestage(world.config)
        # The repeat hashes the changed copy instead of trusting its journal line and size, and is refused;
        # the earlier proof was removed first, so nothing binds the refused file any more.
        with pytest.raises(v5.LaunchWindowError, match="never replaced"):
            v5.prestage_resume(world.config, world.config_path, source=SOURCE)
        proof = Path(world.config["state_root"]) / v5.PRESTAGE_PROOF
        assert not proof.exists() and v5.verify_prestage(world.config) is None
        assert v5.prestaged_witnesses(world.config, row["execution_root"], Path(row["primary_root"]),
                                      Path(row["mirror_root"])) == []
        copy.write_bytes(original)  # Same bytes, new change time: hashed again, kept and witnessed again.
        v5.prestage_resume(world.config, world.config_path, source=SOURCE)
        assert v5.verify_prestage(world.config)["status"] == "prestaged_unchanged"


def test_a_prestage_proof_binds_its_journal_and_boot(build, devices, monkeypatch):
    world = resumed(build, devices, monkeypatch, g2_files=g2_runs())
    v5.prestage_resume(world.config, world.config_path, source=SOURCE)
    row = world.outputs[0]
    witnesses = lambda: v5.prestaged_witnesses(world.config, row["execution_root"], Path(row["primary_root"]),
                                               Path(row["mirror_root"]))
    assert witnesses()
    journal = Path(world.config["state_root"]) / v5.PRESTAGE_JOURNAL
    content = journal.read_bytes()
    with journal.open("a") as handle:
        handle.write("{}\n")  # Any change after the finished run unbinds the proof.
    assert witnesses() == []
    with pytest.raises(v5.LaunchWindowError, match="no longer binds"):
        v5.verify_prestage(world.config)
    journal.write_bytes(content)
    assert witnesses()
    monkeypatch.setattr(workstation, "boot_id", lambda: "00000000-0000-4000-8000-000000000000")
    assert witnesses() == []
    # After a restart the repeat reads every copy again and journals it for the new boot.
    calls = []
    real = v5._prestage_copy
    monkeypatch.setattr(v5, "_prestage_copy", lambda *args: calls.append(1) or real(*args))
    v5.prestage_resume(world.config, world.config_path, source=SOURCE)
    assert len(calls) == sum(row["files"] for row in json.loads(
        (Path(world.config["state_root"]) / v5.PRESTAGE_PROOF).read_text())["roots"])
    assert witnesses() and v5.verify_prestage(world.config)["status"] == "prestaged_unchanged"


def test_launch_refuses_a_changed_prestaged_copy_before_any_window(build, devices, tmp_path, monkeypatch):
    world = resumed(build, devices, monkeypatch, g2_files=g2_runs())
    calls = []
    authorization, ledger = launch_setup(world, tmp_path, monkeypatch, coordinator_for(calls))
    v5.prestage_resume(world.config, world.config_path, source=SOURCE)
    flip(Path(world.outputs[1]["mirror_root"]) / f"runs/{BASE}/attempt1/unit_result.json")
    spy(monkeypatch, calls)
    with pytest.raises(v5.LaunchWindowError, match="changed since the pre-stage"):
        v5.run_v5(world.config, world.config_path, authorization_path=authorization, window_id="resume-fixture")
    assert calls == ["records", "quote", "full", "records", "plan"]
    assert ledger.status()["failed_windows"] == 0 and ledger.status()["unresolved_window"] is None


def test_preservation_storage_keeps_the_launch_rule(monkeypatch, tmp_path):
    facts = {"g2": (1, 100), "g2m": (2, 200), "p4": (1, 100), "p4m": (3, 300)}
    monkeypatch.setattr(workstation, "_storage_facts", lambda path: facts[path.name])
    config = {"durable_root": "/x/g2", "mirror_root": "/x/g2m", "g2_max_download_bytes": 7}
    request = {"durable_root": "/x/p4", "mirror_root": "/x/p4m", "max_download_bytes": 11}
    rows = workstation.preservation_storage(config, {"format": phase3.EXECUTION_V5_FORMAT}, request)
    gib = workstation.GIB
    assert rows == [(1, 100, 5 * gib + 18), (2, 200, 5 * gib + 7), (1, 100, 5 * gib + 18), (3, 300, 5 * gib + 11)]


def pod_closeout(world, tmp_path, index, controller):
    deadline = time.time() + 3600
    transport = workstation.RunpodTransport({"state_root": str(tmp_path / "ctl"), "bootstrap_python": sys.executable,
                                             "runpod": {"ssh_alias": "synthetic-pod"}}, controller)
    transport.deadline_unix, transport.reserve_seconds = deadline, 15
    session = {"pod_id": "s2-pod", "session_id": "s2", "deadline_unix": deadline}
    _, reviewed = v5._bound_record(world.config["phase4"], "request")
    path = workstation._preserve_runpod_outputs(
        world.config, transport, controller, session, output=world.outputs[index],
        receipt_name=f"preservation_receipt_{index}.json", max_download_bytes=10 ** 9,
        host_state_root=reviewed["host_state_root"] if index else None)
    timings = json.loads((Path(world.config["state_root"]) / "closeout_timings.json").read_text())
    return path, timings[world.outputs[index]["execution_root"]]


def gate(config, receipt_path, controller, pod="s2-pod", session="s2"):
    """The controller's own two-copy check before deletion, on this synthetic allocation."""
    state = {"session_id": session, "workload_admitted": True}
    owner = SimpleNamespace(config=config, config_sha=controller.config_sha, _owned_id=lambda: pod,
                            _state=lambda: state)
    return rc.RunpodController._verify_preservation(owner, Path(receipt_path), _allow_set=False)


def rehearsal(build, devices, monkeypatch):
    world = resumed(build, devices, monkeypatch, g2_files=g2_runs())
    v5.prestage_resume(world.config, world.config_path, source=SOURCE)
    v5.restore_session(world.config, RestoreTransport(), {}, world.state_path, SOURCE, lambda: None)
    return world, SimpleNamespace(config_sha="8" * 64, heartbeat=lambda: None)


def test_resume_rehearsal_closeout_skips_only_witnessed_inherited_files(build, devices, local_provider, monkeypatch,
                                                                       tmp_path):
    world, controller = rehearsal(build, devices, monkeypatch)
    for index, row in enumerate(world.outputs):
        receipt_path, timings = pod_closeout(world, tmp_path, index, controller)
        receipt = json.loads(receipt_path.read_text())
        prior = json.loads(Path(v5._resume_receipts(world.config["resume"])[index]["path"]).read_text())
        inherited = [entry for entry in prior["files"] if v5._prestaged_member(entry["path"])]
        # Every inherited file is covered by its pre-stage witness; only the execution state is copied.
        assert timings["covered"] == len(inherited) and timings["download_bytes"] < 2000
        assert timings["download_files"] == (1 if index else 0) and timings["uncovered"] == timings["download_files"]
        # The receipt and inventory are those of the full path: every remote file, with the pod's digests.
        inventory = json.loads((Path(row["primary_root"]) / "closeout/remote_inventory.json").read_text())
        assert receipt["files"][:-1] == inventory["files"] and len(inventory["files"]) == len(inherited) + index
        assert gate(world.config, receipt_path, controller) == sha(receipt_path)


@pytest.mark.parametrize("copy,change", [("primary", "flip"), ("mirror", "flip"),
                                         ("primary", "truncate"), ("mirror", "truncate")])
def test_a_decayed_copy_never_passes_the_closeout_gate(build, devices, local_provider, monkeypatch, tmp_path,
                                                      copy, change):
    world, controller = rehearsal(build, devices, monkeypatch)
    name = f"runs/{RECIPE}_attempt1/cache/0042.pt"
    damaged = Path(world.outputs[0]["primary_root" if copy == "primary" else "mirror_root"]) / name
    if change == "flip":
        flip(damaged)
    else:
        damaged.write_text("x")
    receipt_path, timings = pod_closeout(world, tmp_path, 0, controller)
    # Same size or not, the rewrite changed the copy's identity, so the witness no longer holds: the file
    # takes the full path and is downloaded or mirrored again while the pod still runs.
    assert timings["uncovered"] == 1 and timings["download_files"] == (1 if copy == "primary" else 0)
    assert gate(world.config, receipt_path, controller) == sha(receipt_path)
    assert damaged.read_text() == "payload 42\n"


def test_a_witness_from_another_boot_is_ignored(build, devices, local_provider, monkeypatch, tmp_path):
    world, controller = rehearsal(build, devices, monkeypatch)
    monkeypatch.setattr(workstation, "boot_id", lambda: "00000000-0000-4000-8000-000000000000")
    receipt_path, timings = pod_closeout(world, tmp_path, 0, controller)
    # After a restart every pre-stage witness is void and each file is hashed again (nothing differs here).
    assert timings["covered"] == 0 and timings["download_files"] == 0
    assert gate(world.config, receipt_path, controller) == sha(receipt_path)
    # The launch check refuses the stale proof before any window, and the pre-stage keeps nothing unread.
    with pytest.raises(v5.LaunchWindowError, match="run --prepare-resume again"):
        v5.verify_prestage(world.config)


def test_a_rewrite_of_a_journaled_copy_takes_the_full_path(tmp_path, monkeypatch):
    primary, mirror, state = tmp_path / "primary", tmp_path / "mirror", tmp_path / "state"
    write(primary / "runs/u/a.bin", "payload\n")
    host = workstation.HostPull(SimpleNamespace(), primary, state, max_download_bytes=10, deadline_unix=0,
                                mirror_root=mirror)
    host._mirror("runs/u/a.bin")
    entry = {"path": "runs/u/a.bin", "size_bytes": 8, "sha256": sha(primary / "runs/u/a.bin")}
    synced = []
    monkeypatch.setattr(workstation.os, "sync", lambda: synced.append(1))
    host.journal("/pod/root", [entry])
    row, = [json.loads(line) for line in (state / workstation.VERIFIED_COPIES).read_text().splitlines()]
    # Both copies were synced before the line; the line names this boot and both copies' identities.
    assert synced == [1] and row["boot_id"] == workstation.boot_id()
    assert row["files"] == [[entry["path"], 8, entry["sha256"], workstation.copy_fingerprint(primary / entry["path"]),
                             workstation.copy_fingerprint(mirror / entry["path"])]]
    witnesses = workstation._copy_witnesses({}, host, "/pod/root")
    assert workstation._witnessed(host, entry["path"], entry, witnesses)
    for root in (primary, mirror):
        copy = root / entry["path"]
        original = copy.read_bytes()
        flip(copy)  # Same size, other bytes.
        assert not workstation._witnessed(host, entry["path"], entry, witnesses)
        copy.write_bytes(original)  # Same bytes again, but a new change time: still not witnessed.
        assert not workstation._witnessed(host, entry["path"], entry, witnesses)
    assert workstation._copy_witnesses({}, host, "/pod/other") == {}
    monkeypatch.setattr(workstation, "boot_id", lambda: "00000000-0000-4000-8000-000000000000")
    assert workstation._copy_witnesses({}, host, "/pod/root") == {}


# A decay after the billed pass: the gate refuses, one local repair from a verified copy, the same gate again.

@pytest.mark.parametrize("damage", ["primary", "mirror", "both_inherited", "both_new"])
def test_post_stop_repair_restores_a_copy_only_from_verified_bytes(build, devices, local_provider, monkeypatch,
                                                                   tmp_path, damage):
    world, controller = rehearsal(build, devices, monkeypatch)
    receipt_path, _ = pod_closeout(world, tmp_path, 0, controller)
    name = "closeout/remote_inventory.json" if damage == "both_new" else f"runs/{RECIPE}_attempt1/cache/0042.pt"
    roots = {"primary": ["primary_root"], "mirror": ["mirror_root"]}.get(damage, ["primary_root", "mirror_root"])
    copies = [Path(world.outputs[0][key]) / name for key in roots]
    expected = copies[0].read_bytes()
    old = tree(*(world.tmp / folder for folder in ("old_g2", "old_g2_mirror")))
    for copy in copies:
        flip(copy)
    with pytest.raises(rc.RunpodError, match="differ between local copies"):
        gate(world.config, receipt_path, controller)
    repaired = workstation.repair_preserved_copies(world.config, receipt_path)
    assert tree(*(world.tmp / folder for folder in ("old_g2", "old_g2_mirror"))) == old  # Only read.
    if damage == "both_new":
        # No copy of this pass's own file is left to copy from: nothing is invented, and the gate refuses.
        assert repaired == 0
        with pytest.raises(rc.RunpodError, match="differ between local copies"):
            gate(world.config, receipt_path, controller)
        return
    assert repaired == len(copies) and all(copy.read_bytes() == expected for copy in copies)
    assert gate(world.config, receipt_path, controller) == sha(receipt_path)
    assert not list(world.tmp.rglob("*.repair-partial"))
    assert workstation.repair_preserved_copies(world.config, receipt_path) == 0


@pytest.mark.parametrize("repairable", [True, False])
def test_closeout_repeats_its_gate_once_after_a_repair(monkeypatch, tmp_path, repairable):
    calls = []

    def closeout(path):
        calls.append(path)
        if len(calls) == 1:
            raise rc.RunpodError("Preservation bytes are missing or differ between local copies.")
        return {"status": "DELETED"}

    monkeypatch.setattr(workstation, "repair_preserved_copies", lambda config, path: 1 if repairable else 0)
    run = lambda: workstation._closeout_with_repair({}, SimpleNamespace(closeout=closeout), tmp_path / "set.json")
    if repairable:
        assert run() == {"status": "DELETED"} and len(calls) == 2
    else:
        with pytest.raises(rc.RunpodError, match="differ between local copies"):
            run()
        assert len(calls) == 1


def test_closeout_interrupted_mid_copy_completes_on_its_repeat(build, devices, local_provider, monkeypatch, tmp_path):
    world, controller = rehearsal(build, devices, monkeypatch)
    for root in ("primary_root", "mirror_root"):
        (Path(world.outputs[1][root]) / f"runs/{BASE}/complete.json").unlink()  # Not yet copied anywhere.
    real = workstation.HostPull._mirror

    def crash(self, relative):
        raise KeyboardInterrupt("synthetic stop during closeout")

    monkeypatch.setattr(workstation.HostPull, "_mirror", crash)
    with pytest.raises(KeyboardInterrupt):
        pod_closeout(world, tmp_path, 1, controller)
    monkeypatch.setattr(workstation.HostPull, "_mirror", real)
    receipt_path, timings = pod_closeout(world, tmp_path, 1, controller)
    assert timings["uncovered"] == 2 and gate(world.config, receipt_path, controller) == sha(receipt_path)


def separate_mirror(monkeypatch, mirror):
    current = Path.stat

    def stat(path, **kwargs):
        value = current(path, **kwargs)
        if path == Path(mirror):
            fields = list(value)
            fields[2] += 7
            return os.stat_result(fields)
        return value

    monkeypatch.setattr(Path, "stat", stat)


def test_acknowledged_files_are_journaled_and_covered_on_the_next_closeout(tmp_path, local_provider, monkeypatch):
    world = fold_world(tmp_path)
    separate_mirror(monkeypatch, world.output["mirror_root"])
    log = world.fold / "commands/unit_a_attempt1.log"
    with world.execution as execution:
        execution.admit("unit_a", 60.0)
        log.parent.mkdir()
        log.write_text("the unit's own check refused its device\n")
        execution.record_result("unit_a", "failed_fit", 1.0)
        execution.persist([log])
    preserve_fold(world, tmp_path)
    journal = tmp_path / "fold-state" / workstation.VERIFIED_COPIES
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    assert {row["primary_root"] for row in rows} == {world.output["primary_root"]}
    assert "commands/unit_a_attempt1.log" in {item[0] for item in rows[0]["files"]}
    timings = json.loads((tmp_path / "state" / "closeout_timings.json").read_text())[world.output["execution_root"]]
    # The settle step acknowledged the unit's files (both copies verified) before the inventory, so the
    # same closeout already covers them; the rest, such as the execution state, takes the full path.
    covered = {item[0] for item in rows[0]["files"]}
    assert timings["covered"] == len(covered) and 0 < timings["covered"] < timings["files"]
    # The repeated closeout covers every file both copies were verified for.
    receipt, _ = preserve_fold(world, tmp_path)
    timings = json.loads((tmp_path / "state" / "closeout_timings.json").read_text())[world.output["execution_root"]]
    assert timings["covered"] == timings["files"] and timings["download_files"] == 0
    # A same-size decay of an acknowledged copy voids its witness: the billed pass repairs it before the stop.
    path = Path(world.config["state_root"]) / "preservation_receipt_1.json"
    assert gate({}, path, world.controller, pod="pod", session="session") == sha(path)
    for root in (world.output["primary_root"], world.output["mirror_root"]):
        copy = Path(root) / "commands/unit_a_attempt1.log"
        original = copy.read_bytes()
        flip(copy)
        preserve_fold(world, tmp_path)
        timings = json.loads((tmp_path / "state" / "closeout_timings.json").read_text())[world.output["execution_root"]]
        assert timings["uncovered"] == 1
        assert timings["download_files"] == (1 if root == world.output["primary_root"] else 0)
        assert copy.read_bytes() == original
        assert gate({}, path, world.controller, pod="pod", session="session") == sha(path)


def test_a_rewritten_remote_file_has_two_witnesses_and_takes_the_full_path(tmp_path, local_provider):
    world = fold_world(tmp_path)
    log = world.fold / "commands/unit_a_attempt1.log"
    with world.execution as execution:
        execution.admit("unit_a", 60.0)
        log.parent.mkdir()
        log.write_text("first\n")
        execution.record_result("unit_a", "failed_fit", 1.0)
        execution.persist([log])
    preserve_fold(world, tmp_path)
    log.write_text("other\n")  # Same size, other digest on the pod.
    preserve_fold(world, tmp_path)
    log.write_text("first\n")  # Back to the first digest: two witnesses now exist for this path.
    preserve_fold(world, tmp_path)
    timings = json.loads((tmp_path / "state" / "closeout_timings.json").read_text())[world.output["execution_root"]]
    assert timings["uncovered"] >= 1 and timings["download_files"] == 1
    for root in (world.output["primary_root"], world.output["mirror_root"]):
        assert (Path(root) / "commands/unit_a_attempt1.log").read_text() == "first\n"


# F5: an interrupted post-stop closeout leaves its window open; the identical re-run finishes it.

def stopped_state(world, tmp_path, binding):
    receipt = write(Path(world.config["state_root"]) / "preservation_set.json", {"format": "fixture"})
    return {"config_sha256": sha(world.config_path), "closed": True, "provider_stop_verified": True,
            "start_attempted": True, "workload_admitted": True, "pod_id": "s2-pod", "session_id": "s2",
            "stop_status": {"pod_id": "s2-pod", "status": "EXITED", "runtime_status": "stopped"},
            "launch_binding": binding, "status": "interrupted",
            "closeout_pending": {"path": str(receipt), "sha256": sha(receipt)}}


@pytest.mark.parametrize("outcome", ["interrupted", "completed", "verification_failed", "set_changed"])
def test_post_stop_closeout_is_finished_by_the_identical_rerun(build, tmp_path, monkeypatch, outcome):
    world = build()
    authorization, ledger = launch_setup(world, tmp_path, monkeypatch,
                                         lambda *args, **kwargs: pytest.fail("no new session for a stopped pod"))
    binding = {"config_sha256": sha(world.config_path), "execution_manifest_sha256": SOURCE["execution_manifest_sha256"],
               "authorization_sha256": sha(authorization), "window_id": "resume-fixture",
               "failure_ledger": str(ledger.path)}
    ledger.begin_window("resume-fixture", config_sha256=sha(world.config_path),
                        execution_manifest_sha256=SOURCE["execution_manifest_sha256"],
                        pass_root=Path(world.config["state_root"]))
    write(world.state_path, stopped_state(world, tmp_path, binding))
    calls = []

    def closeout(path):
        calls.append(path)
        if outcome == "interrupted":
            raise KeyboardInterrupt("signal 15")
        if outcome == "verification_failed":
            raise rc.RunpodError("Preservation bytes are missing or differ between local copies.")
        return {"pod_id": "s2-pod", "session_id": "s2", "status": "DELETED", "provider_absence_verified": True,
                "preservation_sha256": sha(path)}

    controller = SimpleNamespace(closeout=closeout)
    if outcome == "set_changed":
        write(Path(world.config["state_root"]) / "preservation_set.json", {"format": "changed"})
    run = lambda: v5.run_v5(world.config, world.config_path, authorization_path=authorization,
                            window_id="resume-fixture", controller=controller)
    if outcome == "completed":
        state = run()
        assert state["closeout"]["status"] == "DELETED" and "closeout_pending" not in state
        assert ledger.status()["unresolved_window"] is None and ledger.status()["failed_windows"] == 0
        return
    expected = {"interrupted": KeyboardInterrupt, "verification_failed": rc.RunpodError,
                "set_changed": workstation.WorkstationError}[outcome]
    with pytest.raises(expected):
        run()
    state = json.loads(world.state_path.read_text())
    if outcome == "verification_failed":
        # A real verification failure is recorded as before, and the window closes as failed.
        assert state["preservation_error"].startswith("closeout: RunpodError")
        assert ledger.status()["failed_windows"] == 1 and ledger.status()["unresolved_window"] is None
        return
    # Interrupted, or a changed set: the window stays open for the same re-run or a decision.
    assert state["closeout_pending"] and "closeout" not in state and not state.get("preservation_error")
    assert ledger.status()["unresolved_window"]["window_id"] == "resume-fixture"
    assert ledger.status()["failed_windows"] == 0 and len(calls) == (0 if outcome == "set_changed" else 1)
    if outcome == "interrupted":
        controller.closeout = lambda path: {"pod_id": "s2-pod", "session_id": "s2", "status": "DELETED",
                                            "provider_absence_verified": True, "preservation_sha256": sha(path)}
        state = run()
        assert state["closeout"]["status"] == "DELETED"
        assert ledger.status()["unresolved_window"] is None and ledger.status()["failed_windows"] == 0


@pytest.mark.parametrize("failure", ["interrupt", "verification"])
def test_coordinator_writes_the_stop_before_its_closeout(cached_driver, monkeypatch, failure):
    case = cached_driver
    case.config["execution_scope"] = "g2_then_fold0_v5"
    case.config["g2_max_download_bytes"] = 10 * workstation.GIB
    case.ready["source"]["format"] = phase3.EXECUTION_V5_FORMAT
    case.ready["source"]["execution_manifest_sha256"] = "a" * 64
    write(case.path, case.config)
    state_path = Path(case.config["state_root"]) / "driver_state.json"
    proof = write(Path(case.config["state_root"]) / "preservation_set.json", {"format": "fixture"})
    monkeypatch.setattr(workstation, "_stage_runpod_runtime", lambda *a: None)
    monkeypatch.setattr(workstation, "_prepare_g2_v5", lambda *a: None)
    monkeypatch.setattr(workstation, "_complete_g2_v5", lambda *a: None)
    monkeypatch.setattr(workstation, "_continue_runpod_phase4", lambda *a: None)
    monkeypatch.setattr(workstation, "_preserve_runpod_session_outputs", lambda *a: proof)
    seen = []

    def closeout(path):
        seen.append(json.loads(state_path.read_text()))
        if failure == "interrupt":
            raise KeyboardInterrupt("signal 15")
        raise rc.RunpodError("Preservation bytes are missing or differ between local copies.")

    case.controller.closeout = closeout
    gate = lambda config, path, ready: {"config_sha256": workstation._sha(path), "execution_manifest_sha256": "a" * 64}
    with pytest.raises(KeyboardInterrupt if failure == "interrupt" else rc.RunpodError):
        workstation._run_runpod_g2(case.config, case.path, controller=case.controller, transport=case.transport,
                                   pre_allocation=gate, prepare_phase4=lambda *a: {},
                                   prepare_g2=lambda *a: {}, complete_g2=lambda *a: {})
    # The stop and the bound set were durable before the closeout began.
    assert seen[0]["closed"] is True and seen[0]["stop_status"]["runtimeStatus"] == "stopped"
    assert seen[0]["closeout_pending"] == {"path": str(proof), "sha256": sha(proof)}
    state = json.loads(state_path.read_text())
    assert workstation.closeout_interrupted(state) is (failure == "interrupt")
    assert bool(state.get("preservation_error")) is (failure == "verification")


# F6: a new unit needs its forecast, the shutdown reserve and the closeout reserve.

@pytest.mark.parametrize("remaining,pod_state,admitted", [(10 * HOUR, None, True), (2000, None, False),
                                                           (2000, "launched", True), (2000, "complete", True)])
def test_queue_keeps_the_closeout_reserve_before_a_new_unit(monkeypatch, tmp_path, remaining, pod_state, admitted):
    from ec_clean30 import workstation as ws

    source = {"scientific_identity_sha256": "a" * 64, "amendment_chain": ["b" * 64]}
    monkeypatch.setattr(phase4.phase3, "verify_execution_source", lambda *args, **kwargs: source)
    monkeypatch.setattr(phase4, "verify_authority", lambda *args, **kwargs: None)
    monkeypatch.setattr(phase4, "_attach_session", lambda *args, **kwargs: None)
    monkeypatch.setattr(phase4.phase3, "require_same_scientific_identity", lambda *args: None)
    monkeypatch.setattr(phase4.phase3, "require_record_precedence", lambda *args: None)
    monkeypatch.setattr(ws, "HostPull", lambda *args, **kwargs: SimpleNamespace(sync_pending=lambda root: None))
    dispatched, probes = [], []
    monkeypatch.setattr(ws, "complete_remote_unit", lambda transport, host, current, **kwargs:
                        dispatched.append(current["unit_id"]) or {"status": "passed", "unit_id": current["unit_id"]})
    root = "/campaign/runs/fold0/phase4"

    def stat_file(path):
        probes.append(path)
        if pod_state == "complete" and path == f"{root}/runs/u1/complete.json":
            return {"size_bytes": 1, "sha256": "0" * 64}
        if pod_state == "launched" and path.endswith("/u1_attempt1.json.job/command.json"):
            return {"size_bytes": 1, "sha256": "0" * 64}
        return None

    transport = SimpleNamespace(stat_file=stat_file, deadline_unix=None, reserve_seconds=30,
                                upload=lambda *a, **k: pytest.fail("no upload for a refused unit"),
                                stop_controller=SimpleNamespace(admit_output_root=lambda root: None))
    deadline = time.time() + remaining
    allocation = {"session_id": "s2", "allocation_started_unix": deadline - 20 * HOUR, "deadline_unix": deadline,
                  "max_total_seconds": 20 * HOUR, "g3_sha256": "c" * 64, "provider": "runpod", "pod_id": "pod",
                  "runpod_config_sha256": "d" * 64}
    request = {**allocation, "execution_manifest": "manifest.json", "durable_root": str(tmp_path / "primary"),
               "mirror_root": str(tmp_path / "mirror"), "host_state_root": str(tmp_path / "state"),
               "max_download_bytes": 100, "remote_runner": "/repo/scripts/run_ec_clean30_phase4.py",
               "remote_request_template": {**allocation, "execution_root": root, "expected_chain": ["b" * 64],
                                           "amendment_chain_receipt": "/receipt"}}
    units = [{"unit_id": "u1", "unit_timeout_seconds": 10, "forecast_seconds": 100}]
    run = lambda: phase4._run_remote_queue(request, units, action="s1-unit", next_gate="gate", transport=transport,
                                           check_session=lambda: None, close_session=False)
    if admitted:
        assert run()["completed"] == ["u1"] and dispatched == ["u1"]
        assert (probes == []) is (pod_state is None)
    else:
        with pytest.raises(phase4.Phase4Error, match="Too little allocation time"):
            run()
        assert dispatched == [] and len(probes) == 2
    assert phase4.CLOSEOUT_RESERVE_SECONDS >= workstation.RESERVE_SECONDS


# Amendment-2 rules: the pinned closures and C7 stay unchanged in every execution file.

def test_pinned_closures_and_private_imports_are_unchanged():
    amendment = json.loads((REPO / "docs/campaigns/ec_class_vs_clean30_2026-10/amendment2/amendment.json").read_text())
    for key, pin in amendment["pinned_execution_functions"].items():
        path, name = key.split("::")
        assert amendment_rules.function_closure_sha256(REPO / path, [name]) == pin, key
    execution = set(amendment["execution_file_set"]["files"])
    known = {str(path.relative_to(REPO)) for folder in ("src", "scripts") for path in (REPO / folder).rglob("*.py")}

    def protected(name):
        target = amendment_rules.module_file(name, known)
        return target is not None and target not in execution

    for path in sorted(execution):
        if path.endswith(".py"):
            text = (REPO / path).read_text()
            assert amendment_rules.private_import_problems(text, path, protected=protected) == [], path
            assert amendment_rules.private_import_problems(text, path) == [], path


@pytest.mark.parametrize("damage", [None, "remove", "replace", "after_verify"])
def test_new_window_verifies_bound_completed_unit_without_admitting_a_fit(build, devices, monkeypatch, damage):
    from contextlib import nullcontext
    from test_ec_clean30_workstation import LocalTransport

    world = resumed(build, devices, monkeypatch, g2_files=g2_runs())
    v5.prestage_resume(world.config, world.config_path, source=SOURCE)
    v5.restore_session(world.config, RestoreTransport(), {}, world.state_path, SOURCE, lambda: None)
    assert v5.verify_prestage(world.config)
    relative = f"runs/{BASE}/complete.json"
    marker = world.root / relative
    expected = sha(marker)
    primary = world.tmp / "new_p4"
    mirror = world.tmp / "new_p4_mirror"
    session = "successor"
    request_file = world.root / "queue" / session / f"{BASE}_attempt1.json"
    assert not Path(str(request_file) + ".job/command.json").exists()
    request = {"unit_id": BASE, "execution_root": str(world.root), "durable_root": str(primary),
               "python": sys.executable, "session_id": session, "pod_id": "new-pod", "provider": "runpod",
               "expected_chain": ["fixture"], "forecast_seconds": 7200}
    verified = []
    # The fixture has a synthetic checkpoint. Exercise real pre-stage, restore, submission and recovery logic.
    monkeypatch.setattr(phase4, "_runtime", lambda *a, **kw: (SOURCE, None, None, None, None))
    monkeypatch.setattr(phase4, "_unit", lambda *a: None)
    monkeypatch.setattr(phase4, "_execution", lambda *a: nullcontext(SimpleNamespace(state={})))
    monkeypatch.setattr(phase4, "_verify_result", lambda req, record, *a: verified.append(record["unit_id"]))

    class RecoveryTransport(LocalTransport):
        restored_completions = workstation.RunpodTransport.restored_completions

        def launch(self, argv, *, cwd, job_dir):
            unit = json.loads(Path(argv[-1]).read_text())
            assert unit["restored_complete_sha256"] == expected
            if job_dir.endswith(".job"):
                if damage == "remove":
                    marker.unlink()
                elif damage == "replace":
                    marker.write_text("{}")
            write(Path(job_dir) / "command.json", {"argv": argv, "cwd": cwd})
            try:
                phase4.run_unit(unit, kind="embedding")
            except (phase4.Phase4Error, FileNotFoundError):
                write(Path(job_dir) / "result.json", {"returncode": 1})
            else:
                if job_dir.endswith(".finalize") and damage == "after_verify":
                    marker.write_text("{}")
                write(Path(job_dir) / "result.json", {"returncode": 0})

    transport = RecoveryTransport()
    transport.stop_controller = SimpleNamespace(config=world.config, cfg={"unit_credit": {}},
        check_window_funds=lambda **kw: pytest.fail("Already-completed fit entered new-work admission."))
    host = workstation.HostPull(transport, primary, world.tmp / "new-host-state", max_download_bytes=100000,
                                deadline_unix=time.time() + 600, mirror_root=mirror)
    args = dict(action="clean-embeddings", runner=str(world.tmp / "scripts/phase4.py"),
                remote_request=str(request_file), completed_relative=relative, timeout=3)
    if damage is None:
        result = workstation.complete_remote_unit(transport, host, request, **args)
        assert result["unit_id"] == BASE and verified and set(verified) == {BASE}
    elif damage == "after_verify":
        with pytest.raises(workstation.WorkstationError, match="changed during verification"):
            workstation.complete_remote_unit(transport, host, request, **args)
        assert verified
    else:
        with pytest.raises(workstation.WorkstationError, match="Remote unit failed"):
            workstation.complete_remote_unit(transport, host, request, **args)
        assert not verified
    assert sha(primary / relative) == sha(mirror / relative) == expected
