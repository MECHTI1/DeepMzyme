"""CPU-only checks of the fixed Phase-4 scope and its execution boundaries."""
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import pytest

from ec_clean30 import campaign, phase4


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return path


def test_exact_frozen_s1_and_clean_scope():
    units = phase4.s1_units()
    assert len(units) == len({unit.unit_id for unit in units}) == 96
    assert Counter(unit.family for unit in units) == dict.fromkeys(campaign.FAMILIES, 16)
    assert {unit.seed for unit in units} == {42, 43}
    assert {unit.inner_fold for unit in units} == {0, 1, 2, 3}
    assert all(unit.epochs == 30 and unit.step == "S1" and unit.fold == 0 and not unit.lever_configs for unit in units)
    clean = phase4.clean_units()
    assert len(clean) == 8
    assert {(row["regime"], row["inner_fold"]) for row in clean} == {(arm, fold) for arm in ("r1_inner", "r2_inner") for fold in range(4)}
    assert all(row["seed"] == 42 and row["epochs"] == 2500 for row in clean)


@pytest.mark.parametrize("plan", [campaign.build_fold_plan(1), campaign.build_fold_plan(0, families=["F2"]),
                                  campaign.build_fold_plan(0, levers=[])])
def test_scope_refuses_other_fold_or_pruned_plan(plan):
    with pytest.raises(phase4.Phase4Error, match="unchanged complete"):
        phase4.s1_units(plan)


@pytest.fixture
def authority(tmp_path):
    source = {"execution_manifest_sha256": "1"*64, "freeze1_sha256": "2"*64}
    evidence = []
    for index in range(10):
        path = write_json(tmp_path / f"measurement{index}.json", {"index": index})
        evidence.append({"path": str(path), "sha256": phase4._sha(path)})
    freeze = write_json(tmp_path / "freeze1c.json", {"campaign_id": campaign.CAMPAIGN_ID, **source,
                      "lanes": 1, "evidence": evidence})
    record = {"campaign_id": campaign.CAMPAIGN_ID, "scope": phase4.SCOPE, "status": "approved",
              "execution_manifest_sha256": source["execution_manifest_sha256"], "freeze1c_sha256": phase4._sha(freeze),
              "ceilings": {"gpu_seconds": 7200, "cpu_seconds": 18000, "cost_usd": 3},
              "allocations": {"gpu1": {"resource": "gpu", "allocation_started_unix": 1000,
                  "deadline_unix": 8020, "max_total_seconds": 7200, "cost_ceiling_usd": 3}}}
    path = write_json(tmp_path / "g3.json", record)
    request = {"freeze1c": str(freeze), "g3_authority": str(path), "g3_sha256": phase4._sha(path),
               "session_id": "gpu1", "allocation_started_unix": 1000, "deadline_unix": 8020, "max_total_seconds": 7200}
    return request, source, record


def test_numerical_authority_and_measurement_binding(authority):
    request, source, record = authority
    verified, _ = phase4.verify_authority(request, source, resource="gpu")
    assert verified == record


@pytest.mark.parametrize("case", ["unapproved", "source", "scope", "ceiling", "cost", "session", "measurement"])
def test_authority_refuses_missing_or_changed_limits(authority, case):
    request, source, record = authority
    if case == "unapproved": record["status"] = "planned"
    elif case == "source": record["execution_manifest_sha256"] = "3"*64
    elif case == "scope": record["scope"] = "all_folds"
    elif case == "ceiling": record["ceilings"]["gpu_seconds"] = 7199
    elif case == "cost": record["ceilings"]["cost_usd"] = 2
    elif case == "session": request["allocation_started_unix"] += 1
    else:
        freeze = phase4._json(Path(request["freeze1c"]))
        Path(freeze["evidence"][0]["path"]).write_text("changed")
    write_json(Path(request["g3_authority"]), record)
    request["g3_sha256"] = phase4._sha(Path(request["g3_authority"]))
    with pytest.raises(phase4.Phase4Error):
        phase4.verify_authority(request, source, resource="gpu")


def test_measurement_relocation_preserves_frozen_hashes(authority, tmp_path):
    request, source, _ = authority
    freeze = phase4._json(Path(request["freeze1c"]))
    relocations = {}
    for index, row in enumerate(freeze["evidence"]):
        original = Path(row["path"])
        target = tmp_path / f"relocated{index}.json"
        target.write_bytes(original.read_bytes())
        original.unlink()
        relocations[row["sha256"]] = str(target)
    phase4.verify_authority({**request, "measurement_files": relocations}, source)


@pytest.mark.parametrize("filesystem,free,allowed", [("ext4", 6*1024**3, True), ("ext4", 5*1024**3, False), ("fuseblk", 50*1024**3, False)])
def test_hot_storage_requires_real_ext4_and_five_gib_reserve(tmp_path, monkeypatch, filesystem, free, allowed):
    monkeypatch.setattr(phase4.subprocess, "check_output", lambda *args, **kwargs: filesystem)
    monkeypatch.setattr(phase4.shutil, "disk_usage", lambda path: SimpleNamespace(free=free))
    if allowed:
        phase4.check_hot_storage(tmp_path / "new", 100_000_000)
    else:
        with pytest.raises(phase4.Phase4Error):
            phase4.check_hot_storage(tmp_path / "new", 100_000_000)


def test_clean_roots_allow_private_code_but_retain_test_and_other_fold_guards(tmp_path):
    root = tmp_path / campaign.CAMPAIGN_ID
    view = SimpleNamespace(campaign_root=root, forbidden_read_roots=[str(root / name) for name in
        ("clean_harness", "identity", "overlay", "views/fold1", "scoring_pass")])
    clone = root / "clean_harness/third_party/CLEAN"
    roots = phase4._clean_roots(view, {"clean_clone": str(clone)})
    assert root / "clean_harness" not in roots
    assert all(root / name in roots for name in ("identity", "overlay", "views/fold1", "scoring_pass"))
    assert any(clone in path.parents for path in roots)


def test_clean_queue_runs_eight_serial_units_and_finalizes_before_next(monkeypatch, tmp_path):
    calls = []
    def execute(request, *, kind):
        assert kind == "clean"
        calls.append(request["unit_id"])
        if len(calls) % 2:
            return {"status": "persisted"}
        return {"status": "passed", "unit_id": request["unit_id"], "export": str(tmp_path / request["unit_id"] / "validation.npz")}
    monkeypatch.setattr(phase4, "run_unit", execute)
    result = phase4.run_clean_queue({})
    expected = [row["unit_id"] for row in phase4.clean_units()]
    assert result["completed"] == expected
    assert calls == [key for key in expected for _ in range(2)]


def test_attempt_keeps_original_unit_identity_and_rejects_invalid_attempt(tmp_path):
    unit = phase4.s1_units()[0]
    request = {"unit_id": unit.unit_id, "execution_root": str(tmp_path), "attempt": 2}
    assert phase4._attempt(request) == tmp_path / "runs" / unit.unit_id / "attempt2"
    for value in (0, -1, True, 1.5):
        with pytest.raises(phase4.Phase4Error):
            phase4._attempt({**request, "attempt": value})


@pytest.fixture
def clean_input_fixture(monkeypatch):
    from training.ec_membership import write_ec_fold_membership

    with tempfile.TemporaryDirectory(prefix="ecc30-p4-input-") as temporary:
        root = Path(temporary)
        official = root / "split30_train_split_0.csv"
        rows = [(f"P0000{index}", f"{index % 6 + 1}.1.1.1", "ACDEFGHIKLMNPQRSTVWY"*4) for index in range(1, 9)]
        rows.append(("P00009", "1.1.1.1;2.1.1.1", "ACDEFGHIKLMNPQRSTVWY"*4))
        official.write_text("Entry\tEC number\tSequence\n" + "".join("\t".join(row)+"\n" for row in rows))
        membership = root / "inner_membership.csv"
        write_ec_fold_membership([{"structure_id": f"{key}__chain_A__EC_{ec}", "accession": key,
            "group_id": f"g{index}", "fold": str(index), "ec_class": ec[0]} for index, (key, ec, _) in enumerate(rows[:4])], membership)
        overlay = root / "ecc30-overlay-v1"
        files = []
        for inner in range(4):
            exclusions = overlay / f"identity/fold0/r1_inner_exclusions_fold0_inner{inner}.tsv"
            exclusions.parent.mkdir(parents=True, exist_ok=True)
            exclusions.write_text(f"accession\treason\nP0000{inner+1}\tinner_validation\n" +
                                  ("P00005\td3_neighbour\n" if inner == 0 else ""))
            allowlist = overlay / f"r2_lists/r2_inner_allowlist_fold0_inner{inner}.txt"
            allowlist.parent.mkdir(exist_ok=True)
            allowlist.write_text("".join(f"P0000{index}\n" for index in (1, 2, 3, 4, 9) if index != inner+1))
            files.extend({"path": path.relative_to(overlay).as_posix(), "sha256": phase4._sha(path)} for path in (exclusions, allowlist))
        manifest = write_json(overlay / "overlay_manifest.json", {"files": files})
        view = SimpleNamespace(fold=0, smoke=False, campaign_root=root / campaign.CAMPAIGN_ID,
            forbidden_read_roots=[str(root / "forbidden")], overlay={"id": overlay.name, "manifest_file": manifest.name,
            "manifest_sha256": phase4._sha(manifest)}, membership={"sha256": phase4._sha(membership)},
            membership_file=lambda: membership)
        source = {"source_sha256": {"src/ec_clean30/phase4.py": "1"*64}, "execution_manifest_sha256": "2"*64,
                  "official_fold0_training_sha256": phase4._sha(official), "published": False}
        authentication = []
        def verify(path, *, require_published=True):
            authentication.append(require_published)
            return source
        monkeypatch.setattr(phase4.phase3, "verify_execution_source", verify)
        monkeypatch.setattr(phase4.views.View, "load", lambda path: view)
        monkeypatch.setattr(phase4.phase3, "_bind_view", lambda *args: None)
        request = {"execution_manifest": str(root / "execution.json"), "view": str(root / "view"), "draft": True,
                   "official_train": str(official), "overlay": str(overlay), "out": str(root / "inputs")}
        yield request, view, source, authentication


def test_cpu_preparation_preserves_exact_queries_exclusions_and_several_class_r2(clean_input_fixture):
    request, view, source, authentication = clean_input_fixture
    record = phase4.prepare_clean_inputs(request)
    out = Path(request["out"])
    assert authentication == [False]  # Draft authentication is limited to this CPU preparation entry.
    assert len(record["units"]) == 8 and record["n_sequences"] == 9
    for unit in record["units"]:
        inner = unit["inner_fold"]
        assert (out / unit["queries"]).read_text() == f"P0000{inner+1}\n"
        if unit["regime"] == "r2_inner":
            assert "P00009" in (out / unit["allowlist"]).read_text()
            assert unit["n_training"] == 4
        else:
            assert unit["n_training"] == (7 if inner == 0 else 8)
    authority = {"clean_inputs_sha256": phase4._sha(out / "clean_inputs.json")}
    phase4._clean_inputs({"clean_inputs": str(out)}, source, view, authority)
    (out / record["units"][0]["queries"]).write_text("P00009\n")
    with pytest.raises(phase4.Phase4Error, match="input changed"):
        phase4._clean_inputs({"clean_inputs": str(out)}, source, view, authority)


def test_draft_cannot_enter_any_real_runtime(clean_input_fixture):
    request, _, _, authentication = clean_input_fixture
    with pytest.raises(phase4.Phase4Error, match="only for CPU input"):
        phase4._source_view(request)
    assert authentication == []


def test_altered_pinned_inner_inputs_refused(clean_input_fixture):
    request, _, _, _ = clean_input_fixture
    altered = Path(request["overlay"]) / "r2_lists/r2_inner_allowlist_fold0_inner0.txt"
    altered.write_text(altered.read_text() + "P00001\n")
    with pytest.raises(phase4.Phase4Error, match="Frozen inner list"):
        phase4.prepare_clean_inputs(request)


def test_real_execution_ownership_host_ack_and_verified_reuse(tmp_path, monkeypatch):
    from benchmarking.pmm_execution import CampaignExecution, ExecutionBlocked, ExecutionPolicy

    root, durable = tmp_path / "worker", tmp_path / "host"
    now = time.time()
    policy = ExecutionPolicy(deadline_unix=now+4000, max_total_seconds=4000, allocation_started_unix=now)
    unit = phase4.s1_units()[0]
    plan = write_json(tmp_path / "plan.json", campaign.build_fold_plan(0))
    request = {"execution_root": str(root), "durable_root": str(durable), "session_id": "fixture-allocation",
               "persistence_mode": "host_pull", "unit_id": unit.unit_id, "plan": str(plan), "forecast_seconds": 20,
               "unit_timeout_seconds": 20, "python": sys.executable}
    monkeypatch.setattr(phase4, "_runtime", lambda request, resource: ({}, None, policy, {}, {}))
    monkeypatch.setattr(phase4, "_verify_result", lambda *args: None)
    original = CampaignExecution.run_subprocess
    calls = []
    program = ("import json,pathlib,sys; r=json.load(open(sys.argv[1])); "
        "p=pathlib.Path(r['execution_root'])/'runs'/r['unit_id']/'attempt1'; p.mkdir(parents=True); "
        "(p/'unit_result.json').write_text(json.dumps({'status':'passed','unit_id':r['unit_id']}))")
    def child(self, argv, **kwargs):
        calls.append(argv)
        assert "_s1-worker" in argv
        return original(self, [sys.executable, "-c", program, argv[-1]], **kwargs)
    monkeypatch.setattr(CampaignExecution, "run_subprocess", child)
    first = phase4.run_unit(request, kind="s1")
    assert first["status"] == "awaiting_host_ack" and len(calls) == 1
    with pytest.raises(ExecutionBlocked, match="Host must"):
        phase4.run_unit(request, kind="s1")
    def acknowledge():
        state = phase4._json(root / "execution_state.json")
        transfer = state["pending_transfer"]
        manifest_path = root / transfer["manifest_path"]
        manifest = phase4._json(manifest_path)
        for row in manifest["files"]:
            target = durable / row["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / row["path"], target)
        phase4.phase3.host_ack(manifest_path, durable, root / transfer["ack_path"], transfer_seconds=0.01)
    acknowledge()
    second = phase4.run_unit(request, kind="s1")
    assert second["status"] == "passed" and len(calls) == 1
    acknowledge()  # The tiny derived completion receipt also reaches independent storage.
    assert phase4.run_unit({**request, "session_id": "next-allocation"}, kind="s1")["status"] == "passed"
    assert len(calls) == 1


@pytest.mark.parametrize("fail_at", [None, 4])
def test_s1_queue_automatically_runs_all_96_fixed_units(tmp_path, monkeypatch, fail_at):
    from ec_clean30 import workstation

    plan = write_json(tmp_path / "plan.json", campaign.build_fold_plan(0))
    monkeypatch.setattr(phase4.phase3, "verify_execution_source", lambda *args, **kwargs: {})
    monkeypatch.setattr(phase4, "verify_authority", lambda *args, **kwargs: None)
    monkeypatch.setattr(workstation, "ControllerTransport", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(workstation, "HostPull", lambda *args, **kwargs: SimpleNamespace(sync_pending=lambda root: None))
    monkeypatch.setattr(phase4, "_attach_session", lambda *args: {})
    closed = []
    monkeypatch.setattr(phase4, "_close_session", lambda *args: closed.append(True) or {"provider_stop_verified": True})
    calls = []
    def complete(transport, host, request, **kwargs):
        calls.append((dict(request), kwargs))
        if len(calls) == fail_at:
            raise RuntimeError("fixture terminal failure")
        return {"status": "passed", "unit_id": request["unit_id"]}
    monkeypatch.setattr(workstation, "complete_remote_unit", complete)
    request = {"execution_manifest": "manifest.json", "controller_root": str(tmp_path), "ssh_alias": "owned-vm",
               "durable_root": str(tmp_path / "host"), "host_state_root": str(tmp_path / "state"),
               "max_download_bytes": 1024, "deadline_unix": time.time()+3600, "local_plan": str(plan),
               "session_id": "allocation2", "remote_request_template": {"execution_root": "/vm/runs/phase4", "session_id": "allocation2",
                   "deadline_unix": time.time()+3600}, "remote_runner": "/vm/phase4.py",
               "family_forecasts": dict.fromkeys(campaign.FAMILIES, 20), "family_timeouts": dict.fromkeys(campaign.FAMILIES, 15)}
    request["remote_request_template"]["deadline_unix"] = request["deadline_unix"]
    if fail_at:
        with pytest.raises(RuntimeError, match="fixture terminal failure"):
            phase4.run_s1_queue(request)
        assert len(calls) == fail_at and closed == [True]
        return
    result = phase4.run_s1_queue(request)
    assert result["completed"] == [unit.unit_id for unit in phase4.s1_units()]
    assert len(calls) == 96
    assert all(kwargs["action"] == "s1-unit" and kwargs["completed_relative"] == f"runs/{row['unit_id']}/complete.json"
               for row, kwargs in calls)
    assert all("/allocation2/" in kwargs["remote_request"] for _, kwargs in calls)
    assert closed == [True]


@pytest.mark.parametrize("case", ["valid", "foreign_session", "missing_provider_stop", "different_deadline"])
def test_attach_requires_actual_owned_provider_protected_session(tmp_path, case):
    expected = {"project": "project", "zone": "us-central1-a", "vm_name": "vm", "instance_id": "123"}
    request = {"expected_identity": expected, "ssh_alias": "owned", "session_id": "allocation", "allocation_started_unix": 1000,
               "deadline_unix": 8020, "max_total_seconds": 7200}
    (tmp_path / "config.env").write_text("PROJECT_ID=project\nZONE=us-central1-a\nVM_NAME=vm\nSELECTED_INSTANCE_ID=123\n"
                                       "MACHINE_TYPE=g2-standard-8\nSSH_HOST_ALIAS=owned\n")
    record = {**expected, "active": True, "session_id": "allocation", "session_start": datetime.fromtimestamp(1000, timezone.utc).isoformat(),
              "termination_ts": datetime.fromtimestamp(8020, timezone.utc).isoformat(), "max_run_duration_s": 7020}
    if case == "foreign_session": record["session_id"] = "another"
    if case == "different_deadline": request["deadline_unix"] += 1
    write_json(tmp_path / "state/current_session.json", record)
    calls = []
    def controller(name, *args, **kwargs):
        calls.append(name)
        state = "TERMINATED" if "stop" in calls else "RUNNING"
        protection = "" if case == "missing_provider_stop" else "Google-side auto-STOP: VERIFIED\n"
        return f"VM: vm project=project zone=us-central1-a\nState: {state}\n{protection}"
    transport = SimpleNamespace(controller_root=tmp_path, controller=controller)
    if case == "valid":
        phase4._attach_session(request, transport)
        assert phase4._close_session(request, transport)["provider_stop_verified"]
        assert calls == ["status", "stop", "status"]
    else:
        with pytest.raises((phase4.Phase4Error, RuntimeError)):
            phase4._attach_session(request, transport)
        assert "stop" not in calls
        if case == "foreign_session":
            with pytest.raises(phase4.Phase4Error, match="Another allocation"):
                phase4._close_session(request, transport)
            assert "stop" not in calls


def test_s1_worker_real_cpu_terminal_replay_and_persistence(tmp_path, tmp_path_factory, monkeypatch):
    from benchmarking.pmm_execution import CampaignExecution, ExecutionPolicy
    from test_ec_clean30_phase3 import transfer_view

    fixture = transfer_view.__wrapped__(tmp_path_factory)
    view = next(fixture)
    try:
        unit = next(unit for unit in phase4.s1_units() if unit.family == "F2")
        root, durable = view.default_runs_dir / "phase4_cpu", tmp_path / "independent_copy"
        now = time.time()
        policy = ExecutionPolicy(deadline_unix=now+4000, max_total_seconds=4000, allocation_started_unix=now)
        source = {"freeze1_sha256": "1"*64, "freeze1b_sha256": "2"*64, "amendment_chain": ["3"*64],
                  "git_head": "4"*40, "source_tree_sha256": "5"*64, "execution_manifest_sha256": "6"*64}
        plan = write_json(tmp_path / "plan.json", campaign.build_fold_plan(0))
        request = {"execution_root": str(root), "durable_root": str(durable), "session_id": "cpu-fixture",
            "persistence_mode": "host_pull", "unit_id": unit.unit_id, "plan": str(plan), "forecast_seconds": 120,
            "unit_timeout_seconds": 120, "python": sys.executable, "device": "cpu", "g3_sha256": "7"*64,
            "cap_prefix": ["env", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_THREADING_LAYER=GNU"],
            "memory_limit_bytes": 2*1024**3}
        monkeypatch.setattr(phase4, "_runtime", lambda request, resource: (source, view, policy, {}, {}))
        def worker(self, argv, **kwargs):
            phase4.s1_worker(request)
            return subprocess.CompletedProcess(argv, 0)
        monkeypatch.setattr(CampaignExecution, "run_subprocess", worker)
        assert phase4.run_unit(request, kind="s1")["status"] == "awaiting_host_ack"
        state = phase4._json(root / "execution_state.json")
        transfer = state["pending_transfer"]
        manifest = root / transfer["manifest_path"]
        for row in phase4._json(manifest)["files"]:
            target = durable / row["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / row["path"], target)
        phase4.phase3.host_ack(manifest, durable, root / transfer["ack_path"], transfer_seconds=0.01)
        completed = phase4.run_unit(request, kind="s1")
        assert completed["status"] == "passed"
        run = Path(completed["runs_dir"]) / unit.unit_id
        prediction = phase4._json(run / "ec_prediction_receipt.json")
        assert prediction["epochs"]["completed"] == 30
        verification = phase4._json(run.parent / "verify_unit_receipt.json")
        assert verification["status"] == "passed" and verification["n_units"] == 1
        assert completed["replay_sha256"] == phase4._sha(run / "ec_prediction_replay/ec_prediction_replay_receipt.json")
        assert (durable / run.relative_to(root) / "terminal_model_checkpoint.pt").is_file()
    finally:
        next(fixture, None)


def test_embedding_queue_is_exactly_base_and_seed42_masked(monkeypatch):
    captured = {}
    def queue(request, units, **kwargs):
        captured.update(units=units, **kwargs)
        return {"status": "passed"}
    monkeypatch.setattr(phase4, "_run_remote_queue", queue)
    request = {"embedding_forecasts": {"base": 100, "masked": 200}, "embedding_timeouts": {"base": 80, "masked": 180}}
    assert phase4.run_embedding_queue(request)["status"] == "passed"
    assert captured["units"] == [{"unit_id": "ecc30_clean_embed_base", "forecast_seconds": 100, "unit_timeout_seconds": 80},
                                 {"unit_id": "ecc30_clean_embed_masked", "forecast_seconds": 200, "unit_timeout_seconds": 180}]
    assert captured["action"] == "clean-embeddings"


@pytest.mark.parametrize("case", ["valid", "train_as_map", "map_as_train", "swapped_threads", "extra_command", "missing_swap", "memory_metadata"])
def test_clean_caps_require_distinct_map_and_training_limits(case):
    request = {"cap_prefix": phase4.clean_cap(phase4.CLEAN_TRAIN_MEMORY_BYTES),
               "distance_cap_prefix": phase4.clean_cap(phase4.CLEAN_MAP_MEMORY_BYTES),
               "threads": 2, "memory_limit_bytes": phase4.CLEAN_TRAIN_MEMORY_BYTES}
    if case == "train_as_map": request["distance_cap_prefix"] = request["cap_prefix"]
    elif case == "map_as_train": request["cap_prefix"] = request["distance_cap_prefix"]
    elif case == "swapped_threads": request["threads"] = 1
    elif case == "extra_command": request["cap_prefix"] += ["echo"]
    elif case == "missing_swap": request["distance_cap_prefix"].remove("MemorySwapMax=0")
    elif case == "memory_metadata": request["memory_limit_bytes"] = phase4.CLEAN_MAP_MEMORY_BYTES
    if case == "valid":
        phase4._clean_caps(request)
    else:
        with pytest.raises(phase4.Phase4Error, match="separate canonical"):
            phase4._clean_caps(request)


def test_clean_map_verified_before_capped_training_and_launcher_preserves_arguments(tmp_path, monkeypatch):
    work, attempt = tmp_path / "work", tmp_path / "attempt"
    work.mkdir(); attempt.mkdir()
    request = {"cap_prefix": phase4.clean_cap(phase4.CLEAN_TRAIN_MEMORY_BYTES),
               "distance_cap_prefix": phase4.clean_cap(phase4.CLEAN_MAP_MEMORY_BYTES),
               "threads": 2, "memory_limit_bytes": phase4.CLEAN_TRAIN_MEMORY_BYTES,
               "clean_python": sys.executable, "clean_clone": str(tmp_path / "clone"),
               "training_script": str(tmp_path / "train.py"), "distance_helper": str(tmp_path / "map.py")}
    write_json(work / phase4.clean.PREPARE_MANIFEST, {"training_name": "fold0", "training_table": {"sha256": "1"*64},
               "embedding_farm": {"base": {"farm_sha256": "2"*64}}, "input_dim": 1280})
    calls = []
    command = [sys.executable, "map.py", "--training_data", "fold0"]
    def fit(**kwargs):
        assert kwargs["epochs"] == 2500 and kwargs["threads"] == 2
        if kwargs.get("dry_run"):
            calls.append("plan")
            return {"steps": [{"step": "distance_map", "command": command, "hash_seed": 0},
                              {"step": "train", "command": ["not-run"], "hash_seed": 42}]}
        calls.append("train")
        assert kwargs["clean_python"] == attempt / "capped_clean_python"
        assert "MemoryMax=3221225472" in kwargs["clean_python"].read_text()
        return {"recipe": phase4.clean.PUBLISHED_RECIPE}
    def run(argv, *, cwd, env, log_path):
        calls.append("map")
        assert argv == request["distance_cap_prefix"] + command
        assert env["PYTHONHASHSEED"] == "0" and env["CUDA_VISIBLE_DEVICES"] == ""
        log_path.write_text("synthetic map complete\n")
        return 0
    def verify(*args, **kwargs):
        calls.append("verify")
        assert kwargs["helper_sha256"] == phase4.clean.DISTANCE_MAP_HELPER_SHA256
        return {"sha256": "3"*64}
    monkeypatch.setattr(phase4.clean, "run_fit", fit)
    monkeypatch.setattr(phase4.clean, "_run", run)
    monkeypatch.setattr(phase4.clean, "verify_distance_map", verify)
    result, launcher = phase4._clean_fit(request, work, attempt, [tmp_path / "forbidden"])
    assert calls == ["plan", "map", "verify", "train"]
    assert result["recipe"]["epochs"] == 2500
    receipt = phase4._json(attempt / "distance_map_execution.json")
    assert receipt["steps"][0]["log_sha256"] == phase4._sha(attempt / "distance_map.log")
    arguments = ["space value", "literal$(value)", "quote'value", "--flag"]
    program = "import json,os,sys; print(json.dumps({'argv':sys.argv[1:],'marker':os.environ['P4_MARKER']}))"
    output = subprocess.check_output([str(launcher), "-c", program, *arguments], text=True,
                                     env={**os.environ, "P4_MARKER": "unchanged value"})
    assert json.loads(output) == {"argv": arguments, "marker": "unchanged value"}


@pytest.mark.parametrize("tamper", ["fit", "launcher", "map"])
def test_clean_reuse_checks_persisted_companions_after_hot_cleanup(tmp_path, monkeypatch, tamper):
    unit = phase4.clean_units()[0]
    export = tmp_path / "validation.npz"
    export.write_bytes(b"synthetic export")
    source = {"execution_manifest_sha256": "1"*64}
    paths = {"fit": write_json(tmp_path / "work" / phase4.clean.FIT_MANIFEST, {"recipe": {"epochs": 2500}}),
             "launcher": tmp_path / "capped_clean_python",
             "map": write_json(tmp_path / "distance_map_execution.json", {"status": "passed"})}
    paths["launcher"].write_text("#!/bin/sh\nexec pinned-python \"$@\"\n")
    result = {"status": "passed", "unit_id": unit["unit_id"], "source": source, "export": str(export),
              "export_sha256": phase4._sha(export), "fit_manifest_sha256": phase4._sha(paths["fit"]),
              "clean_python_launcher_sha256": phase4._sha(paths["launcher"]),
              "distance_map_execution_sha256": phase4._sha(paths["map"])}
    monkeypatch.setattr(phase4.clean, "read_distance_export", lambda path: SimpleNamespace(receipt={
        "seed": 42, "regime": unit["regime"], "inner_fold": unit["inner_fold"]}))
    for relative in ("farms", "work/data/esm_data", "work/data/distance_map"):
        (tmp_path / relative).mkdir(parents=True, exist_ok=True)
        shutil.rmtree(tmp_path / relative)
    phase4._verify_result({"unit_id": unit["unit_id"]}, result, "clean", source, None)
    paths[tamper].write_text("changed")
    with pytest.raises(phase4.Phase4Error, match="fit, launcher or map evidence changed"):
        phase4._verify_result({"unit_id": unit["unit_id"]}, result, "clean", source, None)
