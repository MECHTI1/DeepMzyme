"""Offline operation checks; all runtime/CLI services are synthetic."""
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import types

import pytest

OPS = Path(__file__).parent
sys.path.insert(0, str(OPS))
import ring_common as common
import ring_control as control
import ring_host as host
import ring_finalization as finalization
import ring_archive_verify as archive_verifier
spec = importlib.util.spec_from_file_location("ring_proxy_for_tests", OPS/"refresh_owned_proxy.py")
proxy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proxy)


def handoff():
    return dict(profile=common.BUDGET_PROFILE, total_cap_seconds=36000., training_cap_seconds=34200., main_cap_seconds=27000.,
        prior_intervals=[dict(started_epoch=1789440733.8320558, observed_epoch=1789444166.060942, ended_epoch=1789444561.801127),
                         dict(started_epoch=1789456819.0405653, observed_epoch=1789472254.5176592, ended_epoch=1789472254.5176592)],
        prior_allocated_seconds=common.PRIOR_SECONDS, prior_main_seconds=common.PRIOR_MAIN_SECONDS,
        prior_retry_seconds=common.PRIOR_RETRY_SECONDS, source_artifact_hashes={})


def test_exact_budget_carry_forward_and_cutoffs():
    proof = common.validate_handoff(handoff())
    cfg = dict(allocation_started_epoch=1800000000., prior_allocated_seconds=proof["prior_allocated_seconds"])
    assert common.allocation_deadline(cfg)-cfg["allocation_started_epoch"] == pytest.approx(16736.55383491516)
    assert common.allocation_deadline(cfg, training=True)-cfg["allocation_started_epoch"] == pytest.approx(14936.55383491516)
    assert common.allocation_deadline(cfg, stop_margin=300)-cfg["allocation_started_epoch"] == pytest.approx(16436.55383491516)


@pytest.mark.parametrize("key,value", [("prior_allocated_seconds", 0), ("prior_main_seconds", 0), ("prior_retry_seconds", 0),
                                       ("total_cap_seconds", 72000), ("training_cap_seconds", 36000), ("main_cap_seconds", 36000)])
def test_budget_reset_rejected(key, value):
    value_dict = handoff()
    value_dict[key] = value
    with pytest.raises(ValueError):
        common.validate_handoff(value_dict)


def test_prior_intervals_must_remain_closed():
    proof = handoff()
    del proof["prior_intervals"][-1]["ended_epoch"]
    with pytest.raises(ValueError):
        common.validate_handoff(proof)


class Session(types.SimpleNamespace):
    def model_copy(self, update):
        return Session(**(vars(self)|update))


class Store:
    def __init__(self, name="ring-new", endpoint="owned", during_lock=None):
        self.records = {name: Session(name=name, endpoint=endpoint, token="old-synthetic", url="https://old.example",
                                      kernel_id="old-kernel", running="exec")}
        self.records["unrelated"] = Session(name="unrelated", endpoint="other", token="other-synthetic")
        self.during_lock, self.writes = during_lock, 0
    def get(self, name):
        return self.records.get(name)
    @contextmanager
    def _lock_exclusive(self):
        if self.during_lock:
            self.during_lock(self.records)
        yield None
    def _load_raw(self, stream):
        return self.records.copy()
    def _save_raw(self, stream, records):
        self.records = records
        self.writes += 1


def assignment(endpoint="owned", ttl=3600, url="https://proxy.example"):
    return types.SimpleNamespace(endpoint=endpoint, accelerator=types.SimpleNamespace(value="G4"),
        variant=types.SimpleNamespace(name="GPU"), runtime_proxy_info=types.SimpleNamespace(token="issued-synthetic", url=url, token_expires_in_seconds=ttl))


def test_new_named_proxy_preserves_concurrent_kernel_and_unrelated_mapping():
    def update(records):
        records["ring-new"].kernel_id = "latest-kernel"
    store = Store(during_lock=update)
    other = vars(store.records["unrelated"]).copy()
    result = proxy.synchronize(store, types.SimpleNamespace(list_assignments=lambda: [assignment()]),
                               dict(session_name="ring-new", endpoint="owned"), now=lambda: 100.)
    assert store.records["ring-new"].kernel_id == "latest-kernel"
    assert vars(store.records["unrelated"]) == other
    assert result["session_name"] == "ring-new" and store.writes == 1
    assert "issued-synthetic" not in json.dumps(result) and "proxy.example" not in json.dumps(result)


@pytest.mark.parametrize("rows", [[], [assignment(), assignment()], [assignment(ttl=20)], [assignment(url="https://")]])
def test_proxy_refuses_unusable_assignment_without_mutation(rows):
    store = Store()
    with pytest.raises(ValueError):
        proxy.synchronize(store, types.SimpleNamespace(list_assignments=lambda: rows), dict(session_name="ring-new", endpoint="owned"), now=lambda: 100.)
    assert store.writes == 0


def test_host_cli_rejects_other_or_implicit_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(host, "OPS", tmp_path)
    common.save(tmp_path/"session_config.json", dict(profile=common.PROFILE, session_name="ring-new"))
    monkeypatch.setattr(host.subprocess, "run", lambda *a, **k: pytest.fail("CLI must not execute"))
    for args in [("stop",), ("exec", "-s", "unrelated")]:
        with pytest.raises(ValueError, match="exact owned name"):
            host.cli(*args)


def stop_fixture(tmp_path, monkeypatch, mapping_endpoint="owned"):
    cfg = dict(profile=common.PROFILE, session_name="ring-new", allocation_started_epoch=1800000000.,
               local_budget_root=str(tmp_path/"budget"))
    common.save(tmp_path/"session_config.json", cfg)
    common.save(tmp_path/"budget_handoff.json", handoff())
    common.save(tmp_path/"budget/allocation_ledger.json", dict(intervals=copy.deepcopy(handoff()["prior_intervals"])))
    owned = dict(session_name="ring-new", endpoint="owned", started_epoch=cfg["allocation_started_epoch"])
    common.save(tmp_path/"owned_session.json", owned)
    monkeypatch.setattr(host, "OPS", tmp_path)
    monkeypatch.setattr(host, "ownership_from_history", lambda c: owned)
    state = types.ModuleType("colab_cli.state")
    state.SessionState = lambda **kw: types.SimpleNamespace(**kw)
    state.StateStore = lambda: types.SimpleNamespace(get=lambda name: types.SimpleNamespace(endpoint=mapping_endpoint))
    monkeypatch.setitem(sys.modules, "colab_cli.state", state)
    monkeypatch.setattr(host.time, "time", lambda: cfg["allocation_started_epoch"]+123.)
    return cfg


def test_named_stop_counts_third_interval_and_preserves_previous_two(tmp_path, monkeypatch):
    cfg = stop_fixture(tmp_path, monkeypatch)
    previous = copy.deepcopy(handoff()["prior_intervals"])
    calls = []
    def fake_cli(*args, **kwargs):
        calls.append(args)
        return "Stopped" if args[0] == "stop" else "No active sessions found on server."
    monkeypatch.setattr(host, "cli", fake_cli)
    result = host.stop()
    assert calls == [("stop", "-s", "ring-new"), ("sessions",)]
    ledger = common.read(tmp_path/"budget/allocation_ledger.json")
    assert ledger["intervals"][:2] == previous and len(ledger["intervals"]) == 3
    assert result["cumulative_allocated_seconds"] == pytest.approx(common.PRIOR_SECONDS+123.)
    assert host.stop() == result  # Immutable existing closeout, no new stop.
    assert len(calls) == 2


def test_owned_stop_refuses_replacement_mapping(tmp_path, monkeypatch):
    stop_fixture(tmp_path, monkeypatch, mapping_endpoint="unrelated")
    monkeypatch.setattr(host, "cli", lambda *a, **k: pytest.fail("Must not call stop on replacement mapping"))
    with pytest.raises(ValueError, match="another endpoint"):
        host.stop()
    assert not (tmp_path/"session_stopped.json").exists()


def test_absence_required_before_closed_receipt(tmp_path, monkeypatch):
    stop_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(host, "cli", lambda *a, **k: "owned still listed")
    with pytest.raises(ValueError, match="still present"):
        host.stop()
    assert not (tmp_path/"session_stopped.json").exists()
    assert len(common.read(tmp_path/"budget/allocation_ledger.json")["intervals"]) == 2


def test_archive_uses_existing_chunk_contract_and_safe_metadata_allowlist(tmp_path):
    output, budget = tmp_path/"campaign", tmp_path/"budget"
    common.save(output/"campaign_manifest.json", dict(profile=common.PROFILE))
    common.save(output/"campaign_attempt_ledger.json", [])
    common.save(output/"last_cli_error.json", dict(secret="must not be archived"))
    common.save(output/"runs/run_a/run_config.json", dict(epochs=50))
    common.save(budget/"budget_handoff.json", handoff())
    common.save(budget/"allocation_ledger.json", dict(intervals=handoff()["prior_intervals"]))
    common.save(budget/"bootstrap_budget_usage.json", dict(normal_elapsed_seconds=400.))
    latest = dict(attempt_id="attempt_001", status="completed", run_dir=str(output/"runs/run_a"))
    descriptor = control.package_archive(output, budget, latest, exports=tmp_path/"exports")
    local = tmp_path/"extracted"
    (local/"archives").mkdir(parents=True)
    for part in descriptor["parts"]:
        shutil.copyfile(part["path"], local/"archives"/Path(part["path"]).name)
    source = host.OLDOPS/"verify_pilot_archive.py"
    spec = importlib.util.spec_from_file_location("ring_archive_verifier_fixture", source)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    result = verifier.verify_and_extract(descriptor, local)
    assert result["local_sha256_verified"] and (local/"budget/budget_handoff.json").is_file()
    assert not (local/"last_cli_error.json").exists()
    assert control.package_archive(output, budget, latest, exports=tmp_path/"exports") == descriptor


def test_actual_auth_failure_checks_whoami_and_blocks_later_workload(tmp_path, monkeypatch):
    monkeypatch.setattr(host, "OPS", tmp_path)
    common.save(tmp_path/"session_config.json", dict(profile=common.PROFILE, session_name="ring-new"))
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        if command[1] == "whoami":
            return types.SimpleNamespace(returncode=0, stdout="Provider: oauth2", stderr="")
        return types.SimpleNamespace(returncode=1, stdout="", stderr="HTTP 403 Forbidden")
    monkeypatch.setattr(host.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="403"):
        host.cli("status", "-s", "ring-new")
    assert calls == [["colab", "status", "-s", "ring-new"], ["colab", "whoami"]]
    assert common.read(tmp_path/"auth_stop_required.json")["stop_required"]
    with pytest.raises(ValueError, match="only inspection and owned teardown"):
        host.cli("exec", "-s", "ring-new", "--timeout", "180")
    assert len(calls) == 2


def test_allocation_success_starts_watchdog_before_ownership_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(host, "OPS", tmp_path)
    cfg = dict(profile=common.PROFILE, session_name="ring-no-existing-history-fixture", local_budget_root=str(tmp_path/"budget"))
    common.save(tmp_path/"session_config.json", cfg)
    monkeypatch.setattr(host, "validate_prepared_inputs", lambda cfg: None)
    events = []
    monkeypatch.setattr(host, "cli", lambda *a, **k: events.append("created") or "Created")
    monkeypatch.setattr(host, "start_watchdog", lambda: events.append("watchdog"))
    def failed_proof(cfg):
        events.append("ownership")
        raise ValueError("synthetic ownership verification failure")
    monkeypatch.setattr(host, "ownership_from_history", failed_proof)
    with pytest.raises(ValueError, match="ownership verification"):
        host.allocate()
    assert events == ["created", "watchdog", "ownership"]
    assert (tmp_path/"allocation_requested.json").is_file()
    with pytest.raises(ValueError, match="one allocation"):
        host.allocate()


def checkpoint_fixture(tmp_path):
    directory = tmp_path/"runs/full"
    metric = "val_metal_balanced_acc"
    config = dict(epochs=50, task="metal", metal_label_scheme="merge_fe_class_viii", omit_node_features=())
    history = [dict(epoch=epoch, **{metric: 0.75 if epoch == 35 else 0.65}) for epoch in range(1, 51)]
    common.save(directory/"run_config.json", dict(config=config, history=history))
    normalization = dict(means={"field": [1.0]}, stds={"field": [2.0]}, clamp_value=5.)
    common.save(directory/"run_metadata.json", dict(selection_metric=metric, selected_checkpoint_epoch=35, normalization_stats=normalization))
    common.save(directory/"dataset_summary.json", dict(split="fixture"))
    (directory/"epoch_metrics.csv").write_text("epoch\n"+"\n".join(str(i) for i in range(1, 51)))
    (directory/"best_model_checkpoint.pt").write_bytes(b"synthetic best")
    (directory/"last_model_checkpoint.pt").write_bytes(b"synthetic last")
    checkpoint = dict(epoch=35, selection_metric=metric, selection_metric_value=0.75, config=config, normalization_stats=normalization)
    row = dict(id="full", block="GVP", run_dir=str(directory), selected_epoch=35, balanced_accuracy=0.75,
               normalization_stats_sha256=hashlib.sha256(json.dumps(normalization, sort_keys=True, default=str).encode()).hexdigest())
    return row, checkpoint


@pytest.mark.parametrize("field,value", [("epoch", 50), ("selection_metric", "train_loss"), ("selection_metric_value", 0.74), ("config", {}), ("normalization_stats", {})])
def test_final_checkpoint_rejects_saved_state_mismatch(tmp_path, field, value):
    row, checkpoint = checkpoint_fixture(tmp_path)
    checkpoint[field] = value
    with pytest.raises(ValueError, match="checkpoint"):
        finalization.checkpoint_binding(row, load_checkpoint=lambda path: checkpoint)


def test_final_checkpoint_uses_safe_cpu_load(tmp_path):
    import torch
    row, checkpoint = checkpoint_fixture(tmp_path)
    torch.save(checkpoint, Path(row["run_dir"])/"best_model_checkpoint.pt")
    proof = finalization.checkpoint_binding(row)
    assert proof["selected_epoch"] == 35 and proof["balanced_accuracy"] == 0.75
    assert len(proof["files"]) == 6


@pytest.mark.parametrize("active", [False, True])
def test_final_capture_refuses_active_or_nonterminal_campaign(tmp_path, active):
    import os
    output = tmp_path/"campaign"
    common.save(output/"campaign_state.json", dict(status="completed" if active else "awaiting_archive_transfer"))
    if active:
        common.save(output/"host_step_process.json", dict(pid=os.getpid()))
    with pytest.raises(ValueError, match="active|terminal"):
        finalization.capture({}, tmp_path, output, tmp_path/"budget", exports=tmp_path/"exports")
    assert not (output/"finalization_stage").exists()


def test_terminal_archive_is_separate_and_binds_checkpoints(tmp_path):
    output, budget, root, operators = (tmp_path/name for name in ("campaign", "budget", "source", "operators"))
    row, checkpoint = checkpoint_fixture(output)
    common.save(output/"campaign_manifest.json", dict(profile=common.PROFILE, source_files={}))
    state = dict(status="budget_stopped", declined_block="LATE")
    common.save(output/"campaign_state.json", state)
    common.save(output/"campaign_attempt_ledger.json", [])
    common.save(budget/"budget_handoff.json", handoff())
    common.save(budget/"allocation_ledger.json", dict(intervals=handoff()["prior_intervals"]))
    common.save(budget/"bootstrap_budget_usage.json", dict(normal_elapsed_seconds=100.))
    summary = dict(state=state, completed_smokes=4, completed_full_runs=8, coverage={"GVP": {"complete": True}})
    def summarize(path):
        common.save(path/"ring_validation_results.json", summary)
        return summary
    pilot = types.SimpleNamespace(verify_manifest=lambda r, o: common.read(o/"campaign_manifest.json"), summarize=summarize,
        rows=lambda o, m: [row], base=types.SimpleNamespace(_require_transfer=lambda *a: None, allocation_elapsed=lambda *a: 20000.))
    cfg = dict(allocation_started_epoch=1800000000., source_sha256="synthetic source", operator_files={})
    descriptor = finalization.capture(cfg, root, output, budget, exports=tmp_path/"exports", operator_root=operators,
                                       pilot=pilot, load_checkpoint=lambda path: checkpoint)
    assert descriptor["attempt_id"] == "finalization"
    local = tmp_path/"local/finalization"
    (local/"archives").mkdir(parents=True)
    for part in descriptor["parts"]:
        shutil.copyfile(part["path"], local/"archives"/Path(part["path"]).name)
    receipt = archive_verifier.verify_and_extract(descriptor, local)
    assert receipt["local_sha256_verified"]
    captured = common.read(local/"final_capture_receipt.json")
    assert captured["status"] == "terminal_metadata_verified_gpu_not_stopped"
    assert captured["terminal_state"] == state and captured["runs"][0]["selected_epoch"] == 35
    assert not (local/"completed_runs/full/best_model_checkpoint.pt").exists()
    assert (local/"completed_runs/full/run_metadata.json").is_file()
    assert not (tmp_path/"local/campaign_manifest.json").exists()
    assert finalization.capture(cfg, root, output, budget, exports=tmp_path/"exports", operator_root=operators,
                                pilot=pilot, load_checkpoint=lambda path: checkpoint) == descriptor


def test_post_stop_packaging_refuses_before_actual_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(host, "OPS", tmp_path)
    common.save(tmp_path/"session_config.json", dict(profile=common.PROFILE, session_name="ring-new"))
    with pytest.raises(ValueError, match="Actual verified owned stop"):
        host.closeout()


def closeout_fixture(tmp_path, monkeypatch):
    import tarfile
    cfg = stop_fixture(tmp_path, monkeypatch)
    local = tmp_path/"local"
    cfg["local_output_dir"] = str(local)
    common.save(tmp_path/"session_config.json", cfg)
    common.save(tmp_path/"session_preallocation_config.json", cfg)
    monkeypatch.setattr(host, "OPERATORS", ())
    monkeypatch.setattr(host, "cli", lambda *args, **kwargs: "Stopped" if args[0] == "stop" else "No active sessions found on server.")
    host.stop()
    final = local/"finalization"
    for name, value in (("campaign_manifest.json", {"profile": common.PROFILE}), ("campaign_state.json", {"status": "completed"}),
                        ("ring_validation_results.json", {"completed_full_runs": 16}), ("final_capture_receipt.json", {"status": "terminal_metadata_verified_gpu_not_stopped"})):
        common.save(final/name, value)
    archive = final/"archives/finalization.tar.gz"
    archive.parent.mkdir(parents=True)
    with tarfile.open(archive, "w:gz") as stream:
        for name in ("campaign_manifest.json", "campaign_state.json", "ring_validation_results.json", "final_capture_receipt.json"):
            stream.add(final/name, arcname=name)
    receipt = dict(attempt_id="finalization", archive=str(archive), archive_sha256=common.sha(archive), bytes=archive.stat().st_size,
                   manifest_sha256=common.sha(final/"campaign_manifest.json"), drive_verified=True, local_sha256_verified=True, drive_file_id="synthetic-drive-id")
    for name in ("transfer_receipts/finalization.json", "last_download_receipt.json", "archives/finalization_archive.json"):
        common.save(final/name, receipt)
    return local


@pytest.mark.parametrize("changed", ["endpoint", "started_epoch", "capture_receipt"])
def test_post_stop_rejects_changed_receipt_binding(tmp_path, monkeypatch, changed):
    local = closeout_fixture(tmp_path, monkeypatch)
    if changed == "capture_receipt":
        common.save(local/"finalization/final_capture_receipt.json", {"status": "synthetic changed receipt"})
    else:
        receipt = common.read(tmp_path/"session_stopped.json")
        receipt[changed] = "another-endpoint" if changed == "endpoint" else receipt[changed]-1
        common.save(tmp_path/"session_stopped.json", receipt)
    with pytest.raises(ValueError, match="owned endpoint|verified archive"):
        host.closeout()
    assert not (local/"host_closeout_allocation3").exists()


def test_post_stop_package_preserves_actual_total(tmp_path, monkeypatch):
    local = closeout_fixture(tmp_path, monkeypatch)
    result = host.closeout()
    assert result["cumulative_allocated_seconds"] == pytest.approx(common.PRIOR_SECONDS+123.)
    assert Path(result["archive"]).is_file()
    assert common.sha(Path(result["archive"])) == result["archive_sha256"]
    assert (local/"host_closeout_allocation3/final_capture_receipt.json").is_file()
