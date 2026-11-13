"""CPU-only execution seams for F6 cache preparation, attachment and admission."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ec_clean30 import phase3  # noqa: E402
import ec_clean30  # noqa: E402
from test_ec_clean30_phase3 import transfer_view  # noqa: E402,F401


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def source_identity():
    return {"format": phase3.EXECUTION_V5_FORMAT, "scientific_identity_sha256": digest("science"),
            "fit_chain": [digest("one"), digest("two")], "git_head": "a" * 40}


def test_unpack_creates_empty_writable_attachment_outside_manifest(transfer_view, tmp_path):
    archive = tmp_path / "view.tar"
    phase3.pack_view(transfer_view.root, archive, smoke=True)
    destination = tmp_path / "relocated" / phase3.CAMPAIGN_ID / "smoke_views" / "tiny"
    phase3.unpack_view(archive, destination, expected_sha256=phase3._sha(archive), smoke=True)
    attached = destination / "whole_protein_pretrain"
    assert attached.is_dir() and not any(attached.iterdir())
    assert stat.S_IMODE(attached.stat().st_mode) & stat.S_IWUSR
    manifest = json.loads((destination / phase3.views.VIEW_MANIFEST_NAME).read_text())
    assert not any(row["path"].startswith("whole_protein_pretrain/") for row in manifest["files"])


def test_bound_pretrain_input_rejects_unregistered_bytes(tmp_path, monkeypatch):
    source = {"amendment_path": "amendment.json"}
    expected, candidate = tmp_path / "bound.tsv", tmp_path / "relocated.tsv"
    expected.write_text("registered corpus\n")
    candidate.write_bytes(expected.read_bytes())
    (tmp_path / "amendment.json").write_text(json.dumps({"inputs": {"f6_corpus": {"tables": {"0": {
        "path": "bound.tsv", "sha256": phase3._sha(expected)}}}}}))
    monkeypatch.setattr(phase3, "REPO_ROOT", tmp_path)
    role = "inputs.f6_corpus.tables.0"
    assert phase3._bound_pretrain_file(source, role, candidate)["path"] == "bound.tsv"
    candidate.write_text("foreign corpus\n")
    with pytest.raises(ValueError, match="authenticated"):
        phase3._bound_pretrain_file(source, role, candidate)


def test_sequence_map_and_cpu_sample_use_public_sampling_rule(tmp_path, monkeypatch):
    from ec_clean30 import pretrain_corpus

    fasta = tmp_path / "union.fasta"
    fasta.write_text(">A\nMKT\n>B\nMGG\n>C\nMKT\n")
    calls = []
    chosen = pretrain_corpus.sequence_key("MKT")
    monkeypatch.setattr(pretrain_corpus, "parity_sample_keys", lambda sequences: calls.append(sequences) or [chosen])
    result = phase3.pretrain_sequence_inputs(fasta, tmp_path / "prepared")
    assert calls == [{"A": "MKT", "B": "MGG", "C": "MKT"}]
    assert result["sample_keys"] == [chosen]
    assert Path(result["cpu_sample_fasta"]).read_text() == ">A\nMKT\n"
    assert Path(result["sequence_map"]).read_text().count("\n") == 4
    assert phase3._sha(Path(result["sequence_map"])) == result["sequence_map_sha256"]
    with pytest.raises(FileExistsError):
        phase3.pretrain_sequence_inputs(fasta, tmp_path / "prepared")


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_pretrain_embedding_command_pins_fp32_offline_weights_and_scrubs_git(tmp_path, monkeypatch, device):
    cache = tmp_path / "offline"
    cache.mkdir()
    weights = cache / "weights.pth"
    weights.write_bytes(b"frozen-model")
    fasta, mapping = tmp_path / "union.fasta", tmp_path / "map.tsv"
    fasta.write_text(">A\nMKT\n")
    mapping.write_text("accession\tsha256\tsequence\n")
    view = SimpleNamespace(forbidden_read_roots=[str(tmp_path / "forbidden")])
    monkeypatch.setattr(phase3, "_esmc_cached_weight", lambda path: weights)
    monkeypatch.setenv("GIT_DIR", "/foreign/.git")
    monkeypatch.setenv("INFRA_PROVIDER", "foreign")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    argv, env = phase3.pretrain_embedding_command(
        {"python": sys.executable, "hf_home": str(cache), "esmc_weights": str(weights)},
        {"frozen_esmc_weight_sha256": phase3._sha(weights)}, view, fasta=fasta, sequence_map=mapping,
        sequence_map_sha256=phase3._sha(mapping), output=tmp_path / "output", device=device)
    assert argv[-2:] == ["--forward-precision", "fp32"]
    assert argv[argv.index("--device") + 1] == device
    assert "--max-residue-tokens" not in argv  # F6 uses whole official sequences.
    assert "runtime_precision.bootstrap_code" not in argv[2]
    guard, loader, producer = (argv[2].index(text) for text in (
        "install_guard_from_environment()", phase3.ESMC_FP32_LOAD, "runpy.run_module('embed_helpers.esmc_chain_means'"))
    assert guard < loader < producer  # the float32 loader is in place before the producer imports the SDK
    assert not any(name.startswith("GIT_") for name in env)
    assert "INFRA_PROVIDER" not in env
    assert env["NVIDIA_TF32_OVERRIDE"] == "0" and env["HF_HUB_OFFLINE"] == "1"
    if device == "cpu":
        assert env["CUDA_VISIBLE_DEVICES"] == ""


def _fake_esm_sdk(monkeypatch, *, loaded_dtype=None, foreign=False):
    """A stand-in SDK whose own loader truncates to bfloat16, as the real one does off CPU."""
    import types

    import torch

    calls = []

    class ESMC(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(4, 4)
            self.register_buffer("inv_freq", torch.rand(8))

        @classmethod
        def from_pretrained(cls, model_name, device=None):
            return load_local_model(model_name).to(torch.bfloat16)

    torch.manual_seed(0)
    reference = ESMC()

    def load_local_model(model_name, device=torch.device("cpu")):
        calls.append((model_name, str(device)))
        model = torch.nn.Linear(4, 4) if foreign else ESMC()
        if not foreign:
            model.load_state_dict(reference.state_dict())
        return model if loaded_dtype is None else model.to(loaded_dtype)

    for name, members in (("esm", {}), ("esm.models", {}), ("esm.models.esmc", {"ESMC": ESMC}),
                          ("esm.pretrained", {"load_local_model": load_local_model})):
        module = types.ModuleType(name)
        module.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, module)
    return ESMC, reference, calls


def test_fp32_loader_keeps_the_float32_values_that_the_sdk_loader_truncates(monkeypatch):
    import torch

    ESMC, reference, calls = _fake_esm_sdk(monkeypatch)
    expected = reference.state_dict()
    truncated = ESMC.from_pretrained("esmc_300m", device=torch.device("cpu")).to(torch.float32).state_dict()
    assert all(str(value.dtype) == "torch.float32" for value in truncated.values())
    assert not any(torch.equal(truncated[name], expected[name]) for name in expected)  # the cast back restores nothing
    calls.clear()
    exec(phase3.ESMC_FP32_LOAD, {})
    model = ESMC.from_pretrained("esmc_300m", device=torch.device("cpu"))
    assert calls == [("esmc_300m", "cpu")]
    loaded = model.state_dict()
    assert set(loaded) == set(expected) and all(torch.equal(loaded[name], expected[name]) for name in expected)
    # Another device is reached by a move after the CPU load, never by a load on that device.
    calls.clear()
    moved = ESMC.from_pretrained("esmc_300m", device=torch.device("meta"))
    assert calls == [("esmc_300m", "cpu")]
    assert {(value.device.type, str(value.dtype)) for value in moved.state_dict().values()} == {("meta", "torch.float32")}


@pytest.mark.parametrize("case", ["bfloat16", "foreign"])
def test_fp32_loader_refuses_a_load_that_is_not_float32_esmc(monkeypatch, case):
    import torch

    ESMC, _reference, _calls = _fake_esm_sdk(
        monkeypatch, loaded_dtype=torch.bfloat16 if case == "bfloat16" else None, foreign=case == "foreign")
    exec(phase3.ESMC_FP32_LOAD, {})
    with pytest.raises(SystemExit, match="float32 ESMC load"):
        ESMC.from_pretrained("esmc_300m", device=torch.device("cpu"))


def complete_record():
    return {"format": "deepmzyme_ec_clean30_pretrain_complete_v1", "status": "passed", "fold": 0,
            **phase3.source_identity(source_identity()), "sequence": ["gpu_fp32_cache", "attach", "cpu_sample_parity", "verify"],
            "parity": {"passed": True, "n_compared": 2}, "attachment": {"status": "passed",
                **phase3.source_identity(source_identity()), "parent_manifest_sha256": digest("gpu")},
            "caches": {"cuda": {"sha256": digest("gpu")}, "cpu": {"sha256": digest("cpu")}},
            "attachment_snapshot_sha256": digest("snapshot"),
            "host_transfer": {"manifest_sha256": digest("manifest"), "ack_sha256": digest("ack")},
            "gpu_precision": {"device": "cuda"}, "cpu_precision": {"device": "cpu"}}


@pytest.mark.parametrize("case", ["failed_parity", "empty_sample", "missing_ack", "wrong_order", "missing_attach",
                                  "wrong_fold", "unpersisted", "foreign_parent", "foreign_identity", "missing_snapshot"])
def test_pretrain_gate_refuses_incomplete_preparation(tmp_path, monkeypatch, case):
    monkeypatch.setattr(phase3, "require_record_precedence", lambda record, source: None)
    monkeypatch.setattr(phase3.runtime_precision, "validate_policy", lambda record, *, device: record)
    record = complete_record()
    if case == "failed_parity":
        record["parity"]["passed"] = False
    elif case == "empty_sample":
        record["parity"]["n_compared"] = 0
    elif case == "missing_ack":
        del record["host_transfer"]["ack_sha256"]
    elif case == "wrong_order":
        record["sequence"] = list(reversed(record["sequence"]))
    elif case == "missing_attach":
        record["attachment"] = {}
    elif case == "wrong_fold":
        record["fold"] = 1
    elif case == "foreign_parent":
        record["attachment"]["parent_manifest_sha256"] = digest("other")
    elif case == "foreign_identity":
        record["attachment"]["scientific_identity_sha256"] = digest("other")
    elif case == "missing_snapshot":
        del record["attachment_snapshot_sha256"]
    else:
        record["format"] = "deepmzyme_ec_clean30_pretrain_preparation_v1"
    path = tmp_path / "preparation.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        phase3.verify_pretrain_evidence(path, source_identity())


def test_pretrain_gate_checks_each_child_precision_policy(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(phase3, "require_record_precedence", lambda record, source: calls.append("precedence"))
    monkeypatch.setattr(phase3.runtime_precision, "validate_policy", lambda record, *, device: calls.append(device))
    path = tmp_path / "preparation.json"
    path.write_text(json.dumps(complete_record()))
    assert phase3.verify_pretrain_evidence(path, source_identity())["status"] == "passed"
    assert calls == ["precedence", "cuda", "cpu"]


def test_pretrain_worker_orders_cache_attachment_and_cpu_parity_and_preserves_attachment(tmp_path, monkeypatch):
    from ec_clean30 import pretrain_corpus

    root, attached = tmp_path / "execution", tmp_path / "view" / "whole_protein_pretrain"
    attached.mkdir(parents=True)
    view = SimpleNamespace(root=attached.parent, smoke=False, manifest_sha256=digest("view"))
    source, events = source_identity(), []
    monkeypatch.setattr(phase3, "_runtime", lambda request, **kwargs: (source, view, None))
    monkeypatch.setattr(phase3, "_bound_pretrain_file", lambda *args: {"path": "union.fasta", "sha256": digest("union")})
    monkeypatch.setattr(phase3, "_verify_container_caps", lambda *args, **kwargs: None)
    monkeypatch.setattr(phase3, "pretrain_sequence_inputs", lambda *args: {
        "sequence_map": "map.tsv", "sequence_map_sha256": digest("map"), "cpu_sample_fasta": "sample.fasta",
        "sample_keys": [digest("sequence")]})

    def command(request, source, view, *, output, device, **kwargs):
        return [device, str(output)], {}

    def subprocess_run(argv, **kwargs):
        device, output = argv[-2:]
        events.append(device)
        path = Path(output)
        path.mkdir(parents=True)
        (path / "manifest.json").write_text("{}")

    def attach(request):
        events.append("attach")
        (attached / "receipt.json").write_text("saved")
        return {"status": "passed"}

    monkeypatch.setattr(phase3, "pretrain_embedding_command", command)
    monkeypatch.setattr(phase3.subprocess, "run", subprocess_run)
    monkeypatch.setattr(phase3, "attach_pretrain", attach)
    monkeypatch.setattr(pretrain_corpus, "parent_precision_record", lambda *args, **kwargs: {})
    monkeypatch.setattr(phase3.runtime_precision, "read_policy_log", lambda path, *, device: {"device": device})
    monkeypatch.setattr(pretrain_corpus, "compare_sequence_mean_samples", lambda *args: events.append("parity") or {
        "passed": True, "n_compared": 1})
    monkeypatch.setattr(pretrain_corpus, "verify_whole_protein_pretrain", lambda *args, **kwargs: events.append("verify"))
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps({"container_caps_probe": {}}))
    request = {"execution_root": str(root), "union_fasta": "union.fasta", "vm_probe": str(probe),
               "python": sys.executable, "cpu_cap_prefix": ["cap"], "cap_prefix": ["cap"],
               "session_id": "pretrain-session", "pod_id": "synthetic-pod"}
    out = root / "runs/ecc30_pretrain_fold0_attempt1"
    request_path = root / "commands/ecc30_pretrain_fold0_attempt1.json"
    phase3._write(request_path, request)
    marker = phase3.prepare_pretrain_attachment(out, request=request, source=source, view=view, request_path=request_path)
    monkeypatch.setattr(phase3.views.View, "load", lambda *args: view)
    result = phase3.pretrain_worker(request)
    assert events == ["cuda", "attach", "cpu", "parity", "verify"]
    assert result["status"] == "passed"
    assert json.loads(marker.read_text())["status"] == "pending"
    assert not (out / "attachment_snapshot").exists()  # Only the terminated worker's parent snapshots it.
    proof = provider_termination_proof(tmp_path, marker)
    phase3.restore_pretrain_attachment(marker, worker_terminated=True, termination_proof=proof)
    assert (out / "attachment_snapshot/receipt.json").read_text() == "saved"


def test_later_fold_attachment_stops_before_reading_training_inputs(tmp_path, monkeypatch):
    source = {**source_identity(), "freeze1_sha256": digest("freeze"), "amendment_path": "amendment.json",
              "amendment_chain": source_identity()["fit_chain"], "amendment_chain_receipt": str(tmp_path / "receipt.json")}
    calls = []
    monkeypatch.setattr(phase3, "verify_execution_source", lambda *args, **kwargs: source)
    monkeypatch.setattr(phase3, "authenticated_chain", lambda *args, **kwargs: {})
    monkeypatch.setattr(phase3.views.View, "load", lambda *args: SimpleNamespace(fold=1, smoke=False))
    monkeypatch.setattr(ec_clean30, "futility", SimpleNamespace(verify_futility_decision=lambda *args, **kwargs:
        calls.append(kwargs) or {**phase3.source_identity(source), "decision": "stop_futility"}), raising=False)
    monkeypatch.setattr(phase3, "_bind_view", lambda *args: pytest.fail("Must stop before the fold view is opened"))
    with pytest.raises(ValueError, match="blocked by the futility"):
        phase3.attach_pretrain({"execution_manifest": "manifest.json", "view": "fold1",
            "fold0_step_chain": "steps.jsonl", "campaign_root": str(tmp_path)})
    assert calls[0]["expected_chain"] == source["amendment_chain"]
    assert calls[0]["chain_receipt"] == Path(source["amendment_chain_receipt"])


def test_failed_preparation_preserves_partial_cache_and_attachment_before_retry(tmp_path, monkeypatch):
    from benchmarking.pmm_execution import ExecutionPolicy

    root, attached = tmp_path / "execution", tmp_path / "view/whole_protein_pretrain"
    attached.mkdir(parents=True)
    (attached / "partial_payload.pt").write_bytes(b"partial attached cache")
    view = SimpleNamespace(root=attached.parent, manifest_sha256=digest("view"))
    started = time.time() - 1
    policy = ExecutionPolicy(deadline_unix=started + 4000, max_total_seconds=4000,
                             allocation_started_unix=started)
    monkeypatch.setattr(phase3, "_runtime", lambda request, **kwargs: (source_identity(), view, policy))
    monkeypatch.setattr(phase3.views.View, "load", lambda *args: view)
    program = ("import pathlib,sys,json; p=pathlib.Path(sys.argv[1]); "
               "marker=json.loads((p/'pretrain_attachment.json').read_text()); "
               "assert marker['status']=='pending'; assert marker['identity']['attempt_root']==str(p); "
               "(p/'partial_cache.pt').write_bytes(b'partial cache'); raise SystemExit(7)")
    monkeypatch.setattr(phase3, "_child_command", lambda *args: [sys.executable, "-c", program,
        str(root / "runs/ecc30_pretrain_fold0_attempt1")])
    request = {"execution_root": str(root), "durable_root": str(tmp_path / "independent"),
               "session_id": "synthetic-preparation", "forecast_seconds": 20, "unit_timeout_seconds": 10,
               "python": sys.executable, "cap_prefix": []}
    with pytest.raises(ValueError, match="preserve its terminal attempt"):
        phase3.run_pretrain_preparation(request)
    state = json.loads((root / "execution_state.json").read_text())
    assert state["pending_unit"]["status"] == "failed_preparation"
    assert not state["pending_unit"]["persisted"]
    manifest = json.loads((root / state["pending_transfer"]["manifest_path"]).read_text())
    files = {item["path"] for item in manifest["files"]}
    assert "runs/ecc30_pretrain_fold0_attempt1/partial_cache.pt" in files
    assert "runs/ecc30_pretrain_fold0_attempt1/attachment_snapshot/partial_payload.pt" in files
    assert "runs/ecc30_pretrain_fold0_attempt1/pretrain_attachment.json" in files
    marker = root / "runs/ecc30_pretrain_fold0_attempt1/pretrain_attachment.json"
    preserved = phase3.verify_preserved_pretrain_attachment(marker, original_attempt=marker.parent)
    assert preserved["termination_proof"]["kind"] == "owned_process_group_reaped"


def provider_termination_proof(tmp_path, marker):
    identity = json.loads(marker.read_text())["identity"]
    path = tmp_path / "controller_terminal.json"
    path.write_text(json.dumps({"status": "EXITED", "runtime_status": "stopped",
        "session_id": identity["session_id"], "pod_id": identity["pod_id"]}))
    return {"kind": "provider_terminal", "session_id": identity["session_id"],
            "request_sha256": identity["request_sha256"], "receipt": {"path": str(path), "sha256": phase3._sha(path)}}


@pytest.fixture
def pending_attachment(tmp_path, monkeypatch):
    root, attached = tmp_path / "execution", tmp_path / "view/whole_protein_pretrain"
    attached.mkdir(parents=True)
    (attached / "payload.pt").write_bytes(b"cache bytes")
    view = SimpleNamespace(root=attached.parent, manifest_sha256=digest("view"))
    request = {"execution_root": str(root), "session_id": "pretrain-session", "pod_id": "synthetic-pod"}
    request_path = root / "commands/ecc30_pretrain_fold0_attempt1.json"
    phase3._write(request_path, request)
    marker = phase3.prepare_pretrain_attachment(root / "runs/ecc30_pretrain_fold0_attempt1",
        request=request, source=source_identity(), view=view, request_path=request_path)
    monkeypatch.setattr(phase3.views.View, "load", lambda *args: view)
    return marker, request, view


def test_pending_snapshot_blocks_execution_before_auto_ack(pending_attachment, monkeypatch):
    marker, request, _view = pending_attachment
    monkeypatch.setattr(phase3, "CampaignExecution", lambda *args, **kwargs:
                        pytest.fail("Pending snapshot must block before constructing or entering execution"))
    with pytest.raises(ValueError, match="pending"):
        phase3._execution(request, None)
    assert json.loads(marker.read_text())["status"] == "pending"


@pytest.mark.parametrize("case", ["alive", "foreign_session", "foreign_request", "unowned_parent", "running_provider"])
def test_attachment_restore_requires_owned_termination_evidence(pending_attachment, tmp_path, case):
    marker, _request, _view = pending_attachment
    proof = provider_termination_proof(tmp_path, marker)
    terminated = True
    if case == "alive":
        terminated = False
    elif case == "foreign_session":
        proof["session_id"] = "foreign"
    elif case == "foreign_request":
        proof["request_sha256"] = digest("foreign")
    elif case == "unowned_parent":
        proof = {**proof, "kind": "owned_process_group_reaped", "parent_pid": os.getpid()}
    else:
        receipt = Path(proof["receipt"]["path"])
        receipt.write_text(json.dumps({"status": "RUNNING", "session_id": proof["session_id"], "pod_id": "synthetic-pod"}))
        proof["receipt"]["sha256"] = phase3._sha(receipt)
    with pytest.raises(ValueError):
        phase3.restore_pretrain_attachment(marker, worker_terminated=terminated, termination_proof=proof)
    assert json.loads(marker.read_text())["status"] == "pending"
    assert not (marker.parent / "attachment_snapshot").exists()


def test_failed_snapshot_remains_pending_until_explicit_verified_recovery(pending_attachment, tmp_path, monkeypatch):
    marker, request, _view = pending_attachment
    proof = provider_termination_proof(tmp_path, marker)
    copytree = phase3.shutil.copytree

    def failed_copy(source, destination):
        destination.mkdir()
        (destination / "incomplete").write_bytes(b"partial")
        raise OSError("synthetic full disk")

    monkeypatch.setattr(phase3.shutil, "copytree", failed_copy)
    with pytest.raises(OSError, match="full disk"):
        phase3.restore_pretrain_attachment(marker, worker_terminated=True, termination_proof=proof)
    with pytest.raises(ValueError, match="pending"):
        phase3.require_restored_pretrain_attempts(Path(request["execution_root"]))
    monkeypatch.setattr(phase3.shutil, "copytree", copytree)
    restored = phase3.restore_pretrain_attachment(marker, worker_terminated=True, termination_proof=proof)
    assert restored["status"] == "restored"
    assert restored["files"] == [{"path": "payload.pt", "sha256": digest("cache bytes"), "size_bytes": 11}]
    phase3.require_restored_pretrain_attempts(Path(request["execution_root"]))


def test_preserved_snapshot_verifies_offline_and_refuses_tampering(pending_attachment, tmp_path, monkeypatch):
    marker, _request, view = pending_attachment
    proof = provider_termination_proof(tmp_path, marker)
    phase3.restore_pretrain_attachment(marker, worker_terminated=True, termination_proof=proof)
    original = marker.parent
    local = tmp_path / "independent-copy"
    shutil.copytree(original, local)
    shutil.rmtree(original)
    shutil.rmtree(view.root)
    monkeypatch.setattr(phase3.views.View, "load", lambda *args: pytest.fail("Offline verification opened the remote view"))
    copied_marker = local / phase3.PRETRAIN_ATTACHMENT_MARKER
    assert phase3.verify_preserved_pretrain_attachment(copied_marker, original_attempt=original)["status"] == "restored"
    (local / "attachment_snapshot/payload.pt").write_bytes(b"changed bytes")
    with pytest.raises(ValueError, match="inventory or bytes"):
        phase3.verify_preserved_pretrain_attachment(copied_marker, original_attempt=original)


def test_worker_refuses_foreign_or_restored_pending_marker(pending_attachment):
    marker, request, view = pending_attachment
    assert phase3._require_pretrain_worker_marker(marker.parent, request, source_identity(), view)["status"] == "pending"
    changed = {**request, "session_id": "foreign-session"}
    with pytest.raises(ValueError, match="parent's pending"):
        phase3._require_pretrain_worker_marker(marker.parent, changed, source_identity(), view)
    record = json.loads(marker.read_text())
    record["status"] = "restored"
    marker.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="parent's pending"):
        phase3._require_pretrain_worker_marker(marker.parent, request, source_identity(), view)
