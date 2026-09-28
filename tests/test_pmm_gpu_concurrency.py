"""CPU-only tests for ownership, bounded subprocesses and diagnostic integrity."""
from __future__ import annotations

import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Batch, Data

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("concurrency_probe", ROOT / "benchmark_pmm_gpu_concurrency.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def options(**changes):
    return SimpleNamespace(**dict(dict(steps=100, warmup=10, batches=4, threads=2, max_seconds=600), **changes))


@pytest.mark.parametrize("changes", [{"steps": 0}, {"steps": 1001}, {"warmup": 0}, {"batches": 1},
                                     {"threads": 5}, {"max_seconds": float("nan")}, {"max_seconds": 901}])
def test_unbounded_or_invalid_options_rejected(changes):
    with pytest.raises(ValueError):
        probe.check_options(options(**changes))


def test_representative_full_batches_only():
    batches = [SimpleNamespace(num_graphs=16) for _ in range(9)] + [SimpleNamespace(num_graphs=4)]
    assert probe.select_batches(batches, 3) == [0, 4, 8]
    with pytest.raises(ValueError):
        probe.select_batches(batches, 10)


def test_campaign_lock_refuses_other_owner_and_preserves_state(tmp_path):
    original = {"pending_unit": {"persisted": True}, "active_child": None}
    state = tmp_path / "execution_state.json"
    state.write_text(json.dumps(original))
    before = state.read_bytes()
    with probe.campaign_lock(tmp_path):
        with (tmp_path / "execution.lock").open("a+") as competitor:
            with pytest.raises(BlockingIOError):
                fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            with probe.campaign_lock(tmp_path):
                pass
    with probe.campaign_lock(tmp_path):
        pass
    assert state.read_bytes() == before


@pytest.mark.parametrize("state", [{"active_child": {"pid": 1}}, {"pending_transfer": {"token": "x"}},
                                  {"pending_unit": {"status": "running", "persisted": False}}])
def test_unresolved_campaign_state_blocks_probe(tmp_path, state):
    (tmp_path / "execution_state.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="unresolved"):
        with probe.campaign_lock(tmp_path):
            pass


def test_wait_bounds_and_failed_child():
    with pytest.raises(TimeoutError):
        probe.wait_until(lambda: False, [], time.time() - 1)
    child = subprocess.Popen([sys.executable, "-c", "raise SystemExit(7)"])
    child.wait(timeout=5)
    with pytest.raises(RuntimeError, match="failed"):
        probe.wait_until(lambda: False, [child], time.time() + 3)


def test_cleanup_terminates_and_reaps_process_group():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    probe.terminate_processes([child])
    assert child.poll() is not None


def test_child_deadline_and_parent_identity(tmp_path):
    code = f"""import importlib.util, os, time
s=importlib.util.spec_from_file_location('probe', {str(ROOT / 'benchmark_pmm_gpu_concurrency.py')!r})
m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
m.set_child_lifetime({os.getpid()}, time.time()+0.1)
time.sleep(30)
"""
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=5)
    assert completed.returncode != 0 and b"TimeoutError" in completed.stderr


def test_worker_refuses_foreign_inherited_lock_before_loading_gpu(tmp_path, monkeypatch):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "execution.lock").touch()
    monkeypatch.setattr(probe, "set_child_lifetime", lambda *_: None)
    with (tmp_path / "different.lock").open("w") as unrelated:
        spec_path = tmp_path / "spec.json"
        spec_path.write_text(json.dumps({"parent_pid": os.getpid(), "deadline": time.time() + 5,
                                         "lock_fd": unrelated.fileno(), "campaign_root": str(campaign)}))
        with pytest.raises(ValueError, match="ownership lock"):
            probe.worker(spec_path)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 2)

    def forward(self, batch):
        logits = self.layer(batch.x)
        return {"loss": torch.nn.functional.cross_entropy(logits, batch.y)}


def test_real_backward_steps_are_finite_resettable_and_do_not_mutate_inputs():
    from audit_pmm_replay import graph_hash

    torch.manual_seed(42)
    model = TinyModel()
    original = {k: v.clone() for k, v in model.state_dict().items()}
    batch = Batch.from_data_list([Data(x=torch.tensor([[1., 2.]]), y=torch.tensor([i % 2])) for i in range(16)])
    before = graph_hash(batch)
    results = []
    for _ in range(2):
        model.load_state_dict(original)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        losses = probe.optimize_steps(model, [batch], optimizer, steps=3, device="cpu", clip=1.0)
        results.append((losses, {k: v.clone() for k, v in model.state_dict().items()}))
    assert before == graph_hash(batch)
    assert results[0][0] == results[1][0]
    assert all(torch.equal(results[0][1][k], results[1][1][k]) for k in original)
    assert any(not torch.equal(original[k], results[0][1][k]) for k in original)


def test_comparison_is_diagnostic_and_does_not_hide_large_difference():
    a = torch.tensor([1., 2.], dtype=torch.float64)
    same = probe.compare_vectors(a, a.clone(), atol=1e-5, rtol=1e-4)
    assert same["allclose"] and same["bitwise_equal"]
    changed = probe.compare_vectors(a, a + 0.1, atol=1e-5, rtol=1e-4)
    assert not changed["allclose"] and "never scientific" in changed["comparison_role"]
    with pytest.raises(ValueError):
        probe.compare_vectors(a, a * float("nan"), atol=1e-5, rtol=1e-4)


def test_plan_checks_snapshot_checkpoint_and_campaign_lock_binding(tmp_path, monkeypatch):
    import audit_pmm_replay as audit

    campaign = tmp_path / "campaign"
    run = campaign / "runs" / "fixture"
    prepared = campaign / "runtime" / "prepared"
    run.mkdir(parents=True)
    prepared.mkdir(parents=True)
    batches = [Batch.from_data_list([Data(x=torch.tensor([[1., 2.]])) for _ in range(16)]) for _ in range(4)]
    identity = {"family": "only_gvp", "readout": "none", "source_tree_sha256": "frozen"}
    config = dict(campaign_run_identity=json.dumps(identity), batch_size=16, grad_accum_steps=1,
                  use_amp=False, deterministic=False, lr_schedule="fixed", task="metal", learning_rate=0.001,
                  weight_decay=0.0001, grad_clip_norm=1.0)
    checkpoint = run / "best_model_checkpoint.pt"
    torch.save({"config": config, "normalization_stats": {"mean": 0.}}, checkpoint)
    torch.save({"schema": "fixture", "batches": batches}, prepared / "input_snapshot.pt")
    manifest = dict(status="predictive_inputs_matched", schema="fixture", campaign_run_identity=identity,
                    input_files={"original/path/best_model_checkpoint.pt": audit.file_sha(checkpoint)},
                    snapshot_sha256=audit.file_sha(prepared / "input_snapshot.pt"),
                    normalization_sha256=audit.stable_hash(audit.descriptor({"mean": 0.})),
                    batches=[{"fresh_sha256": audit.graph_hash(b)} for b in batches])
    (prepared / "input_manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(audit, "verify_source", lambda expected: expected)
    args = options(prepared_dir=prepared, checkpoint=checkpoint, campaign_root=campaign,
                   output_dir=campaign / "runtime" / "probe")
    plan = probe.prepare_plan(args)
    assert plan["source_tree_sha256"] == "frozen" and plan["certifies_fit"] is False
    assert not args.output_dir.exists()
    args.campaign_root = tmp_path
    with pytest.raises(ValueError, match="lock identity"):
        probe.prepare_plan(args)
    args.campaign_root = campaign
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Checkpoint differs"):
        probe.prepare_plan(args)
