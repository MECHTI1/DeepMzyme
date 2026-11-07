"""CPU-only view and lane identity gates; no model or provider invocation."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ec_clean30 import phase3, views

PRIVATE_RECEIPT = "DeepMzyme_Data/campaigns/ec_class_vs_clean30_2026-10/amendment2/dry_runs/fold3.json"


@pytest.fixture
def binding(tmp_path, monkeypatch):
    chain = pytest.importorskip("ec_clean30.amendment_chain")
    # The dry-run-v2 receipt is private and never on a pod: the repository root holds no copy of it.
    monkeypatch.setattr(phase3, "REPO_ROOT", tmp_path / "pod_repo")
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    original = receipts / views.DRY_RUN_RECEIPT_NAME
    original.write_text("original Freeze-1 receipt\n")
    original_sha = hashlib.sha256(original.read_bytes()).hexdigest()
    verified = {"dry_run_evidence": {
        "source_commit": "b" * 40,
        "receipts": {"3": {"path": PRIVATE_RECEIPT, "sha256": "d" * 64, "view_manifest_sha256": "a" * 64}},
        "in_view_receipts_unchanged": {"3": original_sha}}}
    calls = []
    monkeypatch.setattr(phase3, "authenticated_chain", lambda source: calls.append(source) or verified)
    view = SimpleNamespace(smoke=False, fold=3, receipts=receipts, manifest_sha256="a" * 64,
                           chain_means_manifest_sha256=lambda layer: layer)
    source = {"scientific_identity_sha256": "c" * 64, "frozen_views": {"3": {
        "view_manifest_sha256": view.manifest_sha256, "dry_run_receipt": {"sha256": original_sha},
        "chain_means": {layer: layer for layer in ("final", "block26")}}}}
    return view, source, verified, original, calls


def test_each_view_binding_rechecks_chain_and_both_receipts(binding):
    view, source, verified, original, calls = binding
    phase3._bind_view(view, source)
    phase3._bind_view(view, source)
    assert len(calls) == 2
    assert original.read_text() == "original Freeze-1 receipt\n"
    assert not phase3.REPO_ROOT.exists()


@pytest.mark.parametrize("fault", ["missing", "no_view_manifest", "other_view", "in_view"])
def test_view_refuses_missing_or_changed_amendment2_dry_run(binding, fault):
    view, source, verified, original, calls = binding
    receipt = verified["dry_run_evidence"]["receipts"]["3"]
    if fault == "missing":
        verified["dry_run_evidence"]["receipts"].clear()
    elif fault == "no_view_manifest":
        del receipt["view_manifest_sha256"]
    elif fault == "other_view":
        receipt["view_manifest_sha256"] = "e" * 64
    else:
        original.write_text("changed\n")
    with pytest.raises(ValueError):
        phase3._bind_view(view, source)


def _lanes():
    from test_ec_clean30_phase3 import _lane_records
    serial, parallel, limits, observed = _lane_records()
    for index, row in enumerate(serial + parallel):
        row.update(scientific_identity_sha256="b" * 64, fit_chain=["c" * 64, "d" * 64],
                   execution_manifest_sha256=str(index) * 64)
    return serial, parallel, limits, observed


def test_lane_gate_accepts_execution_update_under_same_science():
    serial, parallel, limits, observed = _lanes()
    gate = phase3.lane_gate(serial, parallel, batch_wall_seconds=270, limits=limits, observed=observed)
    assert gate["status"] == "passed"
    assert gate["scientific_identity_sha256"] == serial[0]["scientific_identity_sha256"]
    assert gate["fit_chain"] == serial[0]["fit_chain"]
    assert "execution_manifest_sha256" not in gate


@pytest.mark.parametrize("fault", ["scientific_identity_sha256", "fit_chain"])
def test_lane_gate_refuses_mixed_science(fault):
    serial, parallel, limits, observed = _lanes()
    parallel[-1][fault] = "e" * 64 if fault == "scientific_identity_sha256" else ["c" * 64]
    with pytest.raises(phase3.Phase3Error, match="scientific identities"):
        phase3.lane_gate(serial, parallel, batch_wall_seconds=270, limits=limits, observed=observed)
