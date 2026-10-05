"""Tests for the A3 acceptance rules of audit_v3_regression.py (synthetic reports; no replay)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import audit_v3_regression as audit  # noqa: E402
from benchmarking import pmm_ion_campaign as v2  # noqa: E402

PINS = ["unit_a", "unit_b"]
SOURCE = {"tested_src_tree_sha256": "a" * 64, "frozen_src_tree_sha256": "f" * 64, "frozen_tree_matches_pin": True}


def report(**overrides):
    base = {"source_at_start": dict(SOURCE), "source_at_end": dict(SOURCE),
            "replay": {name: {"passed": True} for name in PINS},
            "training": {"configs": {name: {"passed": True} for name in audit.TRAINING_CONFIG_IDS}}}
    base.update(overrides)
    return base


def test_complete_passing_audit_is_accepted():
    verdict = audit.acceptance(report(), pins=PINS)
    assert verdict["verdict"] == "accepted" and not verdict["failures"] and not verdict["missing"]
    assert verdict["tested_src_tree_sha256"] == "a" * 64


def test_a_subset_or_one_part_is_only_a_partial_diagnostic():
    only_replay = audit.acceptance(report(training={}), pins=PINS)
    assert only_replay["verdict"] == "partial diagnostic (not acceptance)" and only_replay["missing"]
    subset = audit.acceptance(report(replay={"unit_a": {"passed": True}}), pins=PINS)
    assert subset["verdict"].startswith("partial") and subset["missing"] == ["replay unit_b"]


def test_any_failure_or_source_change_fails_the_audit():
    failed = audit.acceptance(report(replay={"unit_a": {"passed": True}, "unit_b": {"passed": False}}), pins=PINS)
    assert failed["verdict"] == "failed" and failed["failures"] == ["replay unit_b: not passed"]
    changed = audit.acceptance(report(source_at_end={**SOURCE, "tested_src_tree_sha256": "b" * 64}), pins=PINS)
    assert changed["verdict"] == "failed" and "changed during the audit" in changed["failures"][0]
    unpinned = {**SOURCE, "frozen_tree_matches_pin": False}
    frozen = audit.acceptance(report(source_at_start=unpinned, source_at_end=unpinned), pins=PINS)
    assert frozen["verdict"] == "failed" and "pinned source hash" in frozen["failures"][0]
    unbound = audit.acceptance(report(source_at_start={}), pins=PINS)  # e.g. --accept after a source change
    assert unbound["verdict"] == "failed"
    partial_failure = audit.acceptance(report(training={"configs": {"x": {"passed": False}}}), pins=PINS)
    assert partial_failure["verdict"] == "failed"  # a failure outranks a missing part


def test_tree_hash_follows_the_campaign_source_rule():
    assert audit.tree_sha256(ROOT) == v2.source_tree_sha256()
