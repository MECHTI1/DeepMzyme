"""Tests for the v3 transfer bundle (pmm_v3_bundle.py)."""

from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import pmm_v3_bundle as bundle  # noqa: E402
from benchmarking.pmm_ion_campaign import source_tree_sha256  # noqa: E402


def test_held_out_and_unlisted_paths_never_travel():
    for name in ("prepare_training_and_test_set/pinmymetal_files/classmodel_test_set/x.csv", "src/test/a.py",
                 "data/site_crosswalk.csv", "tests/test_v3_bundle.py", "README.md", "docs/DATASETS.md"):
        assert not bundle.allowed(name), name
    for name in ("src/train.py", "src/training/run.py", "scripts/run.sh", "pmm_v3_campaign.py",
                 "docs/campaigns/pmm_ion_metal_v3/assessment_spec.json"):
        assert bundle.allowed(name), name


def test_tree_hash_follows_the_campaign_rule():
    tracked = subprocess.run(["git", "ls-files", "-z", "src", "scripts"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.split("\0")
    files = {n: (ROOT / n).read_bytes() for n in tracked if n and (ROOT / n).is_file()}
    assert bundle.tree_sha256(files) == source_tree_sha256()


def _bundled_files_dirty() -> bool:
    names = [n for n in subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True,
                                       check=True).stdout.split("\0") if n and bundle.allowed(n)]
    names += [n for n in (*bundle.RUNTIME_FILES, *bundle.SPEC_FILES) if n not in names]
    return bool(subprocess.run(["git", "status", "--porcelain", "--", *names], cwd=ROOT, capture_output=True,
                               text=True, check=True).stdout.strip())


@pytest.mark.skipif(_bundled_files_dirty(), reason="bundle ships committed code only; commit first")
def test_build_and_apply_round_trip_and_refusals(tmp_path):
    folds = tmp_path / "folds" / "v3-test"
    folds.mkdir(parents=True)
    membership = b"source_uid,fold\nu1,0\n"
    for name in bundle.FOLD_FILES:
        (folds / name).write_bytes(membership if name == "fold_membership.csv" else b"{}")
    (folds / "fold_receipt.json").write_text(json.dumps({
        "accepted": True, "outputs_sha256": {"fold_membership.csv": bundle.sha256_bytes(membership)}}))
    manifest = bundle.build(tmp_path / "bundle", folds=folds)
    assert manifest["source_tree_sha256"] == source_tree_sha256()
    with tarfile.open(tmp_path / "bundle" / "v3_code.tar.gz") as tar:
        names = tar.getnames()
    assert all(not any(p in bundle.FORBIDDEN_PARTS for p in Path(n).parts) for n in names)
    receipt = bundle.apply(tmp_path / "bundle", code_dest=tmp_path / "vm" / "DeepMzyme_v3",
                           folds_parent=tmp_path / "vm" / "folds")
    assert receipt["source_tree_sha256"] == manifest["source_tree_sha256"]
    assert (tmp_path / "vm" / "folds" / "v3-test" / "fold_membership.csv").read_bytes() == membership
    with pytest.raises(bundle.BundleError, match="exists"):
        bundle.apply(tmp_path / "bundle", code_dest=tmp_path / "vm" / "DeepMzyme_v3", folds_parent=tmp_path / "other")
    archive = tmp_path / "bundle" / "v3_code.tar.gz"
    archive.write_bytes(archive.read_bytes() + b"x")
    with pytest.raises(bundle.BundleError, match="code archive changed"):
        bundle.apply(tmp_path / "bundle", code_dest=tmp_path / "vm2", folds_parent=tmp_path / "f2")
