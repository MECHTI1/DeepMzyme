"""Tests for the v3 transfer bundle (pmm_v3_bundle.py)."""

from __future__ import annotations

import hashlib
import io
import json
import re
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


CAMPAIGN = "pmm_ion_metal_v3"
ALIAS, ARCHIVE = f"docs/campaigns/{CAMPAIGN}", f"docs/archive/campaigns/{CAMPAIGN}"


def _archived_and_verified(root: Path = ROOT) -> bool:
    """False while the specification is tracked at its original path (campaign open). True only for the verified
    closure layout (log v3-052): the registered alias to the archive destination, a closed README and every pinned
    hash. Anything else, such as an unexpectedly missing specification, fails the test."""
    spec = bundle.SPEC_FILES[0]
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "--", spec], cwd=root, capture_output=True)
    if tracked.returncode == 0:
        return False
    registry = root / "docs" / "archive" / "frozen_moves.json"
    assert registry.is_file(), f"{spec} is not tracked and no archive registry exists"
    moves = [m for m in json.loads(registry.read_text())["moves"] if m["campaign"] == CAMPAIGN]
    assert len(moves) == 1, f"{CAMPAIGN} is not registered exactly once"
    move = moves[0]
    assert move["alias"] == {"path": ALIAS, "target": ARCHIVE}, f"unexpected alias {move['alias']}"
    alias = root / ALIAS
    assert alias.is_symlink() and alias.resolve() == (root / ARCHIVE).resolve(), \
        "the alias is not the registered symlink"
    readme = root / ARCHIVE / "README.md"
    assert readme.is_file(), "the archived README is missing"
    status = next((line for line in readme.read_text().split("\n")[:8] if line.startswith("Status:")), "")
    assert status.startswith("Status: closed ("), f"the archived campaign is not closed: {status!r}"
    pinned = {entry["origin"]: entry for entry in move["pinned"]}
    assert set(bundle.SPEC_FILES) <= set(pinned), "a bundled specification is not pinned"
    for entry in move["pinned"]:
        path = root / entry["path"]
        assert path.is_file() and not path.is_symlink(), f"{entry['path']} is missing"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], f"{entry['path']} changed"
        assert (root / entry["origin"]).resolve() == path.resolve(), f"{entry['origin']} does not reach the archive"
    return True


def _closed_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    archive = root / ARCHIVE
    archive.mkdir(parents=True)
    (archive / "README.md").write_text("# v3\n\nStatus: closed (2026-10-13 test)\n")
    pinned = []
    for name in bundle.SPEC_FILES:
        path = archive / Path(name).name
        path.write_bytes(name.encode())
        pinned.append({"path": f"{ARCHIVE}/{path.name}", "origin": name,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (root / "docs" / "campaigns").mkdir()
    (root / ALIAS).symlink_to(f"../archive/campaigns/{CAMPAIGN}")
    (root / "docs" / "archive" / "frozen_moves.json").write_text(json.dumps(
        {"moves": [{"campaign": CAMPAIGN, "pinned": pinned, "alias": {"path": ALIAS, "target": ARCHIVE}}]}))
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    return root


def test_closure_guard_accepts_only_the_verified_archive(tmp_path):
    root = _closed_repo(tmp_path)
    assert _archived_and_verified(root) is True
    open_repo = tmp_path / "open"
    (open_repo / Path(bundle.SPEC_FILES[0]).parent).mkdir(parents=True)
    (open_repo / bundle.SPEC_FILES[0]).write_text("{}")
    subprocess.run(["git", "init", "-q"], cwd=open_repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=open_repo, check=True)
    assert _archived_and_verified(open_repo) is False


@pytest.mark.parametrize("damage, message", [
    (lambda root: (root / "docs/archive/frozen_moves.json").unlink(), "no archive registry"),
    (lambda root: (root / ARCHIVE / Path(bundle.SPEC_FILES[1]).name).unlink(), "is missing"),
    (lambda root: (root / ARCHIVE / Path(bundle.SPEC_FILES[0]).name).write_text("edited"), "changed"),
    (lambda root: (root / ARCHIVE / "README.md").write_text("Status: active (2026-10-13 test)\n"), "not closed"),
    (lambda root: (root / ALIAS).unlink(), "not the registered symlink"),
])
def test_closure_guard_fails_on_an_unverified_archive(tmp_path, damage, message):
    root = _closed_repo(tmp_path)
    damage(root)
    with pytest.raises(AssertionError, match=message):
        _archived_and_verified(root)


def test_build_refuses_after_closure(tmp_path):
    """After the v3 closure, bundles are built only from tag pmm-ion-metal-v3-final: build refuses before writing."""
    if not _archived_and_verified():
        pytest.skip("the campaign is open; the refusal applies only after the archive move")
    out = tmp_path / "bundle"
    with pytest.raises(bundle.BundleError, match=re.escape(f"{bundle.SPEC_FILES[0]} is not committed")):
        bundle.build(out, folds=tmp_path, a3_acceptance=tmp_path / "a3_report.json")
    assert not out.exists()


@pytest.mark.skipif(_bundled_files_dirty(), reason="bundle ships committed code only; commit first")
def test_build_and_apply_round_trip_and_refusals(tmp_path):
    if _archived_and_verified():
        pytest.skip("pmm_ion_metal_v3 is closed and verified (docs/archive/frozen_moves.json); bundles build only "
                    "from tag pmm-ion-metal-v3-final, where this round trip passed")
    folds = tmp_path / "folds" / "v3-test"
    folds.mkdir(parents=True)
    membership = b"source_uid,fold\nu1,0\n"
    for name in bundle.FOLD_FILES:
        (folds / name).write_bytes(membership if name == "fold_membership.csv" else b"{}")
    (folds / "fold_receipt.json").write_text(json.dumps({
        "accepted": True, "outputs_sha256": {"fold_membership.csv": bundle.sha256_bytes(membership)}}))
    a3 = tmp_path / "a3_report.json"
    a3.write_text(json.dumps({"acceptance": {"verdict": "accepted", "tested_src_tree_sha256": "0" * 64}}))
    with pytest.raises(bundle.BundleError, match="A3 accepted source tree"):
        bundle.build(tmp_path / "refused", folds=folds, a3_acceptance=a3)
    with pytest.raises(bundle.BundleError, match="--a3-acceptance"):
        bundle.build(tmp_path / "refused2", folds=folds)
    a3.write_text(json.dumps({"acceptance": {"verdict": "accepted", "tested_src_tree_sha256": source_tree_sha256()}}))
    manifest = bundle.build(tmp_path / "bundle", folds=folds, a3_acceptance=a3)
    assert manifest["a3_acceptance"]["tested_src_tree_sha256"] == source_tree_sha256()
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


def test_runner_files_are_bundled_and_hashed_like_the_runner():
    sys.path[:0] = [str(ROOT), str(ROOT / "src")]
    import pmm_v3_campaign as v3

    assert bundle.RUNNER_FILES == v3.RUNNER_FILES
    assert set(bundle.RUNNER_FILES) <= set(bundle.RUNTIME_FILES) and "pmm_v3_probes.py" in bundle.RUNTIME_FILES
    for workstation_only in ("pmm_v3_host_pull.py", "pmm_v3_step_b.py"):
        assert not bundle.allowed(workstation_only)


def test_a3_acceptance_must_be_accepted_and_bound_to_the_bundled_tree(tmp_path):
    report = tmp_path / "a3.json"
    for payload, message in (({"acceptance": {"verdict": "failed", "tested_src_tree_sha256": "a"}}, "not an accepted"),
                             ({"acceptance": {"verdict": "partial diagnostic (not acceptance)"}}, "not an accepted"),
                             ({"acceptance": {"verdict": "accepted", "tested_src_tree_sha256": "b"}}, "A3 accepted")):
        report.write_text(json.dumps(payload))
        with pytest.raises(bundle.BundleError, match=message):
            bundle.require_a3_acceptance(report, "a")
    report.write_text(json.dumps({"acceptance": {"verdict": "accepted", "tested_src_tree_sha256": "a"}}))
    assert bundle.require_a3_acceptance(report, "a")["tested_src_tree_sha256"] == "a"


def hand_built_bundle(directory: Path, *, runner_marker: bytes = b"") -> dict:
    """A bundle written without git, so a later code drop can be applied next to the first one."""
    directory.mkdir(parents=True)
    files = {name: b"# runner " + name.encode() + runner_marker for name in bundle.RUNNER_FILES}
    files.update({bundle.SPEC_FILES[0]: b"{}", "src/train.py": b"print('train')\n", "pmm_v3_bundle.py": b"# tool\n"})
    fold_files = {name: (b"source_uid,fold\nu1,0\n" if name == "fold_membership.csv" else b"{}")
                  for name in bundle.FOLD_FILES}
    for archive, root, content in (("v3_code.tar.gz", bundle.CODE_ROOT, files), ("v3_folds.tar.gz", "v3-test", fold_files)):
        with tarfile.open(directory / archive, "w:gz") as tar:
            for name, data in content.items():
                info = tarfile.TarInfo(f"{root}/{name}")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    manifest = {"git_commit": "0" * 40, "code_root": bundle.CODE_ROOT,
                "code": {"path": "v3_code.tar.gz", "sha256": bundle.sha256_file(directory / "v3_code.tar.gz"),
                         "members": sorted(files)},
                "folds": {"path": "v3_folds.tar.gz", "sha256": bundle.sha256_file(directory / "v3_folds.tar.gz"),
                          "fold_set": "v3-test",
                          "fold_membership_sha256": bundle.sha256_bytes(fold_files["fold_membership.csv"])},
                "source_tree_sha256": bundle.tree_sha256(files), "runner_sha256": bundle.runner_sha256(files),
                "spec_sha256": bundle.sha256_bytes(files[bundle.SPEC_FILES[0]])}
    (directory / "v3_bundle_manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_a_later_code_drop_goes_to_its_own_directory_and_reuses_the_verified_fold_set(tmp_path):
    first = hand_built_bundle(tmp_path / "first")
    later = hand_built_bundle(tmp_path / "later", runner_marker=b" (extension 1)")
    vm = tmp_path / "vm"
    bundle.apply(tmp_path / "first", code_dest=vm / "DeepMzyme_v3", folds_parent=vm / "folds")
    original = {p: p.read_bytes() for p in sorted((vm / "DeepMzyme_v3").rglob("*")) if p.is_file()}
    with pytest.raises(bundle.BundleError, match="exists"):  # the default never touches an existing fold set
        bundle.apply(tmp_path / "later", code_dest=vm / "DeepMzyme_v3_ext1", folds_parent=vm / "folds")
    with pytest.raises(bundle.BundleError, match="exists"):  # nor an existing code directory
        bundle.apply(tmp_path / "later", code_dest=vm / "DeepMzyme_v3", folds_parent=vm / "folds",
                     reuse_existing_folds=True)
    with pytest.raises(bundle.BundleError, match="no fold set to reuse"):
        bundle.apply(tmp_path / "later", code_dest=vm / "DeepMzyme_v3_ext1", folds_parent=vm / "elsewhere",
                     reuse_existing_folds=True)
    fold_file = vm / "folds" / "v3-test" / "fold_membership.csv"
    kept = fold_file.read_bytes()
    fold_file.write_bytes(kept + b"u2,1\n")
    with pytest.raises(bundle.BundleError, match="existing fold set differs"):
        bundle.apply(tmp_path / "later", code_dest=vm / "DeepMzyme_v3_ext1", folds_parent=vm / "folds",
                     reuse_existing_folds=True)
    assert not (vm / "DeepMzyme_v3_ext1").exists()  # refused before anything was written
    fold_file.write_bytes(kept)
    receipt = bundle.apply(tmp_path / "later", code_dest=vm / "DeepMzyme_v3_ext1", folds_parent=vm / "folds",
                           reuse_existing_folds=True)
    assert receipt["folds_reused"] is True and receipt["runner_sha256"] == later["runner_sha256"]
    assert later["runner_sha256"] != first["runner_sha256"] and later["source_tree_sha256"] == first["source_tree_sha256"]
    assert {p: p.read_bytes() for p in sorted((vm / "DeepMzyme_v3").rglob("*")) if p.is_file()} == original
    assert fold_file.read_bytes() == kept


def test_a_later_bundle_must_keep_its_parents_source_specification_and_folds(tmp_path):
    parent = hand_built_bundle(tmp_path / "parent")
    same = dict(src_sha=parent["source_tree_sha256"], spec_sha=parent["spec_sha256"],
                fold_sha=parent["folds"]["fold_membership_sha256"])
    bound = bundle.require_same_campaign_inputs(tmp_path / "parent" / "v3_bundle_manifest.json", **same)
    assert bound == {"manifest_sha256": bundle.sha256_file(tmp_path / "parent" / "v3_bundle_manifest.json"),
                     "git_commit": parent["git_commit"], "runner_sha256": parent["runner_sha256"]}
    for key, label in (("src_sha", "source_tree_sha256"), ("spec_sha", "spec_sha256"),
                       ("fold_sha", "fold_membership_sha256")):
        with pytest.raises(bundle.BundleError, match=label):
            bundle.require_same_campaign_inputs(tmp_path / "parent" / "v3_bundle_manifest.json",
                                                **{**same, key: "f" * 64})
