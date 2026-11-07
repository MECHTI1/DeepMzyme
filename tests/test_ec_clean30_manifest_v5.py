"""CPU contracts for the receipt-backed execution manifest and scientific record identity."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ec_clean30 import phase3  # noqa: E402
import ec_clean30  # noqa: E402


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


@pytest.fixture
def v5_world(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    base = repo / phase3.freeze.CAMPAIGN_DOCS_DIR / "freeze1.json"
    base.parent.mkdir(parents=True)
    base.write_text(json.dumps({"views": {"0": {"view_manifest_sha256": digest("view")}}, "pins": {}}))
    code = repo / "src/ec_clean30/phase3.py"
    code.parent.mkdir(parents=True)
    code.write_text("# synthetic execution source\n")
    locks = repo / "runtime_locks.json"
    locks.write_text(json.dumps({"format": "deepmzyme_ec_clean30_runtime_locks_v1", "runtimes": {}}))
    chain = [digest("link one"), digest("link two")]
    verified = {"chain": chain, "fit_chain": chain, "tip_sequence": 2, "published": True,
                "scientific_identity_sha256": digest("science"),
                "execution_file_set": [code.relative_to(repo).as_posix()], "links": []}
    calls = []

    def check(receipt, *, repo, expected_chain=None):
        calls.append((Path(receipt), repo, expected_chain))
        if expected_chain is not None and list(expected_chain) != verified["chain"]:
            raise ValueError("chain differs")
        return copy.deepcopy(verified)

    monkeypatch.setattr(ec_clean30, "amendment_chain", SimpleNamespace(check_chain_receipt=check), raising=False)
    monkeypatch.setattr(ec_clean30, "amendment_rules", SimpleNamespace(
        FREEZE1_SHA256=phase3._sha(base), link_path=lambda n: f"amendment{n}/amendment.json",
        partition=lambda *args: {}), raising=False)
    monkeypatch.setattr(phase3, "_git", lambda repo, *args: "" if args[0] == "status" else (
        code.relative_to(repo).as_posix() if args[0] == "ls-files" else "f" * 40))
    receipt = tmp_path / "session_chain_receipt.json"
    receipt.write_text("{}")
    document = {"format": phase3.EXECUTION_V5_FORMAT, "campaign_id": phase3.CAMPAIGN_ID,
                "freeze1_sha256": phase3._sha(base), "freeze1b_sha256": phase3.FREEZE1B_SHA256,
                "amendment_chain": chain, "amendment_path": "amendment2/amendment.json",
                **phase3.source_identity(verified), "execution_file_set": verified["execution_file_set"],
                "source_sha256": {code.relative_to(repo).as_posix(): phase3._sha(code)},
                "runtime_locks": {"path": "runtime_locks.json", "sha256": phase3._sha(locks)},
                "hardware_profiles": sorted(name for name in phase3.GPU_PROFILES if name.startswith("runpod_"))}
    manifest = repo / "execution.json"
    manifest.write_text(json.dumps(document))
    return repo, manifest, receipt, document, verified, calls


def verify(world, **kwargs):
    repo, manifest, receipt, *_ = world
    return phase3.verify_execution_source(manifest, repo=repo, require_published=False,
                                         chain_receipt_path=receipt, **kwargs)


def test_manifest_v5_checks_receipt_with_known_chain_and_exposes_all_fold_views(v5_world):
    result = verify(v5_world)
    repo, _, receipt, document, _, calls = v5_world
    assert calls == [(receipt, repo, document["amendment_chain"])]
    assert result["frozen_views"]["0"] == result["frozen_fold0_view"]
    assert result["amendment_chain_receipt"] == str(receipt)
    assert phase3.source_identity(result) == phase3.source_identity(document)
    phase3.authenticated_chain(result, repo=repo)
    assert len(calls) == 2  # Each per-unit binding rechecks the receipt; there is no cached full verification.


@pytest.mark.parametrize("case", ["short_chain", "short_tip", "unpublished_chain", "identity", "fit_chain",
                                  "source_bytes", "closed_set", "hardware", "tip_path", "runtime_locks"])
def test_manifest_v5_refuses_invalid_or_stale_bindings(v5_world, case):
    repo, manifest, _, document, verified, _ = v5_world
    if case == "short_chain":
        document["amendment_chain"] = document["amendment_chain"][:1]
    elif case == "short_tip":
        verified["tip_sequence"] = 1
    elif case == "unpublished_chain":
        verified["published"] = False
    elif case == "identity":
        document["scientific_identity_sha256"] = digest("foreign science")
    elif case == "fit_chain":
        document["fit_chain"] = [digest("foreign fit"), digest("second foreign fit")]
    elif case == "source_bytes":
        (repo / document["execution_file_set"][0]).write_text("# changed\n")
    elif case == "closed_set":
        document["execution_file_set"].append("src/unregistered.py")
    elif case == "hardware":
        document["hardware_profiles"].append("gcp_l4")
    elif case == "tip_path":
        document["amendment_path"] = "amendment1/amendment.json"
    else:
        (repo / "runtime_locks.json").write_text("{}")
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        verify(v5_world)


def test_manifest_v5_requires_receipt_and_expected_chain(v5_world):
    repo, manifest, *_ = v5_world
    with pytest.raises(ValueError, match="receipt"):
        phase3.verify_execution_source(manifest, repo=repo, require_published=False)
    with pytest.raises(ValueError, match="expected chain"):
        verify(v5_world, expected_chain=[digest("another one"), digest("another two")])


def test_scientific_records_survive_execution_only_manifest_changes(v5_world):
    source = verify(v5_world)
    record = {**phase3.source_identity(source), "execution_manifest_sha256": digest("old manifest")}
    phase3.require_same_scientific_identity(record, {**source, "execution_manifest_sha256": digest("new manifest")})
    record["fit_chain"] = record["fit_chain"][:1]
    with pytest.raises(ValueError, match="scientific identity or fit chain"):
        phase3.require_same_scientific_identity(record, source)


def test_historical_records_keep_their_exact_execution_binding():
    source = {"execution_manifest_sha256": digest("old manifest")}
    assert phase3.source_identity(source) == source
    with pytest.raises(ValueError):
        phase3.require_same_scientific_identity({"execution_manifest_sha256": digest("other")}, source)


def test_git_subprocess_environment_strips_every_inherited_git_variable(monkeypatch):
    monkeypatch.setenv("GIT_DIR", "/foreign/.git")
    monkeypatch.setenv("GIT_WORK_TREE", "/foreign")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("KEEP_ME", "yes")
    assert not any(key.startswith("GIT_") for key in phase3.git_environment())
    assert phase3.git_environment()["KEEP_ME"] == "yes"
    assert phase3.clean_repository_environment({"GIT_INDEX_FILE": "/foreign/index", "PATH": "/bin"}) == {"PATH": "/bin"}
