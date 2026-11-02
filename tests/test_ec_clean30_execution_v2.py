"""The operational extension cannot change frozen scientific or WP-13 inputs."""
import json

import pytest

from ec_clean30 import phase3
from test_ec_clean30_phase3 import execution_source, _git, _sha


@pytest.fixture
def automation_source(execution_source):
    repo, old, _, _ = execution_source
    commit = _git(repo, "rev-parse", "HEAD")
    for name in phase3.AUTOMATION_FILES:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name not in phase3.EXECUTION_FILES:
            path.write_text(f"# operational extension {name}\n")
    (repo / phase3.EXECUTION_FILES[0]).write_text("# versioned execution authentication\n")
    out = old.with_name("automation.json")
    phase3.make_automation_manifest(old, out, base_commit=commit, repo=repo)
    return repo, old, out


def test_extension_requires_publication_and_preserves_the_original_manifest(automation_source):
    repo, old, out = automation_source
    original = old.read_bytes()
    record = phase3.verify_execution_source(out, repo=repo, require_published=False)
    assert record["format"] == phase3.AUTOMATION_FORMAT
    assert set(record["source_sha256"]) == set(phase3.AUTOMATION_FILES)
    assert record["base_execution"]["sha256"] == _sha(old)
    with pytest.raises(phase3.Phase3Error, match="published"):
        phase3.verify_execution_source(out, repo=repo)
    _git(repo, "add", "src", "scripts", "docs")
    _git(repo, "commit", "--quiet", "-m", "Fixture operational extension")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    assert phase3.verify_execution_source(out, repo=repo)["published"]
    assert old.read_bytes() == original


@pytest.mark.parametrize("case", ["original_manifest", "base_hash", "base_commit", "timing_inputs", "chain",
                                 "execution_file", "frozen_source", "extra_pin", "missing_pin"])
def test_extension_refuses_unbound_source_or_replaced_inputs(automation_source, case):
    repo, old, out = automation_source
    record = json.loads(out.read_text())
    if case == "original_manifest":
        old.write_bytes(old.read_bytes() + b"\n")
    elif case == "base_hash":
        record["base_execution"]["sha256"] = "0" * 64
    elif case == "base_commit":
        record["base_execution"]["commit"] = "HEAD"
    elif case == "timing_inputs":
        record["timing_inputs_sha256"] = "0" * 64
    elif case == "chain":
        record["amendment_chain"] = ["0" * 64]
    elif case == "execution_file":
        (repo / "src/ec_clean30/phase4.py").write_text("changed\n")
    elif case == "frozen_source":
        (repo / "src/frozen.py").write_text("changed scientific source\n")
    elif case == "extra_pin":
        extra = repo / "src/unapproved.py"
        extra.write_text("# unapproved\n")
        record["source_sha256"]["src/unapproved.py"] = _sha(extra)
    elif case == "missing_pin":
        record["source_sha256"].pop("src/ec_clean30/workstation.py")
    out.write_text(json.dumps(record))
    with pytest.raises(phase3.Phase3Error):
        phase3.verify_execution_source(out, repo=repo, require_published=False)
