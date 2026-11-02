"""Provider qualification extends, rather than replaces, the published execution lineage."""
import json

import pytest

from ec_clean30 import phase3
from test_ec_clean30_execution_v2 import automation_source
from test_ec_clean30_phase3 import execution_source, _git, _sha


@pytest.fixture
def runpod_source(automation_source):
    repo, original, automation = automation_source
    _git(repo, "add", "src", "scripts", "docs")
    _git(repo, "commit", "--quiet", "-m", "Fixture published automation")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    commit = _git(repo, "rev-parse", "HEAD")
    for name in set(phase3.RUNPOD_FILES) - set(phase3.AUTOMATION_FILES):
        (repo / name).write_text(f"# provider qualification {name}\n")
    (repo / phase3.EXECUTION_FILES[0]).write_text("# versioned provider authentication\n")
    out = automation.with_name("runpod.json")
    phase3.make_runpod_manifest(automation, out, base_commit=commit, repo=repo)
    return repo, original, automation, out


def test_provider_extension_keeps_both_published_predecessors_and_requires_publication(runpod_source):
    repo, original, automation, out = runpod_source
    prior = [path.read_bytes() for path in (original, automation)]
    record = phase3.verify_execution_source(out, repo=repo, require_published=False)
    assert record["format"] == phase3.RUNPOD_FORMAT
    assert set(record["source_sha256"]) == set(phase3.RUNPOD_FILES)
    assert record["base_automation"]["sha256"] == _sha(automation)
    assert record["hardware_profiles"] == ["gcp_l4", "runpod_l4", "runpod_rtx4090"]
    with pytest.raises(phase3.Phase3Error, match="published"):
        phase3.verify_execution_source(out, repo=repo)
    _git(repo, "add", "src", "scripts", "docs")
    _git(repo, "commit", "--quiet", "-m", "Fixture provider extension")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    assert phase3.verify_execution_source(out, repo=repo)["published"]
    assert [path.read_bytes() for path in (original, automation)] == prior


@pytest.mark.parametrize("case", ["wp13_bytes", "automation_bytes", "base_hash", "base_commit", "timing_inputs",
                                 "chain", "base_execution", "provider_source", "frozen_source", "extra_pin",
                                 "missing_pin", "hardware_profiles"])
def test_provider_extension_refuses_tampered_lineage_inputs_or_source(runpod_source, case):
    repo, original, automation, out = runpod_source
    record = json.loads(out.read_text())
    if case in {"wp13_bytes", "automation_bytes"}:
        path = original if case == "wp13_bytes" else automation
        path.write_bytes(path.read_bytes() + b"\n")
    elif case == "base_hash":
        record["base_automation"]["sha256"] = "0" * 64
    elif case == "base_commit":
        record["base_automation"]["commit"] = "HEAD"
    elif case == "timing_inputs":
        record["timing_inputs_sha256"] = "0" * 64
    elif case == "chain":
        record["amendment_chain"] = ["0" * 64]
    elif case == "base_execution":
        record["base_execution"]["sha256"] = "0" * 64
    elif case == "provider_source":
        (repo / "src/ec_clean30/runpod_controller.py").write_text("unrecorded provider change\n")
    elif case == "frozen_source":
        (repo / "src/frozen.py").write_text("changed scientific source\n")
    elif case == "extra_pin":
        extra = repo / "src/unapproved.py"
        extra.write_text("# unapproved\n")
        record["source_sha256"]["src/unapproved.py"] = _sha(extra)
    elif case == "missing_pin":
        record["source_sha256"].pop("src/ec_clean30/container_caps.py")
    else:
        record["hardware_profiles"].append("unreviewed_gpu")
    out.write_text(json.dumps(record))
    with pytest.raises(phase3.Phase3Error):
        phase3.verify_execution_source(out, repo=repo, require_published=False)


@pytest.mark.parametrize("profile", ["runpod_l4", "runpod_rtx4090"])
def test_runpod_profile_cannot_use_an_older_execution_manifest(profile):
    for source in ({}, {"format": phase3.AUTOMATION_FORMAT, "hardware_profiles": [profile]}):
        with pytest.raises(phase3.Phase3Error, match="authenticated provider extension"):
            phase3._approved_runtime_profile(source, profile)
    assert phase3._approved_runtime_profile(
        {"format": phase3.RUNPOD_FORMAT, "hardware_profiles": [profile]}, profile)["name"] == profile
    assert phase3._approved_runtime_profile({}, "gcp_l4")["device_name"] == "NVIDIA L4"
